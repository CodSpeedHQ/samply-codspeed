use std::path::{Path, PathBuf};
use std::sync::Arc;

use fxprof_processed_profile::{
    LibraryHandle, MarkerTiming, Profile, Symbol, SymbolTable, ThreadHandle,
};
use linux_perf_data::jitdump::{
    JitDumpReader, JitDumpRecord, JitDumpRecordHeader, JitDumpRecordType,
};

use super::jit_category_manager::JitCategoryManager;
use super::jit_function_add_marker::JitFunctionAddMarker;
use super::jit_function_recycler::JitFunctionRecycler;
use super::lib_mappings::{
    LibMappingAdd, LibMappingInfo, LibMappingMove, LibMappingOp, LibMappingOpQueue,
};
use super::timestamp_converter::TimestampConverter;
use super::utils::open_file_with_fallback;

#[derive(Debug)]
pub struct JitDumpManager {
    pending_jitdump_paths: Vec<(ThreadHandle, PathBuf, Vec<PathBuf>)>,
    processors: Vec<SingleJitDumpProcessor>,
    unlink_after_open: bool,
    should_emit_jit_markers: bool,
}

impl JitDumpManager {
    pub fn new(unlink_after_open: bool, should_emit_jit_markers: bool) -> Self {
        JitDumpManager {
            pending_jitdump_paths: Vec::new(),
            processors: Vec::new(),
            unlink_after_open,
            should_emit_jit_markers,
        }
    }

    pub fn add_jitdump_path(
        &mut self,
        thread: ThreadHandle,
        path: impl Into<PathBuf>,
        lookup_dirs: Vec<PathBuf>,
    ) {
        self.pending_jitdump_paths
            .push((thread, path.into(), lookup_dirs));
    }

    pub fn process_pending_records(
        &mut self,
        jit_category_manager: &mut JitCategoryManager,
        profile: &mut Profile,
        mut recycler: Option<&mut JitFunctionRecycler>,
        timestamp_converter: &TimestampConverter,
    ) {
        self.pending_jitdump_paths
            .retain_mut(|(thread, path, lookup_dirs)| {
                fn jitdump_reader_for_path(
                    path: &Path,
                    lookup_dirs: &[PathBuf],
                    unlink_after_open: bool,
                ) -> Option<(JitDumpReader<std::fs::File>, PathBuf)> {
                    let (file, path) = open_file_with_fallback(path, lookup_dirs).ok()?;
                    let reader = JitDumpReader::new(file).ok()?;
                    if unlink_after_open {
                        std::fs::remove_file(&path).ok()?;
                    }
                    Some((reader, path))
                }
                let Some((reader, actual_path)) =
                    jitdump_reader_for_path(path, lookup_dirs, self.unlink_after_open)
                else {
                    return true;
                };
                let lib_handle = crate::shared::utils::lib_handle_for_jitdump(
                    &actual_path,
                    reader.header(),
                    profile,
                );
                self.processors
                    .push(SingleJitDumpProcessor::new(reader, lib_handle, *thread));
                false // "Do not retain", i.e. remove from pending_jitdump_paths
            });

        for jitdump in &mut self.processors {
            jitdump.process_pending_records(
                jit_category_manager,
                profile,
                recycler.as_deref_mut(),
                timestamp_converter,
                self.should_emit_jit_markers,
            );
        }
    }

    pub fn finish(
        mut self,
        jit_category_manager: &mut JitCategoryManager,
        profile: &mut Profile,
        recycler: Option<&mut JitFunctionRecycler>,
        timestamp_converter: &TimestampConverter,
    ) -> Vec<LibMappingOpQueue> {
        self.process_pending_records(jit_category_manager, profile, recycler, timestamp_converter);
        self.processors
            .into_iter()
            .map(|processor| processor.finish(profile))
            .collect()
    }
}

#[derive(Debug)]
struct SingleJitDumpProcessor {
    /// Some() until a JIT_CODE_CLOSE record is encountered.
    reader: Option<JitDumpReader<std::fs::File>>,
    lib_handle: LibraryHandle,
    lib_mapping_ops: LibMappingOpQueue,
    symbols: Vec<Symbol>,
    thread_handle: ThreadHandle,

    /// The relative_address of the next JIT function.
    ///
    /// We define the relative address space for Jitdump files as follows:
    /// Pretend that all JIT code is located in sequence, without gaps, in
    /// the order of JIT_CODE_LOAD entries in the file. A given JIT function's
    /// relative address is the sum of the `code_size`s of all the `JIT_CODE_LOAD`
    /// entries that came before it in the file.
    cumulative_address: u32,
}

impl SingleJitDumpProcessor {
    pub fn new(
        reader: JitDumpReader<std::fs::File>,
        lib_handle: LibraryHandle,
        thread_handle: ThreadHandle,
    ) -> Self {
        Self {
            reader: Some(reader),
            lib_handle,
            lib_mapping_ops: Default::default(),
            symbols: Default::default(),
            thread_handle,
            cumulative_address: 0,
        }
    }

    pub fn process_pending_records(
        &mut self,
        jit_category_manager: &mut JitCategoryManager,
        profile: &mut Profile,
        mut recycler: Option<&mut JitFunctionRecycler>,
        timestamp_converter: &TimestampConverter,
        should_add_marker: bool,
    ) {
        let Some(reader) = self.reader.as_mut() else {
            return;
        };
        while let Ok(Some(next_record_header)) = reader.next_record_header() {
            if (next_record_header.total_size as usize) < JitDumpRecordHeader::SIZE {
                // Corrupted file. Skipping a record smaller than its own header seeks back
                // onto that same header, so the reader would never make progress.
                break;
            }
            match next_record_header.record_type {
                JitDumpRecordType::JIT_CODE_LOAD
                | JitDumpRecordType::JIT_CODE_MOVE
                | JitDumpRecordType::JIT_CODE_UNWINDING_INFO
                | JitDumpRecordType::JIT_CODE_CLOSE => {
                    // These are interesting.
                }
                _ => {
                    // We skip other records. We especially want to skip JIT_CODE_DEBUG_INFO
                    // records because they can be big and we don't need to read them from
                    // the file.
                    if let Ok(true) = reader.skip_next_record() {
                        continue;
                    } else {
                        break;
                    }
                }
            }
            let Ok(Some(raw_jitdump_record)) = reader.next_record() else {
                break;
            };
            match raw_jitdump_record.parse() {
                Ok(JitDumpRecord::CodeLoad(record)) => {
                    let start_avma = record.code_addr;
                    let code_size = record.code_bytes.len() as u32;
                    let end_avma = start_avma + u64::from(code_size);

                    let relative_address_at_start = self.cumulative_address;
                    self.cumulative_address += code_size;

                    let symbol_name = record.function_name.as_slice();
                    let symbol_name = std::str::from_utf8(&symbol_name).unwrap_or("");
                    self.symbols.push(Symbol {
                        address: relative_address_at_start,
                        size: Some(code_size),
                        name: symbol_name.to_owned(),
                    });

                    if should_add_marker {
                        let timestamp =
                            timestamp_converter.convert_time(raw_jitdump_record.timestamp);
                        let symbol_name_handle = profile.handle_for_string(symbol_name);
                        profile.add_marker(
                            self.thread_handle,
                            MarkerTiming::Instant(timestamp),
                            JitFunctionAddMarker(symbol_name_handle),
                        );
                    }

                    let (lib_handle, relative_address_at_start) =
                        if let Some(recycler) = recycler.as_deref_mut() {
                            recycler.recycle(
                                symbol_name,
                                code_size,
                                self.lib_handle,
                                relative_address_at_start,
                            )
                        } else {
                            (self.lib_handle, relative_address_at_start)
                        };

                    let (category, js_frame) =
                        jit_category_manager.classify_jit_symbol(symbol_name, profile);
                    self.lib_mapping_ops.push(
                        raw_jitdump_record.timestamp,
                        LibMappingOp::Add(LibMappingAdd {
                            start_avma,
                            end_avma,
                            relative_address_at_start,
                            info: LibMappingInfo::new_jit_function(lib_handle, category, js_frame),
                        }),
                    );
                    // TODO: Add to unwinder so that it can use the code bytes for prologue / epilogue detection
                }
                Ok(JitDumpRecord::CodeMove(record)) => {
                    self.lib_mapping_ops.push(
                        raw_jitdump_record.timestamp,
                        LibMappingOp::Move(LibMappingMove {
                            old_start_avma: record.old_code_addr,
                            new_start_avma: record.new_code_addr,
                            new_end_avma: record.new_code_addr + record.code_size,
                        }),
                    );
                    // TODO: Remove from + add to unwinder
                }
                Ok(JitDumpRecord::CodeUnwindingInfo(_unwinding_info)) => {
                    // TODO: Queue up, and add to unwinder on next CodeLoad
                }
                Ok(JitDumpRecord::CodeClose) => {
                    self.lib_mapping_ops
                        .push(raw_jitdump_record.timestamp, LibMappingOp::Clear);
                    self.close_and_commit_symbol_table(profile);
                    return;
                }
                _ => {}
            }
        }
    }

    fn close_and_commit_symbol_table(&mut self, profile: &mut Profile) {
        if self.reader.is_none() {
            // We're already closed.
            return;
        }

        let symbol_table = SymbolTable::new(std::mem::take(&mut self.symbols));
        profile.set_lib_symbol_table(self.lib_handle, Arc::new(symbol_table));
        self.reader = None;
    }

    pub fn finish(mut self, profile: &mut Profile) -> LibMappingOpQueue {
        self.close_and_commit_symbol_table(profile);
        self.lib_mapping_ops
    }
}

#[cfg(test)]
mod test {
    use std::sync::mpsc;
    use std::time::Duration;

    use fxprof_processed_profile::{ReferenceTimestamp, SamplingInterval, Timestamp};

    use super::*;

    fn code_load_record(name: &str, code_addr: u64) -> Vec<u8> {
        let code = [0u8; 20];
        let total_size = 16 + 40 + name.len() + 1 + code.len();
        let mut r = Vec::new();
        r.extend_from_slice(&0u32.to_le_bytes()); // JIT_CODE_LOAD
        r.extend_from_slice(&(total_size as u32).to_le_bytes());
        r.extend_from_slice(&0u64.to_le_bytes()); // timestamp
        r.extend_from_slice(&1u32.to_le_bytes()); // pid
        r.extend_from_slice(&1u32.to_le_bytes()); // tid
        r.extend_from_slice(&code_addr.to_le_bytes()); // vma
        r.extend_from_slice(&code_addr.to_le_bytes());
        r.extend_from_slice(&(code.len() as u64).to_le_bytes());
        r.extend_from_slice(&0u64.to_le_bytes()); // code_index
        r.extend_from_slice(name.as_bytes());
        r.push(0);
        r.extend_from_slice(&code);
        r
    }

    /// A valid record, then a header with an unknown type and `total_size == 0`, as seen
    /// after CPython duplicated buffered jitdump bytes on fork, then another valid record.
    fn jitdump_with_zero_sized_record() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(&0x4A695444u32.to_le_bytes()); // magic "JiTD"
        d.extend_from_slice(&1u32.to_le_bytes()); // version
        d.extend_from_slice(&40u32.to_le_bytes()); // header size
        d.extend_from_slice(&183u32.to_le_bytes()); // EM_AARCH64
        d.extend_from_slice(&0u32.to_le_bytes()); // pad
        d.extend_from_slice(&1u32.to_le_bytes()); // pid
        d.extend_from_slice(&0u64.to_le_bytes()); // timestamp
        d.extend_from_slice(&0u64.to_le_bytes()); // flags
        d.extend(code_load_record("py::first", 0x1000));
        d.extend_from_slice(&0x1400u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u64.to_le_bytes());
        d.extend(code_load_record("py::after_corruption", 0x2000));
        d
    }

    #[test]
    fn stops_at_record_smaller_than_its_header() {
        let path =
            std::env::temp_dir().join(format!("jit-zero-sized-record-{}.dump", std::process::id()));
        std::fs::write(&path, jitdump_with_zero_sized_record()).unwrap();

        let (tx, rx) = mpsc::channel();
        let thread_path = path.clone();
        std::thread::spawn(move || {
            let mut profile = Profile::new(
                "",
                ReferenceTimestamp::from_millis_since_unix_epoch(0.0),
                SamplingInterval::from_millis(1),
            );
            let process = profile.add_process("p", 1, Timestamp::from_millis_since_reference(0.0));
            let thread = profile.add_thread(
                process,
                1,
                Timestamp::from_millis_since_reference(0.0),
                true,
            );
            let mut manager = JitDumpManager::new(false, false);
            manager.add_jitdump_path(thread, thread_path, vec![]);
            let queues = manager.finish(
                &mut JitCategoryManager::new(),
                &mut profile,
                None,
                &TimestampConverter {
                    reference_raw: 0,
                    raw_to_ns_factor: 1,
                },
            );
            let ops: Vec<usize> = queues
                .into_iter()
                .map(|queue| {
                    let mut ops = queue.into_queue_iter();
                    std::iter::from_fn(|| ops.next_op_if_at_or_before(u64::MAX)).count()
                })
                .collect();
            tx.send(ops).unwrap();
        });
        let ops = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("jitdump processing did not terminate");
        std::fs::remove_file(&path).unwrap();
        assert_eq!(ops, vec![1]);
    }
}
