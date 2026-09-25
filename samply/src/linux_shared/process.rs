use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use framehop::Unwinder;
use fxprof_processed_profile::{
    CounterHandle, LibraryHandle, MarkerTiming, ProcessHandle, Profile, StringHandle, ThreadHandle,
    Timestamp,
};

use super::process_threads::ProcessThreads;
use super::thread::Thread;
use crate::shared::jit_category_manager::JitCategoryManager;
use crate::shared::jit_function_add_marker::JitFunctionAddMarker;
use crate::shared::jit_function_recycler::JitFunctionRecycler;
use crate::shared::jitdump_manager::JitDumpManager;
use crate::shared::lib_mappings::{LibMappingAdd, LibMappingInfo, LibMappingOp, LibMappingOpQueue};
use crate::shared::marker_file::get_markers;
use crate::shared::perf_map::try_load_perf_map;
use crate::shared::process_sample_data::{MarkerSpanOnThread, ProcessSampleData};
use crate::shared::recycling::{ProcessRecyclingData, ThreadRecycler};
use crate::shared::synthetic_jit_library::SyntheticJitLibrary;
use crate::shared::timestamp_converter::TimestampConverter;
use crate::shared::unresolved_samples::UnresolvedSamples;

pub struct Process<U> {
    pub profile_process: ProcessHandle,
    pub unwinder: U,
    pub jitdump_manager: JitDumpManager,
    pub lib_mapping_ops: LibMappingOpQueue,
    pub name: Option<String>,
    pub threads: ProcessThreads,
    pub pid: i32,
    pub unresolved_samples: UnresolvedSamples,
    pub jit_app_cache_mapping_ops: LibMappingOpQueue,
    pub jit_function_recycler: Option<JitFunctionRecycler>,
    marker_file_paths: Vec<(ThreadHandle, PathBuf, Vec<PathBuf>)>,
    /// Per-thread rings of recent stack windows captured with samples, keyed
    /// by tid. When an unwind walks past the current sample's captured stack
    /// window, it continues through an earlier window of the same thread
    /// whose overlap with the current one is word-for-word identical (see
    /// [`StackSnapshot::continues`]). Other threads' stacks are unrelated, so
    /// their windows must never be used.
    pub stack_read_cache: HashMap<i32, ThreadStackSnapshots>,
    pub prev_mm_filepages_size: i64,
    pub prev_mm_anonpages_size: i64,
    pub prev_mm_swapents_size: i64,
    pub prev_mm_shmempages_size: i64,
    pub mem_counter: Option<CounterHandle>,
    pub extra_event_instances: HashMap<(u64, i32), ExtraEventInstance>,
}

/// Delta-tracking state for one extra-event counter instance, keyed by
/// (kernel event id, tid).
pub struct ExtraEventInstance {
    /// The extra per-sample delta dimension this instance's values feed,
    /// shared with the event's other instances.
    pub dim: usize,
    /// The last raw value seen, used to turn the cumulative values carried by
    /// samples into per-sample deltas.
    pub prev_value: u64,
}

/// The user stack words `[sp, end())` captured with one sample.
pub struct StackSnapshot {
    pub sp: u64,
    pub stable_start: u64,
    pub words: Vec<u64>,
}

impl StackSnapshot {
    pub fn end(&self) -> u64 {
        self.sp + self.words.len() as u64 * 8
    }

    /// The word at `addr`, if it lies inside this snapshot.
    pub fn get(&self, addr: u64) -> Option<u64> {
        let index = usize::try_from(addr.checked_sub(self.sp)? / 8).ok()?;
        self.words.get(index).copied()
    }

    /// Whether `self`, captured by an earlier sample, continues `window` past
    /// its end. Only words at or above this snapshot's stable start are
    /// compared: words below it belong to the leaf frame and can change
    /// between samples.
    pub fn continues(&self, window: &StackSnapshot) -> bool {
        if self.sp <= window.sp
            || self.sp >= window.end()
            || self.end() <= window.end()
            || self.stable_start <= self.sp
            || self.stable_start >= window.end()
        {
            return false;
        }
        let stable_offset = self.stable_start - self.sp;
        let window_offset = self.stable_start - window.sp;
        if stable_offset % 8 != 0 || window_offset % 8 != 0 {
            return false;
        }
        let overlap_words = ((window.end() - self.stable_start) / 8) as usize;
        let self_start = (stable_offset / 8) as usize;
        let window_start = (window_offset / 8) as usize;
        self.words
            .get(self_start..self_start + overlap_words)
            .zip(window.words.get(window_start..window_start + overlap_words))
            .is_some_and(|(self_words, window_words)| self_words == window_words)
    }
}

/// The stack windows of one thread that may still be current, newest first.
/// Retirement and covering keep only snapshots that extend further up the
/// stack than every newer one, so the count stays bounded by the stack depth.
#[derive(Default)]
pub struct ThreadStackSnapshots {
    ring: VecDeque<StackSnapshot>,
}

impl ThreadStackSnapshots {
    /// Adds the newest snapshot. Older snapshots whose stable part it fully
    /// covers are dropped: it holds fresher words for all of their addresses.
    pub fn push(&mut self, snapshot: StackSnapshot) {
        self.ring.retain(|older| {
            older.stable_start < snapshot.stable_start || older.end() > snapshot.end()
        });
        self.ring.push_front(snapshot);
    }

    /// Drops the snapshots whose stable frames have returned by the time of a
    /// sample at `sp`: the thread's stack pointer is now above their lowest
    /// trusted slot, so the words there have been overwritten or will be.
    pub fn retire_returned(&mut self, sp: u64) {
        self.ring.retain(|snapshot| snapshot.stable_start >= sp);
    }

    /// The word at `addr` past the end of `window`, read from the snapshots
    /// that continue it. `chain` holds the indices of the snapshots followed
    /// so far and is extended only when a read needs another hop.
    pub fn read_past(
        &self,
        window: &StackSnapshot,
        chain: &mut Vec<usize>,
        addr: u64,
    ) -> Option<u64> {
        loop {
            let tip = chain.last().map_or(window, |&index| &self.ring[index]);
            if let Some(value) = tip.get(addr) {
                return Some(value);
            }
            // Continuations only start above the tip's sp.
            if addr < tip.sp {
                return None;
            }
            let next = self.find_continuation(tip, chain)?;
            chain.push(next);
        }
    }

    /// The index of the newest snapshot that continues `window` (see
    /// [`StackSnapshot::continues`]), skipping the indices in `used`.
    fn find_continuation(&self, window: &StackSnapshot, used: &[usize]) -> Option<usize> {
        self.ring
            .iter()
            .enumerate()
            .position(|(index, snapshot)| !used.contains(&index) && snapshot.continues(window))
    }
}

pub struct ProcessForkData<U> {
    unwinder: U,
    lib_mapping_ops: LibMappingOpQueue,
}

impl<U> Process<U>
where
    U: Unwinder + Default,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pid: i32,
        process_handle: ProcessHandle,
        main_thread_handle: ThreadHandle,
        main_thread_label: StringHandle,
        name: Option<String>,
        thread_recycler: Option<ThreadRecycler>,
        jit_function_recycler: Option<JitFunctionRecycler>,
        unlink_aux_files: bool,
        should_emit_jit_markers: bool,
    ) -> Self {
        Self {
            profile_process: process_handle,
            unwinder: U::default(),
            jitdump_manager: JitDumpManager::new(unlink_aux_files, should_emit_jit_markers),
            lib_mapping_ops: Default::default(),
            name: name.clone(),
            pid,
            threads: ProcessThreads::new(
                pid,
                process_handle,
                main_thread_handle,
                main_thread_label,
                name,
                thread_recycler,
            ),
            unresolved_samples: Default::default(),
            jit_app_cache_mapping_ops: LibMappingOpQueue::default(),
            jit_function_recycler,
            marker_file_paths: Vec::new(),
            stack_read_cache: HashMap::new(),
            prev_mm_filepages_size: 0,
            prev_mm_anonpages_size: 0,
            prev_mm_swapents_size: 0,
            prev_mm_shmempages_size: 0,
            mem_counter: None,
            extra_event_instances: HashMap::new(),
        }
    }

    /// Called when this process forks and creates a child process.
    pub fn clone_fork_data(&self) -> ProcessForkData<U> {
        ProcessForkData {
            unwinder: self.unwinder.clone(),
            lib_mapping_ops: self.lib_mapping_ops.clone(),
        }
    }

    /// Called on the child process that was created by the fork.
    pub fn adopt_fork_data_from_parent(&mut self, fork_data: ProcessForkData<U>) {
        self.unwinder = fork_data.unwinder;
        self.lib_mapping_ops = fork_data.lib_mapping_ops;
    }

    pub fn rename_with_recycling(
        &mut self,
        name: String,
        recycling_data: ProcessRecyclingData,
    ) -> (ProcessRecyclingData, Option<String>) {
        let ProcessRecyclingData {
            process_handle,
            main_thread_recycling_data,
            thread_recycler,
            jit_function_recycler,
        } = recycling_data;
        let old_process_handle = std::mem::replace(&mut self.profile_process, process_handle);
        let old_jit_function_recycler = self.jit_function_recycler.replace(jit_function_recycler);
        let (old_thread_recycler, old_main_thread_recycling_data) =
            self.threads.rename_process_with_recycling(
                name.clone(),
                process_handle,
                main_thread_recycling_data,
                thread_recycler,
            );
        let old_name = self.name.replace(name);
        let recycling_data = ProcessRecyclingData {
            process_handle: old_process_handle,
            main_thread_recycling_data: old_main_thread_recycling_data,
            thread_recycler: old_thread_recycler,
            jit_function_recycler: old_jit_function_recycler
                .expect("jit_function_recycler should be Some"),
        };
        (recycling_data, old_name)
    }

    pub fn rename_without_recycling(
        &mut self,
        name: String,
        main_thread_label: StringHandle,
        profile: &mut Profile,
    ) {
        profile.set_process_name(self.profile_process, &name);
        self.threads
            .main_thread
            .rename_without_recycling(name.clone(), main_thread_label, profile);
        self.name = Some(name);
    }

    pub fn recycle_or_get_new_thread(
        &mut self,
        tid: i32,
        name: Option<String>,
        start_time: Timestamp,
        profile: &mut Profile,
    ) -> &mut Thread {
        self.threads
            .recycle_or_get_new_thread(tid, name, start_time, profile)
    }

    pub fn check_jitdump(
        &mut self,
        jit_category_manager: &mut JitCategoryManager,
        profile: &mut Profile,
        timestamp_converter: &TimestampConverter,
    ) {
        self.jitdump_manager.process_pending_records(
            jit_category_manager,
            profile,
            self.jit_function_recycler.as_mut(),
            timestamp_converter,
        );
    }

    pub fn add_marker_file_path(
        &mut self,
        thread: ThreadHandle,
        path: &Path,
        lookup_dirs: Vec<PathBuf>,
    ) {
        self.marker_file_paths
            .push((thread, path.to_owned(), lookup_dirs));
    }

    pub fn notify_dead(&mut self, end_time: Timestamp, profile: &mut Profile) {
        self.threads.notify_process_dead(end_time, profile);
        profile.set_process_end_time(self.profile_process, end_time);
    }

    pub fn finish(
        mut self,
        profile: &mut Profile,
        jit_category_manager: &mut JitCategoryManager,
        timestamp_converter: &TimestampConverter,
    ) -> (ProcessSampleData, Option<(String, ProcessRecyclingData)>) {
        self.unwinder = U::default();

        let perf_map_mappings = if !self.unresolved_samples.is_empty() {
            try_load_perf_map(
                self.pid as u32,
                profile,
                jit_category_manager,
                self.jit_function_recycler.as_mut(),
            )
        } else {
            None
        };

        let jitdump_manager = self.jitdump_manager;
        let mut jitdump_ops = jitdump_manager.finish(
            jit_category_manager,
            profile,
            self.jit_function_recycler.as_mut(),
            timestamp_converter,
        );

        if !self.jit_app_cache_mapping_ops.is_empty() {
            jitdump_ops.insert(0, self.jit_app_cache_mapping_ops);
        }

        let mut marker_spans = Vec::new();
        for (thread_handle, marker_file_path, lookup_dirs) in self.marker_file_paths {
            if let Ok(marker_spans_from_this_file) =
                get_markers(&marker_file_path, &lookup_dirs, *timestamp_converter)
            {
                marker_spans.extend(marker_spans_from_this_file.into_iter().map(|span| {
                    MarkerSpanOnThread {
                        thread_handle,
                        start_time: span.start_time,
                        end_time: span.end_time,
                        name: span.name,
                    }
                }));
            }
        }

        let process_sample_data = ProcessSampleData::new(
            self.profile_process,
            std::mem::take(&mut self.unresolved_samples),
            std::mem::take(&mut self.lib_mapping_ops),
            jitdump_ops,
            perf_map_mappings,
            marker_spans,
        );

        let thread_recycler = self.threads.finish();

        let process_recycling_data = if let (
            Some(name),
            Some(jit_function_recycler),
            (Some(thread_recycler), main_thread_recycling_data),
        ) = (self.name, self.jit_function_recycler, thread_recycler)
        {
            let recycling_data = ProcessRecyclingData {
                process_handle: self.profile_process,
                main_thread_recycling_data,
                thread_recycler,
                jit_function_recycler,
            };
            Some((name, recycling_data))
        } else {
            None
        };

        (process_sample_data, process_recycling_data)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_regular_lib_mapping(
        &mut self,
        timestamp: u64,
        start_address: u64,
        end_address: u64,
        relative_address_at_start: u32,
        info: LibMappingInfo,
    ) {
        self.lib_mapping_ops.push(
            timestamp,
            LibMappingOp::Add(LibMappingAdd {
                start_avma: start_address,
                end_avma: end_address,
                relative_address_at_start,
                info,
            }),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_lib_mapping_for_injected_jit_lib(
        &mut self,
        timestamp: u64,
        profile_timestamp: Timestamp,
        symbol_name: Option<&str>,
        start_address: u64,
        end_address: u64,
        mut relative_address_at_start: u32,
        mut lib_handle: LibraryHandle,
        jit_category_manager: &mut JitCategoryManager,
        profile: &mut Profile,
        should_add_marker: bool,
    ) {
        if should_add_marker {
            let main_thread = self.threads.main_thread.profile_thread;
            let timing = MarkerTiming::Instant(profile_timestamp);
            let name = match symbol_name {
                Some(name) => profile.handle_for_string(name),
                None => profile.handle_for_string("<unknown>"),
            };
            profile.add_marker(main_thread, timing, JitFunctionAddMarker(name));
        }

        if let (Some(name), Some(recycler)) = (symbol_name, self.jit_function_recycler.as_mut()) {
            let code_size = (end_address - start_address) as u32;
            (lib_handle, relative_address_at_start) =
                recycler.recycle(name, code_size, lib_handle, relative_address_at_start);
        }

        let (category, js_frame) =
            jit_category_manager.classify_jit_symbol(symbol_name.unwrap_or(""), profile);
        self.lib_mapping_ops.push(
            timestamp,
            LibMappingOp::Add(LibMappingAdd {
                start_avma: start_address,
                end_avma: end_address,
                relative_address_at_start,
                info: LibMappingInfo::new_jit_function(lib_handle, category, js_frame),
            }),
        );
    }

    pub fn add_jit_function(
        &mut self,
        timestamp_raw: u64,
        jit_lib: &mut SyntheticJitLibrary,
        name: String,
        start_avma: u64,
        size: u32,
        info: LibMappingInfo,
    ) {
        let relative_address = jit_lib.add_function(name, size);

        self.jit_app_cache_mapping_ops.push(
            timestamp_raw,
            LibMappingOp::Add(LibMappingAdd {
                start_avma,
                end_avma: start_avma + u64::from(size),
                relative_address_at_start: relative_address,
                info,
            }),
        );
    }

    pub fn get_or_make_mem_counter(&mut self, profile: &mut Profile) -> CounterHandle {
        *self.mem_counter.get_or_insert_with(|| {
            profile.add_counter(
                self.profile_process,
                "malloc",
                "Memory",
                "Amount of allocated memory",
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{StackSnapshot, ThreadStackSnapshots};

    const WINDOW_SP: u64 = 0x7fff_0000_0000;
    const WINDOW_WORDS: usize = 4096;

    fn window() -> StackSnapshot {
        StackSnapshot {
            sp: WINDOW_SP,
            stable_start: WINDOW_SP,
            words: vec![0; WINDOW_WORDS],
        }
    }

    /// An earlier snapshot that matches `window()` on its last
    /// `overlap_words` words and extends one word past it.
    fn earlier(overlap_words: usize) -> StackSnapshot {
        let stable_start = window().end() - overlap_words as u64 * 8;
        let sp = stable_start - 8;
        StackSnapshot {
            sp,
            stable_start,
            words: vec![0; overlap_words + 2],
        }
    }

    #[test]
    fn only_words_from_stable_start_must_match() {
        let window = window();

        let mut leaf_differs = earlier(1);
        leaf_differs.words[0] = 0xdead;
        assert!(leaf_differs.continues(&window));

        let mut overlap_differs = earlier(1);
        overlap_differs.words[1] = 0xdead;
        assert!(!overlap_differs.continues(&window));
    }

    #[test]
    fn snapshots_are_retired_once_their_stable_frames_returned() {
        let mut snapshots = ThreadStackSnapshots::default();
        snapshots.push(earlier(16));
        let stable_start = snapshots.ring[0].stable_start;

        snapshots.retire_returned(stable_start);
        assert_eq!(snapshots.ring.len(), 1);

        snapshots.retire_returned(stable_start + 8);
        assert!(snapshots.ring.is_empty());
    }

    #[test]
    fn covered_snapshots_are_dropped() {
        let mut snapshots = ThreadStackSnapshots::default();
        snapshots.push(earlier(16));
        // Same range, captured later: the older copy is redundant.
        snapshots.push(earlier(16));
        assert_eq!(snapshots.ring.len(), 1);
        // Reaches less far up the stack: both are kept.
        snapshots.push(earlier(8));
        assert_eq!(snapshots.ring.len(), 2);
    }
}
