use std::collections::VecDeque;
use std::iter::{Cloned, Rev};

use fxprof_processed_profile::{
    Category, CategoryColor, FrameAddress, FrameFlags, FrameHandle, ProcessHandle, Profile,
    SubcategoryHandle,
};

use super::jit_category_manager::{JsFrame, JsName};
use super::lib_mappings::{AndroidArtInfo, LibMappingsHierarchy};
use super::types::{StackFrame, StackMarker, StackMode};

#[derive(Debug)]
pub struct StackConverter {
    user_category: SubcategoryHandle,
    kernel_category: SubcategoryHandle,
    libart_frame_buffer: VecDeque<SecondPassFrameInfo>,
}

struct FirstPassFrameInfo {
    mode: StackMode,
    lookup_address: u64,
    from_ip: bool,
}

#[derive(Debug)]
struct SecondPassFrameInfo {
    location: FrameAddress,
    category: SubcategoryHandle,
    js_frame: Option<JsFrame>,
    art_info: Option<AndroidArtInfo>,
}

struct FirstPassIter<I: Iterator<Item = StackFrame>>(I);

struct SecondPassIter<'a, I: Iterator<Item = FirstPassFrameInfo>> {
    inner: I,
    process_handle: ProcessHandle,
    lib_mappings: &'a LibMappingsHierarchy,
    user_category: SubcategoryHandle,
    kernel_category: SubcategoryHandle,
}

struct LibartFilteringIter<'c, I: Iterator<Item = SecondPassFrameInfo>> {
    inner: I,
    last_emitted_was_java: bool,
    buffer: &'c mut VecDeque<SecondPassFrameInfo>,
}

struct ConvertedStackIterD<I: Iterator<Item = SecondPassFrameInfo>> {
    inner: I,
    pending_frame_handle: Option<FrameHandle>,
    pending_marker: Option<StackMarker>,
    js_name_for_baseline_interpreter: Option<JsName>,
}

#[allow(clippy::type_complexity)]
pub struct ConvertedStackIter<'a>(
    ConvertedStackIterD<
        LibartFilteringIter<
            'a,
            SecondPassIter<'a, FirstPassIter<Rev<Cloned<std::slice::Iter<'a, StackFrame>>>>>,
        >,
    >,
);

impl<I: Iterator<Item = StackFrame>> Iterator for FirstPassIter<I> {
    type Item = FirstPassFrameInfo;

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn next(&mut self) -> Option<Self::Item> {
        let (mode, lookup_address, from_ip) = match self.0.next()? {
            StackFrame::InstructionPointer(addr, mode) => (mode, addr, true),
            StackFrame::ReturnAddress(addr, mode) => (mode, addr.saturating_sub(1), false),
            StackFrame::AdjustedReturnAddress(addr, mode) => (mode, addr, false),
            StackFrame::StackMarker(_) => unreachable!("stack markers must be root-most"),
        };
        Some(FirstPassFrameInfo {
            mode,
            lookup_address,
            from_ip,
        })
    }
}

impl<I: Iterator<Item = FirstPassFrameInfo>> Iterator for SecondPassIter<'_, I> {
    type Item = SecondPassFrameInfo;

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn next(&mut self) -> Option<Self::Item> {
        let FirstPassFrameInfo {
            mode,
            lookup_address,
            from_ip,
        } = self.inner.next()?;
        let (location, category, js_frame, art_info) = match mode {
            StackMode::User => match self.lib_mappings.convert_address(lookup_address) {
                Some((relative_lookup_address, info)) => {
                    let location = if from_ip {
                        let relative_address = relative_lookup_address;
                        FrameAddress::RelativeAddressFromInstructionPointer(
                            info.lib_handle,
                            relative_address,
                        )
                    } else {
                        FrameAddress::RelativeAddressFromAdjustedReturnAddress(
                            info.lib_handle,
                            relative_lookup_address,
                        )
                    };
                    (
                        location,
                        info.category.unwrap_or(self.user_category),
                        info.js_frame,
                        info.art_info,
                    )
                }
                None => {
                    let p = self.process_handle;
                    let location = match from_ip {
                        true => FrameAddress::InstructionPointer(p, lookup_address),
                        false => FrameAddress::AdjustedReturnAddress(p, lookup_address),
                    };
                    (location, self.user_category, None, None)
                }
            },
            StackMode::Kernel => {
                let location = match from_ip {
                    true => FrameAddress::KernelInstructionPointer(lookup_address),
                    false => FrameAddress::KernelAdjustedReturnAddress(lookup_address),
                };
                (location, self.kernel_category, None, None)
            }
        };
        Some(SecondPassFrameInfo {
            location,
            category,
            js_frame,
            art_info,
        })
    }
}

impl<I: Iterator<Item = SecondPassFrameInfo>> Iterator for LibartFilteringIter<'_, I> {
    type Item = SecondPassFrameInfo;

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(buffered_libart_frame) = self.buffer.pop_front() {
            return Some(buffered_libart_frame);
        }

        let mut frame = self.inner.next()?;

        if self.last_emitted_was_java {
            // Buffer frames until we find a non-libart frame.
            while frame.art_info == Some(AndroidArtInfo::LibArt) {
                self.buffer.push_back(frame);

                match self.inner.next() {
                    Some(next_frame) => {
                        frame = next_frame;
                    }
                    None => {
                        return self.buffer.pop_front();
                    }
                }
            }
        }

        if frame.art_info == Some(AndroidArtInfo::JavaFrame) {
            self.buffer.clear();
            self.last_emitted_was_java = true;
            return Some(frame);
        }

        self.last_emitted_was_java = false;
        if self.buffer.is_empty() {
            Some(frame)
        } else {
            self.buffer.push_back(frame);
            self.buffer.pop_front()
        }
    }
}

impl<I: Iterator<Item = SecondPassFrameInfo>> ConvertedStackIterD<I> {
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Approximate the converted depth for stack elision. JS frames can expand
        // the stack and ART filtering can shrink it.
        let pending =
            self.pending_frame_handle.is_some() as usize + self.pending_marker.is_some() as usize;
        let (lower, upper) = self.inner.size_hint();
        (lower + pending, upper.map(|upper| upper + pending))
    }

    fn next(&mut self, profile: &mut Profile) -> Option<FrameHandle> {
        if let Some(pending_frame_handle) = self.pending_frame_handle.take() {
            return Some(pending_frame_handle);
        }
        if let Some(marker) = self.pending_marker.take() {
            let category = profile.handle_for_category(Category("Unwinding", CategoryColor::Gray));
            let label = profile.handle_for_string(marker.label());
            return Some(profile.handle_for_frame_with_label(label, category, FrameFlags::empty()));
        }
        let SecondPassFrameInfo {
            location,
            category,
            js_frame,
            ..
        } = self.inner.next()?;

        let mut frame_flags = FrameFlags::empty();

        // Work around an imperfection in Spidermonkey's stack frames.
        // We sometimes have missing BaselineInterpreterStubs in the OSR-into-BaselineInterpreter case.
        // Usually, a BaselineInterpreter frame is directly preceded by a BaselineInterpreterStub frame.
        // However, sometimes you get Regular(x) -> None -> None -> None -> BaselineInterpreter,
        // without a BaselineInterpreterStub frame. In that case, the name "x" from the ancestor
        // JsFrame::RegularInAdditionToNativeFrame (which is really an InterpreterStub frame for the C++ interpreter)
        // should be used for the BaselineInterpreter frame. This will create a stack
        // node with the right name, category and JS-only flag, and helps with correct attribution.
        // Unfortunately it means that we'll have two prepended JS label frames for the same function
        // in that case, but that's still better than accounting those samples to the wrong JS function.
        let extra_js_name = match js_frame {
            Some(JsFrame::NativeFrameIsJs) => {
                frame_flags |= FrameFlags::IS_JS;
                None
            }
            Some(JsFrame::RegularInAdditionToNativeFrame(js_name)) => {
                // Remember the name for a potentially upcoming unnamed BaselineInterpreter frame.
                self.js_name_for_baseline_interpreter = Some(js_name);
                Some(js_name)
            }
            Some(JsFrame::BaselineInterpreterStub(js_name)) => {
                // Discard the name of an ancestor JS function.
                self.js_name_for_baseline_interpreter = None;
                Some(js_name)
            }
            Some(JsFrame::BaselineInterpreter) => self.js_name_for_baseline_interpreter.take(),
            None => None,
        };

        let mut frame_handle =
            profile.handle_for_frame_with_address(location, category, frame_flags);
        if let Some(JsName::NonSelfHosted(js_name)) = extra_js_name {
            // Prepend a JS frame.
            // We don't treat Spidermonkey "self-hosted" functions as JS (e.g. filter/map/push).
            let prepended_js_frame =
                profile.handle_for_frame_with_label(js_name, category, FrameFlags::IS_JS);
            let buffered_frame = std::mem::replace(&mut frame_handle, prepended_js_frame);
            self.pending_frame_handle = Some(buffered_frame);
        };

        Some(frame_handle)
    }
}

impl ConvertedStackIter<'_> {
    pub fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    pub fn next(&mut self, profile: &mut Profile) -> Option<FrameHandle> {
        self.0.next(profile)
    }
}

impl StackConverter {
    pub fn new(user_category: SubcategoryHandle, kernel_category: SubcategoryHandle) -> Self {
        Self {
            user_category,
            kernel_category,
            libart_frame_buffer: VecDeque::new(),
        }
    }

    /// Takes a stack going from callee to root caller.
    ///
    /// Returns an iterator going from root caller to callee.
    pub fn convert_stack<'a>(
        &'a mut self,
        process: ProcessHandle,
        stack: &'a [StackFrame],
        lib_mappings: &'a LibMappingsHierarchy,
        extra_first_frame: Option<FrameHandle>,
    ) -> ConvertedStackIter<'a> {
        // Producers append markers after stack reconciliation. Keep the root marker
        // outside JS/ART processing, immediately inside any extra label frame.
        let (stack, pending_marker) = match stack.split_last() {
            Some((StackFrame::StackMarker(marker), frames)) => {
                (frames, (!frames.is_empty()).then_some(*marker))
            }
            _ => (stack, None),
        };
        let pass1 = FirstPassIter(stack.iter().cloned().rev());
        let pass2 = SecondPassIter {
            inner: pass1,
            process_handle: process,
            lib_mappings,
            user_category: self.user_category,
            kernel_category: self.kernel_category,
        };
        self.libart_frame_buffer.clear();
        let pass3 = LibartFilteringIter {
            inner: pass2,
            last_emitted_was_java: false,
            buffer: &mut self.libart_frame_buffer,
        };
        let pass4 = ConvertedStackIterD {
            inner: pass3,
            pending_frame_handle: extra_first_frame,
            pending_marker,
            js_name_for_baseline_interpreter: None,
        };
        ConvertedStackIter(pass4)
    }
}

#[cfg(test)]
mod tests {
    use fxprof_processed_profile::{
        Category, CategoryColor, CpuDelta, FrameFlags, ProcessHandle, Profile, ReferenceTimestamp,
        SamplingInterval, SubcategoryHandle, ThreadHandle, Timestamp,
    };
    use serde_json::Value;

    use super::super::lib_mappings::{LibMappingOpQueue, LibMappingsHierarchy};
    use super::super::stack_depth_limiting_frame_iter::StackDepthLimitingFrameIter;
    use super::super::types::{StackFrame, StackMarker, StackMode};
    use super::StackConverter;

    fn new_profile() -> (Profile, ProcessHandle, ThreadHandle, SubcategoryHandle) {
        let mut profile = Profile::new(
            "test",
            ReferenceTimestamp::from_millis_since_unix_epoch(0.0),
            SamplingInterval::from_millis(1),
        );
        let process = profile.add_process("test", 1, Timestamp::from_millis_since_reference(0.0));
        let thread = profile.add_thread(
            process,
            1,
            Timestamp::from_millis_since_reference(0.0),
            true,
        );
        let user_category = profile
            .handle_for_category(Category("User", CategoryColor::Blue))
            .into();
        (profile, process, thread, user_category)
    }

    fn serialized_stack_labels(profile: &Value, stack_index: usize) -> Vec<String> {
        let shared = &profile["shared"];
        let stack_table = &shared["stackTable"];
        let frame_table = &shared["frameTable"];
        let func_table = &shared["funcTable"];
        let mut stack_index = Some(stack_index);
        let mut labels = Vec::new();
        while let Some(index) = stack_index {
            let frame_index = stack_table["frame"][index].as_u64().unwrap() as usize;
            let func_index = frame_table["func"][frame_index].as_u64().unwrap() as usize;
            let name_index = func_table["name"][func_index].as_u64().unwrap() as usize;
            labels.push(
                shared["stringArray"][name_index]
                    .as_str()
                    .unwrap()
                    .to_string(),
            );
            stack_index = stack_table["prefix"][index]
                .as_u64()
                .map(|index| index as usize);
        }
        labels.reverse();
        labels
    }

    #[test]
    fn markers_are_preserved_as_distinct_root_labels() {
        let (mut profile, process, thread, user_category) = new_profile();
        let mappings = LibMappingsHierarchy::new(LibMappingOpQueue::default());
        let mut converter = StackConverter::new(user_category, user_category);
        let extra_label_string = profile.handle_for_string("sample");
        let extra_label = profile.handle_for_frame_with_label(
            extra_label_string,
            user_category,
            FrameFlags::empty(),
        );
        for (index, marker, extra) in [
            (0, StackMarker::Truncated, Some(extra_label)),
            (1, StackMarker::UnwindFailed, None),
            (2, StackMarker::Truncated, None),
        ] {
            let frames = [
                StackFrame::InstructionPointer(0x1234, StackMode::User),
                StackFrame::StackMarker(marker),
            ];
            let mut converted = converter.convert_stack(process, &frames, &mappings, extra);
            let stack_handle = profile.handle_for_stack_frames(|profile| converted.next(profile));
            profile.add_sample(
                thread,
                Timestamp::from_millis_since_reference(index as f64),
                stack_handle,
                CpuDelta::ZERO,
                (index + 1) as i32,
            );
        }
        let ordinary_frames = [StackFrame::InstructionPointer(0x1234, StackMode::User)];
        let mut ordinary = converter.convert_stack(process, &ordinary_frames, &mappings, None);
        let ordinary_stack = profile.handle_for_stack_frames(|profile| ordinary.next(profile));
        profile.add_sample(
            thread,
            Timestamp::from_millis_since_reference(3.0),
            ordinary_stack,
            CpuDelta::ZERO,
            4,
        );

        let serialized = serde_json::to_value(&profile).unwrap();
        let thread_json = &serialized["threads"][0];
        let samples = &thread_json["samples"];
        let sample_stacks = samples["stack"].as_array().unwrap();
        let labels: Vec<_> = sample_stacks
            .iter()
            .map(|stack| serialized_stack_labels(&serialized, stack.as_u64().unwrap() as usize))
            .collect();
        assert_eq!(labels[0], ["sample", "[truncated]", "0x1234"]);
        assert_eq!(labels[1], ["[unwind_failed]", "0x1234"]);
        assert_eq!(labels[2], ["[truncated]", "0x1234"]);
        assert_eq!(labels[3], ["0x1234"]);
    }

    #[test]
    fn deep_stack_elision_preserves_the_diagnostic_root() {
        let (mut profile, process, thread, user_category) = new_profile();
        let mappings = LibMappingsHierarchy::new(LibMappingOpQueue::default());
        let mut converter = StackConverter::new(user_category, user_category);
        let mut frames: Vec<_> = (0..502)
            .map(|address| StackFrame::InstructionPointer(address as u64, StackMode::User))
            .collect();
        frames.push(StackFrame::StackMarker(StackMarker::Truncated));
        let converted = converter.convert_stack(process, &frames, &mappings, None);
        let mut converted =
            StackDepthLimitingFrameIter::new(&mut profile, converted, user_category);
        let stack_handle = profile.handle_for_stack_frames(|profile| converted.next(profile));
        profile.add_sample(
            thread,
            Timestamp::from_millis_since_reference(0.0),
            stack_handle,
            CpuDelta::ZERO,
            1,
        );

        let serialized = serde_json::to_value(&profile).unwrap();
        let stack_index = serialized["threads"][0]["samples"]["stack"][0]
            .as_u64()
            .unwrap() as usize;
        let labels = serialized_stack_labels(&serialized, stack_index);
        assert_eq!(labels.first().map(String::as_str), Some("[truncated]"));
        assert!(labels.iter().any(|label| label.contains("frames elided")));
        assert_eq!(labels.last().map(String::as_str), Some("0x0"));
    }
}
