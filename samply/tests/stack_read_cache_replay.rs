//! Replays a recorded perf.data through `samply import` and checks that the
//! stack read cache doesn't splice stacks from earlier samples.
//!
//! The fixture in `fixtures/other/stack-read-cache/` was recorded from the
//! `tools/stack-cache-repro` workload (see `make-fixture.sh` there): phase A
//! recurses through `a_recurse` deeper than the 32000-byte user stack copy and
//! works at every depth, then phase B recurses to the same depth through
//! `b_recurse` and only works in `b_leaf`. A `b_leaf` stack that contains
//! `a_recurse` (or the reverse) was completed from stale cached stack words.
//! With an unchecked address -> word cache, every phase B stack in this
//! fixture is spliced onto phase A's frames.
//!
//! The workload is a static x86_64 binary, so the replay doesn't depend on the
//! host's libraries and runs on any host.

use std::io::{Cursor, Read};
use std::ops::Range;
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use object::{Object, ObjectSymbol};
use samply::shared::prop_types::{CoreClrProfileProps, ProfileCreationProps};
use serde_json::Value;

const LIB_NAME: &str = "stack-cache-repro";
/// Frames of the deepest `a_recurse` recursion: depths 0..=35.
const FULL_A_DEPTH: usize = 36;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/other/stack-read-cache")
}

fn gunzip(path: &Path) -> Vec<u8> {
    let mut bytes = Vec::new();
    GzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

fn props() -> ProfileCreationProps {
    ProfileCreationProps {
        profile_name: None,
        fallback_profile_name: "stack-read-cache".into(),
        main_thread_only: false,
        reuse_threads: false,
        fold_recursive_prefix: false,
        unlink_aux_files: false,
        create_per_cpu_threads: false,
        arg_count_to_include_in_process_name: 0,
        override_arch: None,
        presymbolicate: false,
        coreclr: CoreClrProfileProps::default(),
        unknown_event_markers: false,
        should_emit_jit_markers: false,
        should_emit_cswitch_markers: false,
    }
}

/// Address ranges of the workload's functions, as relative addresses (the
/// binary is position-independent, so they equal the symbol addresses).
struct Functions {
    a_leaf: Range<u64>,
    a_recurse: Range<u64>,
    b_leaf: Range<u64>,
    b_recurse: Range<u64>,
}

impl Functions {
    fn from_binary(data: &[u8]) -> Self {
        let file = object::File::parse(data).unwrap();
        let range = |suffix: &str| {
            let symbol = file
                .symbols()
                .find(|symbol| symbol.name().is_ok_and(|name| name.ends_with(suffix)))
                .unwrap_or_else(|| panic!("no symbol ending with {suffix}"));
            symbol.address()..symbol.address() + symbol.size()
        };
        Self {
            a_leaf: range("6a_leaf"),
            a_recurse: range("9a_recurse"),
            b_leaf: range("6b_leaf"),
            b_recurse: range("9b_recurse"),
        }
    }
}

/// The workload-binary frame addresses of every sample, leaf first.
fn sample_stacks(profile: &Value) -> Vec<Vec<u64>> {
    let libs = profile["libs"].as_array().unwrap();
    let shared = &profile["shared"];
    let column = |table: &str, name: &str| shared[table][name].as_array().unwrap().clone();
    let resource_lib = column("resourceTable", "lib");
    let func_resource = column("funcTable", "resource");
    let frame_func = column("frameTable", "func");
    let frame_address = column("frameTable", "address");
    let stack_prefix = column("stackTable", "prefix");
    let stack_frame = column("stackTable", "frame");
    let index = |value: &Value| value.as_u64().map(|v| v as usize);

    let frame_in_workload = |frame: usize| -> Option<u64> {
        let resource = index(&func_resource[index(&frame_func[frame])?])?;
        let lib = index(&resource_lib[resource])?;
        (libs[lib]["name"] == LIB_NAME).then(|| frame_address[frame].as_u64())?
    };

    let mut stacks = Vec::new();
    for thread in profile["threads"].as_array().unwrap() {
        for stack in thread["samples"]["stack"].as_array().unwrap() {
            let mut addresses = Vec::new();
            let mut cursor = index(stack);
            while let Some(stack) = cursor {
                if let Some(address) = frame_in_workload(index(&stack_frame[stack]).unwrap()) {
                    addresses.push(address);
                }
                cursor = index(&stack_prefix[stack]);
            }
            stacks.push(addresses);
        }
    }
    stacks
}

#[test]
fn replayed_stacks_are_not_spliced_from_earlier_samples() {
    let dir = fixture_dir();
    let binary = gunzip(&dir.join("stack-cache-repro.gz"));
    let functions = Functions::from_binary(&binary);
    // The perf.data refers to the binary by its recording path; the converter
    // falls back to looking it up by file name in the binary lookup dirs.
    let binary_dir = tempfile::tempdir().unwrap();
    std::fs::write(binary_dir.path().join(LIB_NAME), &binary).unwrap();

    let perf_data = gunzip(&dir.join("stack-cache-repro.perf.data.gz"));
    let profile = samply::import::perf::convert(
        Cursor::new(perf_data),
        None,
        vec![binary_dir.path().to_owned()],
        vec![],
        props(),
    )
    .unwrap();
    let profile = serde_json::to_value(&profile).unwrap();

    let count_in =
        |stack: &[u64], range: &Range<u64>| stack.iter().filter(|a| range.contains(a)).count();
    let stacks = sample_stacks(&profile);
    let a_stacks: Vec<_> = stacks
        .iter()
        .filter(|stack| stack.first().is_some_and(|a| functions.a_leaf.contains(a)))
        .collect();
    let b_stacks: Vec<_> = stacks
        .iter()
        .filter(|stack| stack.first().is_some_and(|a| functions.b_leaf.contains(a)))
        .collect();
    assert!(
        !a_stacks.is_empty() && !b_stacks.is_empty(),
        "fixture has samples in both leaves"
    );

    let spliced_b = b_stacks
        .iter()
        .filter(|s| count_in(s, &functions.a_recurse) > 0)
        .count();
    let spliced_a = a_stacks
        .iter()
        .filter(|s| count_in(s, &functions.b_recurse) > 0)
        .count();
    assert_eq!(
        (spliced_a, spliced_b),
        (0, 0),
        "spliced stacks out of {} phase A and {} phase B samples",
        a_stacks.len(),
        b_stacks.len()
    );

    // The full phase A chain is deeper than one stack copy, so it can only be
    // unwound completely through cached words of earlier samples.
    let complete_a = a_stacks
        .iter()
        .filter(|s| count_in(s, &functions.a_recurse) == FULL_A_DEPTH)
        .count();
    assert!(
        complete_a > 0,
        "no phase A stack was completed past the stack copy"
    );
}
