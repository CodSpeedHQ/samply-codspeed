# Stack-cache reproduction

This workload runs two phases. Phase A recurses through `a_recurse` and spends
work at every depth; phase B uses the same deep layout through `b_recurse` but
spends work only in `b_leaf`. The stack is deeper than samply's 32,000-byte
user-stack capture, making stale cached words observable as impossible
cross-phase ancestry.

From this directory, run:

```sh
./replay.sh [path/to/samply ...]
```

With no arguments, `samply` is taken from `PATH`. The script builds the
workload, records one `perf.data`, imports it with each requested samply
binary, and prints per-phase statistics. Set `RERECORD=1` to replace an
existing recording; set `OUT_DIR` to choose another output directory.

The table reports total samples containing each phase's leaf, impossible
**stitched** samples containing the other phase's recursive function,
**truncated** samples that do not reach `main`, and **complete** samples (the
remainder). The script exits non-zero if any stitched sample is found.

The workload uses x86_64 inline assembly, so it builds on x86_64 only.

For ordinary users, recording only needs `perf_event_paranoid <= 2` and the
`cycles:u` event used by the script.

The workload takes an optional phase duration in milliseconds (default 3500).

`make-fixture.sh` records a short run into `fixtures/other/stack-read-cache/`.
`samply/tests/stack_read_cache_replay.rs` replays that recording through
`samply import` in `cargo test`, so the check needs no perf permissions. It
fails if any stack is spliced from an earlier sample, or if no phase A stack
is completed past the stack copy.
