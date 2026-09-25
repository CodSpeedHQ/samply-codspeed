#!/usr/bin/env bash
# Regenerate fixtures/other/stack-read-cache/, used by
# samply/tests/stack_read_cache_replay.rs.
#
# Builds the workload as a static x86_64 binary (so the replay doesn't depend
# on the host's libc), records 400 ms per phase with DWARF call graphs, and
# stores both gzipped. The binary is recorded from a fixed path; the test
# finds it by file name in a lookup dir instead.
#
# Needs a static glibc: set GLIBC_STATIC_LIB to a directory with libc.a
# (on NixOS: $(nix-build '<nixpkgs>' -A glibc.static --no-out-link)/lib).
set -euo pipefail

script_dir=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_dir=${FIXTURE_DIR:-$script_dir/../../fixtures/other/stack-read-cache}
record_dir=/tmp/samply-stack-cache-fixture
target=x86_64-unknown-linux-gnu

rustflags="-C target-feature=+crt-static"
if [[ -n ${GLIBC_STATIC_LIB:-} ]]; then
    rustflags+=" -L native=$GLIBC_STATIC_LIB"
fi
RUSTFLAGS=$rustflags cargo build --release --target "$target" --manifest-path "$script_dir/Cargo.toml"

# The recording path ends up in perf.data, so it is fixed; refuse to reuse it.
if [[ -e $record_dir ]]; then
    echo "$record_dir already exists; move it away first" >&2
    exit 1
fi
mkdir -p "$record_dir" "$fixture_dir"
cp "$script_dir/target/$target/release/stack-cache-repro" "$record_dir/stack-cache-repro"
strip --strip-debug "$record_dir/stack-cache-repro"
(cd "$record_dir" && perf record -e cycles:u --call-graph dwarf,32000 -F 199 -m 16 \
    -o stack-cache-repro.perf.data -- ./stack-cache-repro 400)

gzip -9 -c "$record_dir/stack-cache-repro" > "$fixture_dir/stack-cache-repro.gz"
gzip -9 -c "$record_dir/stack-cache-repro.perf.data" > "$fixture_dir/stack-cache-repro.perf.data.gz"
trash-put "$record_dir"
