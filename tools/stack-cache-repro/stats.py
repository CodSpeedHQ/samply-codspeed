#!/usr/bin/env python3
"""Count stitched and truncated stacks in profiles of stack-cache-repro.

Usage: stats.py <profile.json.gz>...

Phase A spins in `a_leaf` under `a_recurse`, phase B spins in `b_leaf` under
`b_recurse`. A `b_leaf` stack containing `a_recurse` (or the reverse) is
impossible, so it is counted as stitched. A stack not reaching `main` is
truncated. Exits with status 1 if any profile contains a stitched stack.
"""
import gzip
import json
import os
import sys

LEAVES = {
    "a": ("stack_cache_repro::a_leaf", "stack_cache_repro::b_recurse"),
    "b": ("stack_cache_repro::b_leaf", "stack_cache_repro::a_recurse"),
}
MAIN = "stack_cache_repro::main"


def stacks(path):
    """Yield the set of function names on each sampled stack."""
    with gzip.open(path, "rt") as f:
        profile = json.load(f)
    shared = profile["shared"]
    names = shared["stringArray"]
    func_name = shared["funcTable"]["name"]
    frame_func = shared["frameTable"]["func"]
    prefix = shared["stackTable"]["prefix"]
    stack_frame = shared["stackTable"]["frame"]
    for thread in profile["threads"]:
        for stack in thread["samples"]["stack"]:
            chain = set()
            while stack is not None:
                chain.add(names[func_name[frame_func[stack_frame[stack]]]])
                stack = prefix[stack]
            yield chain


def count(path):
    """Return {phase: [total, stitched, truncated]} for one profile."""
    counts = {phase: [0, 0, 0] for phase in LEAVES}
    for chain in stacks(path):
        for phase, (leaf, foreign) in LEAVES.items():
            if leaf not in chain:
                continue
            counts[phase][0] += 1
            if foreign in chain:
                counts[phase][1] += 1
            elif MAIN not in chain:
                counts[phase][2] += 1
    return counts


def main(paths):
    if not paths:
        sys.exit(__doc__)
    print(f"{'profile':40} {'phase':5} {'total':>6} {'stitched':>8} {'truncated':>9} {'complete':>8}")
    any_stitched = False
    for path in paths:
        for phase, (total, stitched, truncated) in count(path).items():
            complete = total - stitched - truncated
            any_stitched |= stitched > 0
            print(f"{os.path.basename(path):40} {phase:5} {total:6} {stitched:8} {truncated:9} {complete:8}")
    if any_stitched:
        print("FAIL: found stitched stacks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
