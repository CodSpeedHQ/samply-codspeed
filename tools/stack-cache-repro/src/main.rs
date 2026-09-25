use std::time::{Duration, Instant};

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
}

fn monotonic_ns() -> u64 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        clock_gettime(1, &mut ts); // 1 = CLOCK_MONOTONIC
    }
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

const FRAME_SIZE: usize = 2048;
const MAX_DEPTH: usize = 35; // 35 * 2048 = 71680 bytes (> 32000 bytes)
const WORK_PER_LEVEL_A: u64 = 40_000;
const WORK_IN_LEAF_A: u64 = 1_000_000;
const WORK_IN_LEAF_B: u64 = 1_000_000;

#[inline(never)]
fn a_leaf() {
    let mut x: u64 = 0x1111_2222;
    for i in 0..WORK_IN_LEAF_A {
        unsafe {
            std::arch::asm!(
                "add {0}, {1}",
                "xor {0}, 0x55",
                inout(reg) x,
                in(reg) i,
                options(nostack, nomem),
            );
        }
    }
    std::hint::black_box(x);
}

#[inline(never)]
fn a_recurse(depth: usize, max_depth: usize) {
    let mut buf = [0xAAu8; FRAME_SIZE];
    buf[0] = depth as u8;
    buf[FRAME_SIZE - 1] = (depth & 0xff) as u8;
    unsafe {
        std::arch::asm!("/* a_buf {0} */", in(reg) buf.as_ptr(), options(nostack));
    }

    // Spend time at every level in Phase A to populate stack_read_cache for all depths
    let mut x: u64 = 0x1234;
    for i in 0..WORK_PER_LEVEL_A {
        unsafe {
            std::arch::asm!(
                "add {0}, {1}",
                inout(reg) x,
                in(reg) i,
                options(nostack, nomem),
            );
        }
    }

    if depth < max_depth {
        a_recurse(depth + 1, max_depth);
    } else {
        a_leaf();
    }
}

#[inline(never)]
fn a_root(max_depth: usize) {
    unsafe {
        std::arch::asm!("/* a_root */", options(nostack));
    }
    a_recurse(0, max_depth);
}

#[inline(never)]
fn b_leaf() {
    let mut x: u64 = 0x3333_4444;
    for i in 0..WORK_IN_LEAF_B {
        unsafe {
            std::arch::asm!(
                "add {0}, {1}",
                "xor {0}, 0xaa",
                inout(reg) x,
                in(reg) i,
                options(nostack, nomem),
            );
        }
    }
    std::hint::black_box(x);
}

#[inline(never)]
fn b_recurse(depth: usize, max_depth: usize) {
    let mut buf = [0xBBu8; FRAME_SIZE];
    buf[0] = depth as u8;
    buf[FRAME_SIZE - 1] = (depth & 0xff) as u8;
    unsafe {
        std::arch::asm!("/* b_buf {0} */", in(reg) buf.as_ptr(), options(nostack));
    }

    // Phase B does negligible work during recursion traversal,
    // so no samples land at shallow depths to overwrite the cache!
    if depth < max_depth {
        b_recurse(depth + 1, max_depth);
    } else {
        b_leaf();
    }
}

#[inline(never)]
fn b_root(max_depth: usize) {
    unsafe {
        std::arch::asm!("/* b_root */", options(nostack));
    }
    b_recurse(0, max_depth);
}

#[inline(never)]
fn run_phase_a(duration: Duration) {
    let start = Instant::now();
    while start.elapsed() < duration {
        a_root(MAX_DEPTH);
    }
}

#[inline(never)]
fn run_phase_b(duration: Duration) {
    let start = Instant::now();
    while start.elapsed() < duration {
        b_root(MAX_DEPTH);
    }
}

fn main() {
    // Optional first argument: milliseconds per phase (default 3500).
    let phase_ms = std::env::args()
        .nth(1)
        .map_or(3500, |ms| ms.parse().expect("phase duration in ms"));
    let phase_duration = Duration::from_millis(phase_ms);

    let t0 = monotonic_ns();
    println!("PHASE_A_START: {t0}");
    run_phase_a(phase_duration);
    let t1 = monotonic_ns();
    println!("PHASE_A_END: {t1}");

    let t2 = monotonic_ns();
    println!("PHASE_B_START: {t2}");
    run_phase_b(phase_duration);
    let t3 = monotonic_ns();
    println!("PHASE_B_END: {t3}");

    println!("DONE");
}
