// Fixture for Rust async function names in DWARF. `no_std` keeps the binary small.
// Build: rustc +stable --edition 2021 -C opt-level=2 -C debuginfo=2 -C panic=abort -C force-frame-pointers=yes async-rust.rs -o async-rust
#![no_std]
#![no_main]

use core::future::Future;
use core::hint::black_box;
use core::pin::{pin, Pin};
use core::task::{Context, Poll, Waker};

const ROUNDS: u64 = 500;
const RECORD_COUNT: u64 = 100_000;
const RECORDS_PER_YIELD: u64 = 1024;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

struct YieldNow(bool);

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            return Poll::Ready(());
        }
        self.0 = true;
        Poll::Pending
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
    }
}

#[inline(never)]
fn mix(value: u64) -> u64 {
    let mut hash = black_box(value) ^ 0x9e37_79b9_7f4a_7c15;
    for _ in 0..32 {
        hash = hash.rotate_left(5).wrapping_mul(0x100_0000_01b3);
    }
    hash
}

async fn parse_records(count: u64) -> u64 {
    let mut acc = 0u64;
    for index in 0..count {
        acc = acc.wrapping_add(mix(index));
        if index % RECORDS_PER_YIELD == 0 {
            YieldNow(false).await;
        }
    }
    acc
}

async fn checksum(count: u64) -> u64 {
    let prefix = async {
        let mut acc = 0u64;
        for index in 0..count / 2 {
            acc ^= mix(index ^ acc);
        }
        YieldNow(false).await;
        acc
    }
    .await;
    prefix ^ parse_records(count).await
}

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    let mut total = 0u64;
    for round in 0..ROUNDS {
        total = total.wrapping_add(block_on(checksum(black_box(RECORD_COUNT + round))));
    }
    black_box(total);
    0
}
