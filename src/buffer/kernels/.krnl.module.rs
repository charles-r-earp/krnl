
pub const fn bytes(h: u64) -> [u8; 1] {
    if h != 1 {
        panic!("module has been modified, rebuild with `cargo +nightly build --features krnl/build`");
    }
    [0]
}
