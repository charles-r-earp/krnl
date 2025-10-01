fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")
        .unwrap()
        .to_ascii_lowercase();
    if target_arch != "spirv" {
        krnlc::bindings::BindingsBuilder::default().emit().unwrap();
    }
}
