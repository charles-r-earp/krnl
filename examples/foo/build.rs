use krnlc::{bindings::BindingsBuilder, rust_in::ModuleBuilder};

fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")
        .unwrap()
        .to_ascii_lowercase();
    if target_arch != "spirv" {
        let spirv = ModuleBuilder::new().build();
        BindingsBuilder::from_spirv(spirv).emit().unwrap();
    }
}
