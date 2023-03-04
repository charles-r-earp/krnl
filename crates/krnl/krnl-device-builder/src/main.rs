use spirv_builder::SpirvBuilder;

fn main() {
    let pathbuf = SpirvBuilder::new("krnl-device", "spirv-unknown-vulkan1.2")
        .deny_warnings(true)
        .build()
        .unwrap()
        .module
        .unwrap_single()
        .to_path_buf();
    std::fs::copy(&pathbuf, "fill.spv").unwrap();
}
