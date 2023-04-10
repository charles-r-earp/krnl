use spirv_builder::SpirvBuilder;

fn main() {
    let path = SpirvBuilder::new("shader", "spirv-unknown-vulkan1.2")
        .deny_warnings(true)
        .build()
        .unwrap()
        .module
        .unwrap_single()
        .to_path_buf();
    std::fs::copy(path, "shader.spv").unwrap();    
}
