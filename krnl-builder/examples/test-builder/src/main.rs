use krnl_builder::{version::Version, ModuleBuilder};

type Result<T, E = anyhow::Error> = std::result::Result<T, E>;

fn main() -> Result<()> {
    let module = ModuleBuilder::new("test-device")
        .vulkan(Version::from_major_minor(1, 1))
        .build()?;
    let kernel = module.kernel_info("one")?;
    dbg!(kernel);
    Ok(())
}
