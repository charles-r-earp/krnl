use krnl_builder::ModuleBuilder;

type Result<T, E = anyhow::Error> = std::result::Result<T, E>;

fn main() -> Result<()> {
    let module = ModuleBuilder::new("test-device", "vulkan1.1")
        .build()?;
    dbg!(&module);
    let kernel = module.kernel("fill")?;
    dbg!(kernel);
    Ok(())
}
