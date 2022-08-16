use krnl_builder::ModuleBuilder;

type Result<T, E = anyhow::Error> = Result<T, E>;

fn main() -> Result<()> {
    let module = ModuleBuilder::new("test-device")
        .vulkan((1, 1))
        .build()?;
    Ok(())
}
