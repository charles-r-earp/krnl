use krnl::{device::Device, kernel::{module, Kernel}};

#[module(vulkan(1, 0))]
mod kernels {
    use krnl_core::kernel;

    #[kernel(threads(1))] pub fn main() {}
}

#[test]
fn list_devices() -> anyhow::Result<()> {
    for i in 0 .. 3 {
        Device::new(i)?;
    }
    Ok(())
}

#[test]
fn test_module() -> anyhow::Result<()> {
    let device = Device::new(0)?;
    let info = kernels::module()?.kernel_info("main")?;
    let kernel = Kernel::builder(device, info)
        .build()?;
    Ok(())
}
