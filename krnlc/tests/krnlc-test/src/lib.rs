use krnl::{
    kernel::module,
};

#[module(
    dependencies(
        "\"krnl-core\" = { path = \"/home/charles/Documents/rust/krnl/krnl-core\" }"
    ),
)]
mod foo {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::mem::UnsafeMut;

    #[kernel(threads(1))]
    pub fn bar(#[global] y: &mut UnsafeMut<[u32]>) {
        unsafe {
            y.unsafe_mut()[0] = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krnl::{
        anyhow::Result,
        device::Device,
        buffer::Buffer,
        future::BlockableFuture,
    };

    #[test]
    fn foo_bar() -> Result<()> {
        let device = Device::new(0)?;
        let mut y = Buffer::from(vec![0, 0])
            .into_device(device.clone())?
            .block()?;
        let kernel = foo::bar::Kernel::builder()
            .build(device.clone())?;
        let builder = kernel.dispatch_builder(y.as_slice_mut())?
            .groups(1)?;
        builder.dispatch()?;
        let y = y.to_vec()?.block()?;
        panic!("{:?}", y)
    }
}
