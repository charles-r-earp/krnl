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

    #[kernel(threads(256))]
    pub fn bar(#[builtin] global_index: u32, #[global] y: &mut UnsafeMut<[u32]>, #[push] x: u32) {
        unsafe {
            y.unsafe_mut()[global_index as usize] = x;
        }
    }

    #[kernel(threads(256), for_each)]
    pub fn bar_for_each(#[item] y: &mut u32, #[push] x: u32) {
        *y = x;
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
        let mut y = Buffer::from(vec![0; 4])
            .into_device(device.clone())?
            .block()?;
        let kernel = foo::bar::Kernel::builder()
            .build(device.clone())?;
        let builder = kernel.dispatch_builder(y.as_slice_mut(), 1u32)?
            .groups(1)?;
        builder.dispatch()?;
        let y = y.to_vec()?.block()?;
        assert_eq!(y, vec![1u32; 4]);
        Ok(())
    }

    #[test]
    fn foo_bar_for_each() -> Result<()> {
        let device = Device::new(0)?;
        let mut y = Buffer::from(vec![0; 4])
            .into_device(device.clone())?
            .block()?;
        let kernel = foo::bar_for_each::Kernel::builder()
            .build(device.clone())?;
        let builder = kernel.dispatch_builder(y.as_slice_mut(), 1u32)?
            .groups(1)?;
        builder.dispatch()?;
        let y = y.to_vec()?.block()?;
        assert_eq!(y, vec![1u32; 4]);
        Ok(())
    }
}
