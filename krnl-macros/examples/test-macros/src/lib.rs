use krnl::kernel::module;

#[module]
#[krnl(build = true)]
mod foo {
    use krnl_core::kernel;

    #[kernel(vulkan(1, 1), threads(1), elementwise)]
    pub fn bar(y: &mut u32, x: u32) {
        *y = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krnl::{anyhow::Result, buffer::Buffer, device::Device, future::BlockableFuture};

    #[test]
    fn bar() -> Result<()> {
        let device = Device::new(0)?;
        let mut y = Buffer::from_vec(vec![1])
            .into_device(device.clone())?
            .block()?;
        foo::bar::build(device)?.dispatch(y.as_slice_mut(), 5)?;
        let y = y.to_vec()?.block()?;
        assert_eq!(y[0], 5);
        Ok(())
    }
}
