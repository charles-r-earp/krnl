use krnl::{buffer::Buffer, device::Device, macros::module};

#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    #[kernel]
    pub fn vector_sum(#[item] a: f32, #[item] b: &mut f32) {
        *b += a
    }
}


fn main() -> Result<(), Box<dyn std::error::Error>>{
    let a = vec![1f32; 10];
    let b = vec![2f32; 10];
    let device = Device::builder().build()?;
    let a = Buffer::from(a).into_device(device.clone())?;
    let mut b = Buffer::from(b).into_device(device.clone())?;
    kernels::vector_sum::builder()?.build(device)?.dispatch(a.as_slice(), b.as_slice_mut())?;
    let b = b.into_vec()?;
    println!("{b:?}");
    Ok(())
}
