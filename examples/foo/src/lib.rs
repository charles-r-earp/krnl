#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(not(target_arch = "spirv"))]
use krnl::{
    buffer::Buffer,
    context::{Context, Device},
    kernel::KernelDef,
};
use krnl::{macros::kernel, scalar::Scalar};
use num_traits::{Num, NumAssign};

fn axpy_impl<T: Scalar + Num + NumAssign>(alpha: T, x: T, y: &mut T) {
    *y += alpha * x;
}

#[kernel]
fn axpy<#[kernel(impl=[f32])] T: Scalar + Num + NumAssign>(
    alpha: T,
    #[kernel(item)] x: T,
    #[kernel(item)] y: &mut T,
) {
    axpy_impl(alpha, x, y);
}

#[cfg(not(target_arch = "spirv"))]
pub fn main() {
    let context = if cfg!(feature = "device") {
        Context::Device(Device::builder().build().unwrap())
    } else {
        Context::Host
    };

    let alpha = 2f32;
    let x = Buffer::from(vec![1f32])
        .into_context(context.clone())
        .unwrap();
    let mut y = Buffer::zeros(context.clone(), x.len()).unwrap();
    if let Some((x, y)) = x
        .as_slice()
        .into_host_slice()
        .zip(y.as_slice_mut().into_host_slice_mut())
    {
        for (x, y) in x.iter().copied().zip(y) {
            axpy_impl(alpha, x, y);
        }
    } else {
        axpy::builder(())
            .build(context)
            .unwrap()
            .exec((alpha, x.as_slice(), y.as_slice_mut()))
            .unwrap();
    }
    let y = y.into_vec().unwrap();
    dbg!(y);
}
