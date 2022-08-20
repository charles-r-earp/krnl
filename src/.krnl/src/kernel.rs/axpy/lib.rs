#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv),
    deny(warnings)
)]
#[cfg(target_arch = "spirv")]
extern crate spirv_std;
use krnl_core::{kernel, scalar::Scalar};
pub fn axpy<T: Scalar>(x: &T, alpha: T, y: &mut T) {
    *y += alpha * *x;
}
#[kernel(elementwise, threads(256))]
pub fn axpy_f32(x: &f32, alpha: f32, y: &mut f32) {
    axpy(x, alpha, y);
}
