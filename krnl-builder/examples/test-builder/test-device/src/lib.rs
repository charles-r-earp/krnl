#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv),
    deny(warnings),
)]

extern crate spirv_std;

use krnl_core::kernel;
#[cfg(target_arch = "spirv")]
use krnl_core::glam::UVec3;

#[kernel(threads(4, 8))]
pub fn fill(#[builtin] global_id: UVec3, y: &mut [u32], x: u32) {
    y[global_id.x as usize] = x;
}
