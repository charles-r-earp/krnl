#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv),
    deny(warnings)
)]

use krnl_core::kernel;

#[kernel(threads(1), elementwise, capabilities("Int64"))]
pub fn one(y: &mut u64) {
    *y = 1;
}
