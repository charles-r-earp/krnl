#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]
#![deny(warnings)]

use krnl_core::kernel;

#[kernel(threads(1))] pub fn foo() {}
