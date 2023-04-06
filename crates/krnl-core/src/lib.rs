#![cfg_attr(target_arch = "spirv", no_std, feature(asm_experimental_arch))]
#![forbid(unsafe_op_in_unsafe_fn)]

pub extern crate glam;
pub extern crate half;
pub extern crate krnl_macros as macros;
pub extern crate num_traits;
pub extern crate spirv_std;

pub mod buffer;
pub mod kernel;
pub mod scalar;
