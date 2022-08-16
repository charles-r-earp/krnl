#![cfg_attr(target_arch = "spirv", no_std)]

pub mod scalar;

pub use krnl_macros::kernel;

#[doc(hidden)]
pub mod __private;
