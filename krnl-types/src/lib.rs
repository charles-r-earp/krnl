#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(not(target_arch = "spirv"))]
pub mod kernel;
pub mod scalar;
#[cfg(not(target_arch = "spirv"))]
pub mod version;

#[doc(hidden)]
pub mod __private;
