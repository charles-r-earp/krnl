#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(not(target_arch = "spirv"))]
pub mod kernel;
#[cfg(not(target_arch = "spirv"))]
pub mod version;
pub mod scalar;

#[doc(hidden)]
pub mod __private;
