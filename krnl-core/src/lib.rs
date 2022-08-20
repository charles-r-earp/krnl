#![cfg_attr(target_arch = "spirv", no_std)]

pub use glam;

#[doc(inline)]
pub use krnl_types::scalar;

#[doc(inline)]
pub use krnl_macros::kernel;

pub mod mem;

#[doc(hidden)]
pub mod __private;
