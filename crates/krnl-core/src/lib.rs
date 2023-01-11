#![cfg_attr(target_arch = "spirv", no_std)]
#![forbid(unsafe_op_in_unsafe_fn)]

pub use glam;
pub use half;
pub use num_traits;
pub use spirv_std;

pub mod mem;
pub mod scalar;
