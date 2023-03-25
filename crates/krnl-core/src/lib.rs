#![cfg_attr(target_arch = "spirv", no_std, feature(asm_experimental_arch))]
#![forbid(unsafe_op_in_unsafe_fn)]

pub use dry;
pub use glam;
pub use half;
pub use krnl_macros;
pub use krnl_macros as macros;
pub use num_traits;
pub use paste;
pub use spirv_std;

pub mod buffer;
pub mod kernel;
pub mod scalar;
