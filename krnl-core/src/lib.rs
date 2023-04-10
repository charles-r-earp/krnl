/*!

# **krnl-core**
Shared core library for [**krnl**](https://docs.rs/krnl).

*/

#![cfg_attr(target_arch = "spirv", no_std, feature(asm_experimental_arch))]
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub extern crate glam;
pub extern crate half;
pub extern crate krnl_macros as macros;
pub extern crate num_traits;
pub extern crate spirv_std;

/// Buffers for use in kernels.
pub mod buffer;
/// Kernel structs passed to kernels.
pub mod kernel;
/// Scalars and numerical traits.
pub mod scalar;
