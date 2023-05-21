/*!

# **krnl-core**
Shared core library for [**krnl**](https://docs.rs/krnl).

*/

#![cfg_attr(target_arch = "spirv", no_std, feature(asm_experimental_arch))]
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[allow(missing_docs)]
pub extern crate glam;
#[allow(missing_docs)]
pub extern crate half;
#[allow(missing_docs)]
pub extern crate krnl_macros as macros;
#[allow(missing_docs)]
pub extern crate num_traits;
#[allow(missing_docs)]
pub extern crate spirv_std;

/// Buffers for use in kernels.
pub mod buffer;
/// Kernel structs passed to kernels.
pub mod kernel;
/// Scalars and numerical traits.
pub mod scalar;
