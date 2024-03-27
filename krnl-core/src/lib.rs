//! Shared core library for [krnl](https://docs.rs/krnl).
#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(asm_experimental_arch, min_specialization)
)]
#![cfg_attr(doc_cfg, feature(doc_cfg, doc_auto_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]

/// half
pub extern crate half;
/// krnl-macros
pub extern crate krnl_macros as macros;
/// num-traits
pub extern crate num_traits;
/// spirv-std
pub extern crate spirv_std;

/// Buffers for use in kernels.
pub mod buffer;
/// Kernel structs passed to kernels.
#[cfg_attr(doc_cfg, doc(cfg(target_arch = "spirv")))]
pub mod kernel;
/// Numerical types.
pub mod scalar;
