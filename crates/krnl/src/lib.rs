#![forbid(unsafe_op_in_unsafe_fn)]

pub extern crate anyhow;
pub extern crate krnl_core;
pub extern crate krnl_macros as macros;
#[doc(hidden)]
pub extern crate once_cell;

#[doc(inline)]
pub use krnl_core::scalar;

pub mod buffer;
pub mod device;
