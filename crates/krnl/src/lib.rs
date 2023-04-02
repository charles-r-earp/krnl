#![forbid(unsafe_op_in_unsafe_fn)]

pub use anyhow;
pub use krnl_core;
pub use krnl_macros as macros;
#[doc(hidden)]
pub use once_cell;

#[doc(inline)]
pub use krnl_core::scalar;

pub mod buffer;
pub mod device;
