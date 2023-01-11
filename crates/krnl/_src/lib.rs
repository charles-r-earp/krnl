#![forbid(unsafe_op_in_unsafe_fn)]
#![allow(warnings)]

pub use anyhow;
pub use bincode;

pub use krnl_core;
pub use krnl_macros;

#[doc(inline)]
pub use krnl_core::scalar;

pub mod buffer;
pub mod device;
pub mod future;
pub mod kernel;

#[doc(hidden)]
pub mod __private {
    pub use bytemuck;
}
