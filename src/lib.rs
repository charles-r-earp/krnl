#![forbid(unsafe_op_in_unsafe_fn)]

#[doc(hidden)]
pub use krnl_core as core;

#[doc(inline)]
pub use krnl_core::scalar;

mod result {
    pub(crate) type Result<T, E = anyhow::Error> = std::result::Result<T, E>;
}

pub mod buffer;
pub mod device;
pub mod future;
pub mod kernel;
