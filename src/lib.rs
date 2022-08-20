#![forbid(unsafe_op_in_unsafe_fn)]

pub use krnl_core;

#[doc(inline)]
pub use krnl_core::scalar;

pub mod result {
    pub type Result<T, E = anyhow::Error> = std::result::Result<T, E>;
}

pub mod buffer;
pub mod device;
pub mod future;
pub mod kernel;

#[doc(hidden)]
pub mod __private {
    pub use bincode;
}
