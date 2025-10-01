#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(target_arch = "spirv")]
pub use spirv_std;

pub mod macros {
    pub use krnl_macros::{device_only, host_only, kernel};
}
use macros::host_only;

pub mod kernel;
pub mod scalar;

host_only! {
    pub mod buffer;
    pub mod context;

    #[derive(Clone, Debug)]
    pub enum Error {}
    pub type Result<T, E = Error> = std::result::Result<T, E>;

    mod kernels;
}

#[cfg(all(krnlc, krnlc_pkg = "krnl", target_arch = "spirv"))]
pub mod kernels;
