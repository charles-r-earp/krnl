#[cfg(feature = "bindings")]
pub mod bindings;
#[cfg(feature = "cli")]
pub mod cli;
#[cfg(feature = "print")]
mod print;
#[cfg(feature = "rust-in")]
pub mod rust_in;

pub mod reflect;
mod scalar;
mod spirv;

#[cfg(feature = "cli")]
const VERSION_AND_SHA: &str = {
    if !env!("CARGO_PKG_VERSION_PRE").is_empty() {
        concat!(env!("CARGO_PKG_VERSION"), " ", env!("VERGEN_GIT_SHA"))
    } else {
        env!("CARGO_PKG_VERSION")
    }
};
