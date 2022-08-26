#![cfg_attr(target_arch = "spirv", no_std, feature(asm_experimental_arch))]
#![cfg_attr(
    all(target_arch = "spirv", feature = "spirv-panic"),
    feature(lang_items)
)]
#![forbid(unsafe_op_in_unsafe_fn)]

pub use glam;

#[doc(inline)]
pub use krnl_types::scalar;

#[doc(inline)]
pub use krnl_macros::kernel;

pub mod mem;
pub mod ops;
pub mod slice;

#[doc(hidden)]
pub mod __private;

#[cfg(all(target_arch = "spirv", feature = "spirv-panic"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[cfg(all(target_arch = "spirv", feature = "spirv-panic"))]
#[lang = "eh_personality"]
extern "C" fn rust_eh_personality() {}
