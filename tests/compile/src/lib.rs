#![cfg_attr(target_arch = "spirv", no_std)]

use krnl::macros::kernel;
#[cfg(not(target_arch = "spirv"))]
use krnl::{buffer::SliceMut, kernel::KernelDef};

#[kernel]
fn spec_constants(#[kernel(spec)] x: u32, #[kernel(item)] y: &mut u32) {
    *y = x;
}

#[cfg(not(target_arch = "spirv"))]
fn _spec_constants(x: u32, y: SliceMut<u32>) {
    let kernel = spec_constants::builder((x,)).build(y.context()).unwrap();
    kernel.exec((y,)).unwrap();
}

#[kernel]
fn spec_constants_array(#[kernel(spec)] x: [u32; 3], #[kernel(item)] y: &mut [u32; 4]) {
    let [x1, x2, x3] = x;
    *y = [x1, x2, x3, 0];
}

#[cfg(not(target_arch = "spirv"))]
fn _spec_constants_array(x: [u32; 3], y: SliceMut<[u32; 4]>) {
    let kernel = spec_constants_array::builder((x,))
        .build(y.context())
        .unwrap();
    kernel.exec((y,)).unwrap();
}

#[kernel]
fn push_constants(x: u32, #[kernel(item)] y: &mut u32) {
    *y = x;
}

#[cfg(not(target_arch = "spirv"))]
fn _push_constants(x: u32, y: SliceMut<u32>) {
    let kernel = push_constants::builder(()).build(y.context()).unwrap();
    kernel.exec((x, y)).unwrap();
}

#[kernel]
fn push_constants_array(x: [u32; 3], #[kernel(item)] y: &mut [u32; 4]) {
    let [x1, x2, x3] = x;
    *y = [x1, x2, x3, 0];
}

#[cfg(not(target_arch = "spirv"))]
fn _push_constants_array(x: [u32; 3], y: SliceMut<[u32; 4]>) {
    let kernel = push_constants_array::builder(())
        .build(y.context())
        .unwrap();
    kernel.exec((x, y)).unwrap();
}

///```no_run
/// use krnl::{macros::kernel, kernel::KernelDef, context::Context};
///
/// #[kernel(no_build)]
/// pub fn safe_kernel() {}
///
/// let kernel = safe_kernel::builder(()).build(Context::Host).unwrap();
/// kernel.exec(()).unwrap();
///```
struct _SafeKernel {}

///```compile_fail
/// use krnl::{macros::kernel, kernel::KernelDef, context::Context};
///
/// #[kernel(no_build)]
/// pub unsafe fn unsafe_kernel() {}
///
/// let kernel = unsafe_kernel::builder(()).build(Context::Host).unwrap();
/// kernel.exec(()).unwrap();
///```
struct _UnsafeKernel {}
