#![cfg_attr(target_arch = "spirv", no_std, feature(asm_experimental_arch))]

use core::cell::UnsafeCell;

use krnl::macros::kernel;
#[cfg(not(target_arch = "spirv"))]
use krnl::{
    buffer::{Slice, SliceMut},
    kernel::KernelDef,
};

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

///```no_run
/// use krnl::{macros::kernel, kernel::KernelDef, context::Context};
///
/// #[kernel(no_build)]
/// pub fn builtins(
///   #[kernel(global_thread_id)] global_thread_id: usize,
///   #[kernel(thread_id)] thread_id: usize,
/// ) {}
///
/// let kernel = builtins::builder(()).build(Context::Host).unwrap();
/// kernel.exec(()).unwrap();
///```
struct _Builtins {}

///```compile_fail
/// use krnl::{macros::kernel, kernel::KernelDef, context::Context};
///
/// #[kernel(no_build)]
/// pub fn invalid_builtin_type(
///   #[kernel(global_thread_id)] invalid: i32,
/// ) {}
///
/// let kernel = invalid_builtin_type::builder(()).build(Context::Host).unwrap();
/// kernel.exec(()).unwrap();
///```
struct _InvalidBuiltinType {}

///```compile_fail
/// use krnl::{macros::kernel, kernel::KernelDef, context::Context};
///
/// #[kernel(no_build)]
/// pub fn invalid_builtin(
///   #[kernel(invalid)] invalid: usize,
/// ) {}
///
/// let kernel = invalid_builtin::builder(()).build(Context::Host).unwrap();
/// kernel.exec(()).unwrap();
///```
struct _InvalidBuiltin {}

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

#[kernel]
unsafe fn group_buffer_const_64(
    #[kernel(thread_id)] thread_id: usize,
    x: &[u32],
    y: &[UnsafeCell<u32>],
) {
    use krnl::spirv_std::{
        self,
        arch::{IndexUnchecked, workgroup_memory_barrier_with_group_sync as group_barrier},
    };

    #[kernel(group, len = 64)]
    let x_group: &[UnsafeCell<u32>];

    unsafe {
        *x_group[thread_id].get() = *x.index_unchecked(thread_id);
        group_barrier();
    }

    if thread_id == 0 {
        let mut acc = 0;
        for i in 0..64 {
            acc += unsafe { *x_group[i].get() };
        }
        unsafe {
            *y.index_unchecked(0).get() = acc;
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
pub unsafe fn _group_buffer_const_64(x: Slice<u32>, y: SliceMut<u32>) {
    assert!(x.len() == 64);
    assert!(y.len() == 1);
    let kernel = group_buffer_const_64::builder(())
        .threads(64)
        .build(y.context())
        .unwrap();
    unsafe {
        kernel.groups(1).exec((x, y)).unwrap();
    }
}

#[kernel]
unsafe fn group_buffer_spec(
    #[kernel(spec)] n: u32,
    #[kernel(thread_id)] thread_id: usize,
    x: &[u32],
    y: &[UnsafeCell<u32>],
) {
    use krnl::spirv_std::arch::workgroup_memory_barrier_with_group_sync as group_barrier;

    let n = n as usize;

    #[kernel(group, len = n)]
    let x_group: &[UnsafeCell<u32>];

    unsafe {
        *x_group[thread_id].get() = x[thread_id];
        group_barrier();
    }
    if thread_id == 0 {
        let mut acc = 0;
        for i in 0..n {
            acc += unsafe { *x_group[i].get() };
        }
        unsafe {
            *y[0].get() = acc;
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
pub unsafe fn _group_buffer_spec(x: Slice<u32>, y: SliceMut<u32>) {
    let n = x.len();
    assert!(y.len() == 1);
    let kernel = group_buffer_spec::builder((n as u32,))
        .threads(n)
        .build(y.context())
        .unwrap();
    unsafe {
        kernel.groups(1).exec((x, y)).unwrap();
    }
}
