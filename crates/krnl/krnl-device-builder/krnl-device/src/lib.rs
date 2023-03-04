#![no_std]
use krnl_core::{
    glam::UVec3,
    mem::{ArrayLength, UnsafeIndexMut, UnsafeSliceMut},
    spirv_std::spirv,
};

#[cfg(target_arch = "krnl")]
#[kernel(threads(256))]
pub fn fill(#[builtin] global_id: UVec3, #[global] y: &mut [f32]) {
    let idx = global_id.x as usize;
    if idx < y.len() {
        unsafe {
            *y.unsafe_index_mut(idx) = 1f32;
        }
    }
}

#[spirv(compute(threads(256)))]
pub fn fill(
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(local_invocation_id)] thread_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] y: &mut [f32],
) {
    let threads = UVec3::new(256, 1, 1);
    let global_id = group_id * threads + thread_id;
    let ref mut y = unsafe { UnsafeSliceMut::from_raw_parts(y, 0, 1) };
    fn fill_impl(
        #[builtin] global_id: UVec3,
        y: &mut (impl ArrayLength + UnsafeIndexMut<usize, Output = f32>),
    ) {
        let idx = global_id.x as usize;
        if idx < y.len() {
            unsafe {
                *y.unsafe_index_mut(idx) = 1f32;
            }
        }
    }
    fill_impl(global_id, y);
}

#[cfg(target_arch = "krnl")]
#[kernel(for_each, threads(256))]
pub fn fill_for_each(#[item] y: &mut f32) {
    *y = 1f32;
}

#[spirv(compute(threads(256)))]
pub fn fill_for_each(
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(local_invocation_id)] thread_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] y: &mut [f32],
) {
    let threads = UVec3::new(256, 1, 1);
    let idx = (group_id.x * threads.x + thread_id.x) as usize;
    fn fill_for_each_impl(y: &mut f32) {
        *y = 1f32;
    }
    if idx < y.len() {
        let mut y = unsafe { UnsafeSliceMut::from_raw_parts(y, 0, 1) };
        let y = unsafe { y.unsafe_index_mut(idx) };
        fill_for_each_impl(y);
    }
}
