#![no_std]
use glam::UVec3;
use spirv_std::spirv;

#[spirv(compute(threads(256)))]
pub fn fill(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] y: &mut [f32],
) {
    let idx = global_id.x as usize;
    use spirv_std::arch::IndexUnchecked;
    unsafe {
        *y.index_unchecked_mut(idx) = 1f32;
    }
}
