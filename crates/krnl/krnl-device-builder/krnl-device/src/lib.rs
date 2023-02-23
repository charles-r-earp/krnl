#![no_std]
use glam::UVec3;
use spirv_std::spirv;

#[spirv(compute(threads(256)))]
pub fn fill(
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] y: &mut [f32],
) {
    let idx = group_id.x as usize * 256;
    y[idx] = 1f32;
}
