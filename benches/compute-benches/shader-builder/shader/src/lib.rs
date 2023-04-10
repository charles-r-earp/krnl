#![no_std]
use spirv_std::spirv;
use glam::UVec3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConsts {
    n: u32,
    alpha: f32,
}

#[spirv(compute(threads(128)))]
pub fn saxpy(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    x: &[f32],
     #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &PushConsts,
) {
    let PushConsts {
        n,
        alpha
    } = *push_consts;
    let idx = global_id.x as usize;
    if idx < n as usize {
        y[idx] += alpha * x[idx];
    } 
}
