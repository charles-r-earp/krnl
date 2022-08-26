use krnl::kernel::module;

#[module(target("vulkan1.1"))]
mod kernels {
    #[allow(unused_imports)]
    use krnl_core::{
        glam::UVec3,
        kernel,
        mem::{GlobalMut, GroupUninitMut},
        slice::{Slice, SliceMut},
    };

    #[kernel(threads(1))]
    pub fn group_test(
        #[builtin] global_id: UVec3,
        #[builtin] global_threads: UVec3,
        x: &Slice<u32>,
        #[group]
        x_group: &mut GroupUninitMut<[u32; 1]>,
        y: &mut GlobalMut<SliceMut<u32>>,
    ) {
        let y = unsafe { y.global_mut() };
        let x_group = unsafe { x_group.group_uninit_mut() };
        let mut idx = global_id.x;
        while (idx as usize) < y.len() {
            x_group[0] = x[idx as usize];
            // unsafe { group_barrier(); }
            y[idx as usize] = x_group[0];
            idx += global_threads.x;
        }
    }

    /*
    #[kernel(threads(256))]
    pub fn copy_u32(#[builtin] global_id: UVec3, #[builtin] global_threads: UVec3, x: &Slice<u32>, y: &mut GlobalMut<SliceMut<u32>>) {
        let y = unsafe {
            y.global_mut()
        };
        let mut idx = global_id.x;
        while (idx as usize) < y.len() {
            y[idx as usize] = x[idx as usize];
            idx += global_threads.x;
        }
    }

    #[kernel(threads(256), elementwise)]
    pub fn fill_elementwise_u32(y: &mut u32, x: u32) {
        *y = x;
    }*/
}

#[test]
fn test_module() {
    kernels::module().unwrap();
}
