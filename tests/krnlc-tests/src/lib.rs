use krnl::macros::module;

#[module]
pub mod foo {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::buffer::{UnsafeIndex, UnsafeSlice};
    use krnl_core::macros::kernel;

    pub fn foo_impl(idx: usize, y: UnsafeSlice<u32>) {
        if idx < y.len() {
            unsafe {
                *y.unsafe_index_mut(idx) = 1;
            }
        }
    }

    #[kernel(threads(256))]
    pub fn foo(#[global] y: UnsafeSlice<u32>) {
        foo_impl(kernel.global_id() as usize, y);
    }

    #[kernel(threads(256))]
    pub fn foo_itemwise(#[item] y: &mut u32) {
        *y = 1;
    }
}

#[module]
pub mod bar {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    #[cfg(target_arch = "spirv")]
    use krnl_core::buffer::UnsafeIndex;
    use krnl_core::macros::kernel;

    #[kernel(threads(TS))]
    pub fn bar<#[spec] const TS: u32>(
        #[global] x: Slice<u32>,
        #[group] x_group: UnsafeSlice<u32, { (2 * TS) as usize }>,
        #[global] y: UnsafeSlice<u32>,
    ) {
        use krnl_core::spirv_std::{self, arch::workgroup_memory_barrier, macros::debug_printfln};

        let mut global_id = kernel.global_id() as usize;
        let thread_id = kernel.thread_id() as usize;
        let group_id = kernel.group_id() as usize;

        while global_id < x.len() {
            unsafe {
                *x_group.unsafe_index_mut(thread_id) += x[global_id];
            }
            global_id += kernel.global_threads() as usize;
        }
        unsafe {
            workgroup_memory_barrier();
        }
        if group_id < y.len() && thread_id == 0 {
            let mut acc = 0;
            for i in 0..x_group.len() {
                unsafe {
                    acc += *x_group.unsafe_index(i);
                }
            }
            unsafe {
                *y.unsafe_index_mut(group_id) = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
    #[test]
    fn foo_host() {
        use krnl::krnl_core::arch::{Length, UnsafeSliceMut};
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let mut y_vec = vec![0; 100];
        let y = UnsafeSliceMut::from(y_vec.as_mut_slice());
        (0..y.len())
            .into_par_iter()
            .for_each(|idx| foo::foo_impl(idx, &y));
        assert!(y_vec.iter().all(|x| *x == 1));
    }*/

    #[test]
    fn foo_device() {
        use krnl::{buffer::Buffer, device::Device};
        let device = Device::builder().build().unwrap();
        let y_vec = vec![0; 100];
        let mut y = Buffer::from(y_vec).to_device(device.clone()).unwrap();
        let kernel = foo::foo::builder().unwrap().build(device.clone()).unwrap();
        kernel
            .with_global_threads([y.len() as u32])
            .dispatch(y.as_slice_mut())
            .unwrap();
        device.wait().unwrap();
        let y_vec = y.to_vec().unwrap();
        assert!(y_vec.iter().all(|x| *x == 1));
    }

    #[test]
    fn foo_itemwise_device() {
        use krnl::{buffer::Buffer, device::Device};
        let device = Device::builder().build().unwrap();
        let y_vec = vec![0; 100];
        let mut y = Buffer::from(y_vec).to_device(device.clone()).unwrap();
        let kernel = foo::foo_itemwise::builder()
            .unwrap()
            .build(device.clone())
            .unwrap();
        kernel.dispatch(y.as_slice_mut()).unwrap();
        device.wait().unwrap();
        let y_vec = y.to_vec().unwrap();
        assert!(y_vec.iter().all(|x| *x == 1));
    }

    #[test]
    fn bar_device() {
        use krnl::{
            buffer::{Buffer, Slice},
            device::Device,
        };
        let device = Device::builder().build().unwrap();
        let x_vec = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let y_vec = vec![0];
        let x = Slice::from(x_vec.as_slice())
            .to_device(device.clone())
            .unwrap();
        let mut y = Buffer::from(y_vec).to_device(device.clone()).unwrap();
        let kernel = bar::bar::builder()
            .unwrap()
            .specialize((x.len() / 2) as u32)
            .unwrap()
            .build(device.clone())
            .unwrap();
        kernel
            .with_groups([1])
            .dispatch(x.as_slice(), y.as_slice_mut())
            .unwrap();
        device.wait().unwrap();
        let y_vec = y.to_vec().unwrap();
        assert_eq!(y_vec[0], x_vec.iter().copied().sum());
    }
}
