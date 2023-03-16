use krnl::krnl_macros::module;

#[module]
pub mod foo {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    //use krnl_core::arch::{Length, UnsafeIndexMut};
    use krnl_core::krnl_macros::kernel;

    /*
    pub fn foo_impl(idx: usize, y: &(impl UnsafeIndexMut<usize, Output = u32> + Length)) {
        if idx < y.len() {
            unsafe {
                *y.unsafe_index_mut(idx) = 1;
            }
        }
    }

    #[kernel(threads(256))]
    pub fn foo(#[global] y: &mut [u32]) {
        foo_impl(kernel.global_id() as usize, y);
    }*/

    #[kernel(threads(256))]
    pub fn foo_itemwise(#[item] y: &mut u32) {
        *y = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /*#[test]
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
        let kernel = foo::foo_itemwise::builder()
            .unwrap()
            .build(device.clone())
            .unwrap();
        kernel.dispatch(y.as_slice_mut()).unwrap();
        device.wait().unwrap();
        let y_vec = y.to_vec().unwrap();
        assert!(y_vec.iter().all(|x| *x == 1));
    }

    /*
    #[test]
    fn bar() {
        use krnl::device::Device;
        let device = Device::builder().build().unwrap();
        let kernel = foo::bar::builder().unwrap().build(device.clone()).unwrap();
        kernel.dispatch([1]).unwrap();
        device.wait().unwrap();
    }
    */
}
