use krnl::krnl_macros::module;

#[module]
#[krnl(no_build)]
mod foo {
    use krnl::krnl_core::arch::{Length, UnsafeIndexMut};

    fn foo_impl(idx: usize, y: &(impl UnsafeIndexMut<usize, Output = u32> + Length)) {
        if idx < y.len() {
            unsafe {
                *y.unsafe_index_mut(idx) = 1;
            }
        }
    }

    #[cfg(target_arch = "spirv")]
    #[krnl_core::spirv_std::spirv(compute(threads(1)))]
    pub fn foo(
        #[spirv(global_invocation_id)] global_id: krnl_core::glam::UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] y: &mut [u32],
    ) {
        let y = unsafe { krnl_core::arch::UnsafeSliceMut::from_raw_parts(y, 0, 1) };
        let idx = global_id.x as usize;
        foo_impl(idx, y);
    }

    #[cfg(test)]
    #[test]
    pub fn test_foo() {
        use krnl::krnl_core::arch::UnsafeSliceMut;
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let mut y_vec = vec![0; 100];
        let y = UnsafeSliceMut::from(y_vec.as_mut_slice());
        (0..y.len())
            .into_par_iter()
            .for_each(|idx| foo_impl(idx, &y));
        assert!(y_vec.iter().all(|x| *x == 1));
    }
}
