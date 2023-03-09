use krnl::krnl_macros::module;

#[module]
pub mod foo {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::arch::{Length, UnsafeIndexMut};
    use krnl_core::krnl_macros::kernel;

    pub fn foo_impl(idx: usize, y: &(impl UnsafeIndexMut<usize, Output = u32> + Length)) {
        if idx < y.len() {
            unsafe {
                *y.unsafe_index_mut(idx) = 1;
            }
        }
    }

    #[kernel(threads(256))]
    pub fn foo(#[global] y: &mut [u32]) {
        foo_impl(global_id as usize, y);
    }

    /* generates */
    #[cfg(not(target_arch = "spirv"))]
    pub mod foo {
        use super::__krnl::{
            anyhow::{self, Result},
            buffer::SliceMut,
            device::{Device, Kernel as KernelBase, KernelBuilder as KernelBuilderBase},
            once_cell::sync::Lazy,
        };

        pub struct KernelBuilder {
            inner: &'static KernelBuilderBase,
        }

        pub fn builder() -> Result<KernelBuilder> {
            static BUILDER: Lazy<Result<KernelBuilderBase, String>> = Lazy::new(|| {
                let bytes = __krnl_kernel!(foo);
                KernelBuilderBase::from_bytes(bytes).map_err(|e| e.to_string())
            });
            match &*BUILDER {
                Ok(inner) => {
                    debug_assert!(inner.safe());
                    Ok(KernelBuilder { inner })
                }
                Err(e) => Err(anyhow::Error::msg(e)),
            }
        }

        impl KernelBuilder {
            pub fn build(&self, device: Device) -> Result<Kernel> {
                let inner = self.inner.build(device)?;
                Ok(Kernel { inner })
            }
        }

        pub struct Kernel {
            inner: KernelBase,
        }

        impl Kernel {
            pub fn dispatch(&self, groups: [u32; 1], y: SliceMut<u32>) -> Result<()> {
                unsafe { self.inner.dispatch(&groups, &[y.into()], &[]) }
            }
        }
    }

    /*
    #[cfg(target_arch = "spirv")]
    #[krnl_core::spirv_std::spirv(compute(threads(1)))]
    pub fn foo(
        #[spirv(global_invocation_id)] global_id: krnl_core::glam::UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] y: &mut [u32],
    ) {
        let y = unsafe { krnl_core::arch::UnsafeSliceMut::from_raw_parts(y, 0, 1) };
        let idx = global_id.x as usize;
        foo_impl(idx, y);
    }*/
}

#[cfg(test)]
mod tests {
    use super::*;
    use krnl::anyhow::Result;

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
    }

    #[test]
    fn foo_device() -> Result<()> {
        use krnl::{buffer::Buffer, device::Device};
        let device = Device::builder().build()?;
        let y_vec = vec![0; 100];
        let mut y = Buffer::from(y_vec).to_device(device.clone())?;
        let kernel = foo::foo::builder()?.build(device)?;
        let n = y.len() as u32;
        let threads = 256;
        let groups = n / threads + if n % threads != 0 { 1 } else { 0 };
        kernel.dispatch([groups], y.as_slice_mut())?;
        let y_vec = y.to_vec()?;
        assert!(y_vec.iter().all(|x| *x == 1));
        Ok(())
    }
}
