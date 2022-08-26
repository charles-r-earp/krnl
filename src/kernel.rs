/*!
```text
use krnl::{krnl_core, kernel::module, scalar::Scalar, buffer::{Slice, SliceMut}, result::Result};

#[module(
    target("vulkan1.1"),
    dependency("krnl-core", path = "krnl-core"),
    dependency("spirv-std", git = "https://github.com/EmbarkStudios/rust-gpu"),
    attr(cfg_attr(
        target_arch = "spirv",
        no_std,
        feature(register_attr),
        register_attr(spirv),
        deny(warnings),
    )),
)]
pub mod axpy {
    #[cfg(target_arch = "spirv")]
    extern crate spirv_std;

    use krnl_core::{scalar::Scalar, kernel};

    pub fn axpy<T: Scalar>(x: &T, alpha: T, y: &mut T) {
        *y += alpha * *x;
    }

    #[kernel(elementwise, threads(256))]
    pub fn axpy_f32(x: &f32, alpha: f32, y: &mut f32) {
        axpy(x, alpha, y);
    }
}

fn main() -> Result<()> {
    axpy::module().unwrap();
    Ok(())
}
```
*/
use core::marker::PhantomData;
use krnl_core::__private::raw_module::{
    PushInfo, RawKernelInfo, RawModule, Safety, SliceInfo, Spirv,
};
use std::{collections::HashMap, sync::Arc};

#[doc(inline)]
pub use krnl_types::kernel::{KernelInfo, Module};

#[doc(inline)]
pub use krnl_macros::module;

/*
pub mod error {
    use super::*;

    pub struct KernelNotFound {
        kernel: String,
    }

    pub struct KernelThreadsError {
        threads: Vec<u32>,
        n: usize,
    }

    pub struct KernelSafetyError {
        declared: Safety,
        requested: Safety,
    }

    pub enum KernelError {
        NotFound(KernelNotFound),
        Threads(KernelThreadsError),
        Safety(KernelSafetyError),
    }

    pub struct KernelArgError {
        msg: String,
    }

    pub struct KernelUnsupportedError {
        msg: String,
    }

    pub enum BuildKernelError {
        Arg(KernelArgError),
        Unsupported(KernelUnsupportedError),
    }

    pub struct KernelCompileError {}
}
use error::*;

pub mod builder {
    use super::*;

    pub struct KernelBuilder<'a, S, const N: usize> {
        pub(super) kernel_info: Arc<KernelInfo>,
        pub(super) _m: PhantomData<&'a S>,
    }

    impl<'a, S, const N: usize> KernelBuilder<'a, S, N> {
        pub fn build(self) -> Result<Kernel<'a, S, N>, BuildKernelError> {
            todo!()
        }
    }

    pub struct DispatchBuilder<'a, S> {
        pub(super) kernel_info: Arc<KernelInfo>,
        pub(super) push_consts: Vec<u32>,
        pub(super) groups: [u32; 3],
        pub(super) _m: PhantomData<&'a S>,
    }

    impl DispatchBuilder<'_, Safe> {
        pub fn dispatch(self) {
            todo!()
        }
    }

    impl DispatchBuilder<'_, Unsafe> {
        pub unsafe fn dispatch(self) {
            todo!()
        }
    }
}
use builder::*;

pub mod safety {
    pub enum Safe {}
    pub enum Unsafe {}
}
use safety::*;

pub struct Kernel<'a, S, const N: usize> {
    kernel_info: Arc<KernelInfo>,
    push_consts: Vec<u32>,
    _m: PhantomData<&'a S>,
}

impl<'a, S, const N: usize> Kernel<'a, S, N> {
    pub fn global_threads(
        self,
        global_threads: [u32; N],
    ) -> Result<DispatchBuilder<'a, S>, KernelCompileError> {
        todo!()
    }
    pub fn groups(self, groups: [u32; N]) -> Result<DispatchBuilder<'a, S>, KernelCompileError> {
        todo!()
    }
}

pub struct Module {
    kernels: HashMap<String, Arc<KernelInfo>>,
}

impl<'de> Deserialize<'de> for Module {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de> {
        let raw_module = RawModule::deserialize(deserializer)?;
        let kernels = raw_module.kernels.into_iter()
            .map(|(name, info)| (name, Arc::new(info)))
        Ok(Self {
            kernels,
        })
    }
}
*/
