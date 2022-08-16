use core::marker::PhantomData;
use krnl_core::module::{KernelInfo, Module, PushInfo, Safety, SliceInfo, Spirv};
use std::sync::Arc;

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
