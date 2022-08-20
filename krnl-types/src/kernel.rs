use crate::__private::raw_module::{RawKernelInfo, RawModule};
use std::sync::Arc;

pub mod error {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("Kernel {:?} not found in module {:?}!", .name, .module.name)]
    pub struct KernelNotFound {
        pub(super) name: String,
        pub(super) module: Arc<RawModule>,
    }
}
use error::KernelNotFound;

#[derive(Debug)]
pub struct Module {
    raw: Arc<RawModule>,
}

impl Module {
    #[doc(hidden)]
    pub fn __from_raw(raw: Arc<RawModule>) -> Self {
        Self { raw }
    }
    #[doc(hidden)]
    pub fn __raw(&self) -> &Arc<RawModule> {
        &self.raw
    }
    pub fn kernel(&self, kernel: impl Into<String>) -> Result<KernelInfo, KernelNotFound> {
        let name = kernel.into();
        if let Some(info) = self.raw.kernels.get(&name) {
            Ok(KernelInfo {
                module: self.raw.clone(),
                info: info.clone(),
            })
        } else {
            Err(KernelNotFound {
                name,
                module: self.raw.clone(),
            })
        }
    }
}

#[derive(Debug)]
pub struct KernelInfo {
    module: Arc<RawModule>,
    info: Arc<RawKernelInfo>,
}

impl KernelInfo {
    #[doc(hidden)]
    pub fn __from_raw_parts(module: Arc<RawModule>, info: Arc<RawKernelInfo>) -> Self {
        Self { module, info }
    }
    #[doc(hidden)]
    pub fn __module(&self) -> &Arc<RawModule> {
        &self.module
    }
    #[doc(hidden)]
    pub fn __info(&self) -> &Arc<RawKernelInfo> {
        &self.info
    }
}
