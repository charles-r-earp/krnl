use crate::__private::raw_module::{RawKernelInfo, RawModule};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

pub mod error {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("Kernel {:?} not found in module {:?}!", .name, .module.name)]
    pub struct KernelNotFound {
        pub(super) name: String,
        pub(super) module: Arc<RawModule>,
    }
}
use error::*;

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, Serialize, Deserialize, Debug)]
pub struct VulkanVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl VulkanVersion {
    pub fn from_major_minor(major: u32, minor: u32) -> Self {
        Self {
            major,
            minor,
            patch: 0,
        }
    }
}

impl Default for VulkanVersion {
    fn default() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }
}

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
    pub fn kernel_info(&self, kernel: impl AsRef<str>) -> Result<KernelInfo, KernelNotFound> {
        let kernel = kernel.as_ref();
        if let Some(info) = self.raw.kernels.get(&*kernel) {
            Ok(KernelInfo {
                module: self.raw.clone(),
                info: info.clone(),
            })
        } else {
            Err(KernelNotFound {
                name: kernel.to_string(),
                module: self.raw.clone(),
            })
        }
    }
}

#[derive(Clone, Debug)]
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
