use crate::{scalar::ScalarType, version::Version};
use serde::{Deserialize, Serialize};
use spirv::Capability;
use std::{
    collections::HashMap,
    fmt::{self, Debug},
    sync::Arc,
};
pub mod error {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("Kernel {:?} not found in module {:?}!", .name, .module.name)]
    pub struct KernelNotFound {
        pub(super) name: String,
        pub(super) module: Arc<ModuleInner>,
    }
}
use error::*;

/*
#[doc(hidden)]
#[derive(Serialize, Deserialize, Debug)]
pub struct ModuleWithHash {
    pub module: Module,
    pub hash: u64,
}
*/

#[derive(Serialize, Deserialize, Debug)]
#[serde(transparent)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

impl Module {
    pub fn kernel_info(&self, kernel: impl AsRef<str>) -> Result<KernelInfo, KernelNotFound> {
        let kernel = kernel.as_ref();
        if let Some(info) = self.inner.kernels.get(&*kernel) {
            Ok(KernelInfo {
                module: self.inner.clone(),
                inner: info.clone(),
            })
        } else {
            Err(KernelNotFound {
                name: kernel.to_string(),
                module: self.inner.clone(),
            })
        }
    }
}

#[derive(Clone, Debug)]
pub struct KernelInfo {
    module: Arc<ModuleInner>,
    inner: Arc<KernelInfoInner>,
}

#[doc(hidden)]
#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct ModuleInner {
    pub name: String,
    pub kernels: HashMap<String, Arc<KernelInfoInner>>,
}

impl ModuleInner {
    pub fn get_from_module(module: &Module) -> &Arc<Self> {
        &module.inner
    }
    pub fn get_from_kernel_info(kernel_info: &KernelInfo) -> &Arc<Self> {
        &kernel_info.module
    }
    pub fn into_module(self) -> Module {
        Module {
            inner: Arc::new(self),
        }
    }
}

#[doc(hidden)]
#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct KernelInfoInner {
    pub name: String,
    pub vulkan_version: Version,
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<String>,
    pub safe: bool,
    pub threads: Vec<u32>,
    pub elementwise: bool,
    pub slice_infos: Vec<SliceInfo>,
    pub push_infos: Vec<PushInfo>,
    pub num_push_words: u32,
    pub spec_infos: Vec<SpecInfo>,
    pub spirv: Option<Spirv>,
}

impl KernelInfoInner {
    pub fn get_from_kernel_info(kernel_info: &KernelInfo) -> &Arc<Self> {
        &kernel_info.inner
    }
    pub fn compile_options(&self) -> CompileOptions {
        CompileOptions {
            vulkan_version: self.vulkan_version,
            capabilities: self.capabilities.clone(),
            extensions: self.extensions.clone(),
        }
    }
}

#[doc(hidden)]
#[derive(Eq, PartialEq, Serialize, Deserialize)]
pub struct Spirv {
    pub words: Vec<u32>,
}

impl Debug for Spirv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Spirv({}B)", self.words.len() * 4)
    }
}

#[doc(hidden)]
#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct SliceInfo {
    pub name: String,
    pub scalar_type: ScalarType,
    pub mutable: bool,
    pub elementwise: bool,
}

#[doc(hidden)]
#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct PushInfo {
    pub name: String,
    pub scalar_type: ScalarType,
    pub offset: u32,
}

#[doc(hidden)]
#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct SpecInfo {
    pub name: String,
    pub scalar_type: ScalarType,
}

#[doc(hidden)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CompileOptions {
    pub vulkan_version: Version,
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<String>,
}
