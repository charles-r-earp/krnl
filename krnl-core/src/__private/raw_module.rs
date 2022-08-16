use crate::scalar::ScalarType;
use spirv::Capability;
use std::{collections::HashMap, fmt::{self, Debug}};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct RawModule {
    pub source: String,
    pub name: String,
    pub target: Target,
    pub kernels: HashMap<String, KernelInfo>,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum Target {
    Vulkan(u32, u32),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct KernelInfo {
    pub name: String,
    pub target: Target,
    pub capabilities: Vec<Capability>,
    pub safety: Safety,
    pub slice_infos: Vec<SliceInfo>,
    pub push_infos: Vec<PushInfo>,
    pub threads: Vec<u32>,
    pub spirv: Option<Spirv>,
}

#[derive(Serialize, Deserialize)]
pub struct Spirv {
    pub words: Vec<u32>,
}

impl Debug for Spirv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Spirv")
            .field("words", &format!("{}B", &self.words.len()))
            .finish()
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub enum Safety {
    Safe,
    Unsafe,
}

#[derive(Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub enum Mutability {
    Immutable,
    Mutable,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SliceInfo {
    pub name: String,
    pub scalar_type: ScalarType,
    pub mutability: Mutability,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PushInfo {
    pub name: String,
    pub scalar_type: ScalarType,
    pub offset: u32,
}
