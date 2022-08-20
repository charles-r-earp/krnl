use crate::scalar::ScalarType;
use serde::{Deserialize, Serialize};
use spirv::Capability;
use std::{
    collections::HashMap,
    fmt::{self, Debug},
    sync::Arc,
};

#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct RawModule {
    pub name: String,
    pub target: String,
    pub kernels: HashMap<String, Arc<RawKernelInfo>>,
}

#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct RawKernelInfo {
    pub name: String,
    pub target: String,
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<String>,
    pub safety: Safety,
    pub slice_infos: Vec<SliceInfo>,
    pub push_infos: Vec<PushInfo>,
    pub threads: Vec<u32>,
    pub spirv: Option<Spirv>,
}

#[derive(Eq, PartialEq, Serialize, Deserialize)]
pub struct Spirv {
    pub words: Vec<u32>,
}

impl Debug for Spirv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Spirv({}B)", self.words.len() * 4)
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

#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct SliceInfo {
    pub name: String,
    pub scalar_type: ScalarType,
    pub mutability: Mutability,
}

#[derive(Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct PushInfo {
    pub name: String,
    pub scalar_type: ScalarType,
    pub offset: u32,
}
