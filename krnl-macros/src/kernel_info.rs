#![allow(warnings)]
use serde::{Deserialize, Serialize};
use spirv::Capability;
use std::{
    fmt::{self, Debug, Write},
    hash::Hash,
};

#[doc(hidden)]
#[derive(Eq, PartialEq, Hash, Serialize, Deserialize, Debug)]
pub struct CompileOptions {
    pub vulkan: (u32, u32),
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<String>,
}

/*
impl CompileOptions {
    pub fn to_cfg_string(&self) -> String {
        let mut cfg = format!("krnl_device_crate_vulkan{}_{}", self.vulkan.0, self.vulkan.1);
        let mut target_features = self.capabilities.iter().map(|x| format!("{x:?}")).chain(self.extensions.iter().map(|x| format!("ext:{x}"))).collect::<Vec<_>>();
        target_features.sort();
        for target_feature in target_features {
            write!(&mut cfg, "_{target_feature}");
        }
        cfg
    }
}
*/

#[doc(hidden)]
#[derive(Eq, PartialEq, Hash, Serialize, Deserialize, Debug)]
pub struct KernelInfo {
    pub path: String,
    pub name: String,
    pub vulkan: (u32, u32),
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<String>,
    pub safe: bool,
    pub threads: [u32; 3],
    pub dimensionality: u32,
    pub elementwise: bool,
    pub args: Vec<Arg>,
}

impl KernelInfo {
    pub fn compile_options(&self) -> CompileOptions {
        let mut capabilities = self.capabilities.clone();
        capabilities.sort();
        let mut extensions = self.extensions.clone();
        extensions.sort();
        CompileOptions {
            vulkan: self.vulkan,
            capabilities,
            extensions,
        }
    }
    pub fn slice_infos(&self) -> impl Iterator<Item = &SliceInfo> {
        self.args.iter().filter_map(|x| match x {
            Arg::Slice(x) => Some(x),
            _ => None,
        })
    }
    pub fn push_infos(&self) -> impl Iterator<Item = &PushInfo> {
        self.args.iter().filter_map(|x| match x {
            Arg::Push(x) => Some(x),
            _ => None,
        })
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

#[derive(Eq, PartialEq, Hash, Serialize, Deserialize, Debug)]
pub enum Arg {
    Slice(SliceInfo),
    Push(PushInfo),
}

#[doc(hidden)]
#[derive(Eq, PartialEq, Hash, Serialize, Deserialize, Debug)]
pub struct SliceInfo {
    pub name: String,
    pub scalar_type: ScalarType,
    pub mutable: bool,
    pub elementwise: bool,
}

#[doc(hidden)]
#[derive(Eq, PartialEq, Hash, Serialize, Deserialize, Debug)]
pub struct PushInfo {
    pub name: String,
    pub scalar_type: ScalarType,
    pub offset: u32,
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize, Debug)]
pub enum ScalarType {
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl ScalarType {
    pub fn from_str(input: &str) -> Option<Self> {
        use ScalarType::*;
        match input {
            "u8" => Some(U8),
            "i8" => Some(I8),
            "u16" => Some(U16),
            "i16" => Some(I16),
            "f16" => Some(F16),
            "bf16" => Some(BF16),
            "u32" => Some(U32),
            "i32" => Some(I32),
            "f32" => Some(F32),
            "u64" => Some(U64),
            "i64" => Some(I64),
            "f64" => Some(F64),
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        use ScalarType::*;
        match self {
            U8 => "u8",
            I8 => "i8",
            U16 => "u16",
            I16 => "i16",
            F16 => "f16",
            BF16 => "bf16",
            U32 => "u32",
            I32 => "i32",
            F32 => "f32",
            U64 => "u64",
            I64 => "i64",
            F64 => "f64",
        }
    }
    pub fn size(&self) -> usize {
        use ScalarType::*;
        match self {
            U8 | I8 => 1,
            U16 | I16 | F16 | BF16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }
}
