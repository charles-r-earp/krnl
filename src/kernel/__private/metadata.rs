use serde::{Deserialize, Serialize};

/*#[derive(Serialize, Deserialize)]
pub(crate) struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}*/

#[derive(Serialize, Deserialize)]
pub(crate) struct ModuleData {
    pub(crate) source: String,
}

#[cfg(not(proc_macro))]
pub struct ModuleDesc {
    pub module_path: &'static str,
    pub module_data_hash: u64,
    pub compressed_kernel_descs: &'static [u8],
}

/*
pub struct KernelDesc2 {
    name: Cow<'static, str>,
    source: &'static str,
    spirv: &'static [u32],
    features: Features,
    threads: UVec3,
    spec_descs: &'static [SpecDesc],
    buffer_descs: &'static [BufferDesc2],
}

impl KernelDesc2 {
    pub fn specialize(mut self, specs: &[ScalarElem]) -> Self {
        todo!()
    }
}

struct SpecDesc {
    name: &'static str,
    thread: Option<u8>,
}

struct BufferDesc2 {
    name: &'static str,
    mutable: bool,
    item: bool,
}
*/
