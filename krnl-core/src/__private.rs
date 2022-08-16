#[cfg(not(target_arch = "spirv"))]
pub static KRNL_MODULE_PATH: &'static str = "KRNL_MODULE_PATH";

#[cfg(not(target_arch = "spirv"))]
pub mod raw_module;
