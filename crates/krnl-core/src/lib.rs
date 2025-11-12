#![cfg_attr(target_arch = "spirv", no_std, feature(asm_experimental_arch))]

#[doc(hidden)]
pub mod __private {
    #[cfg(target_arch = "spirv")]
    use core::{arch::asm, mem::MaybeUninit};

    #[derive(Clone, Copy, Debug)]
    #[repr(u32)]
    pub enum __KrnlInst {
        Safe,
        WorkgroupSize,
        Spec,
        Input,
        DataType,
        Item,
        GroupSlice,
        Panic,
    }
    use __KrnlInst as KrnlInst;

    #[cfg(not(target_arch = "spirv"))]
    impl KrnlInst {
        pub const SET_NAME: &'static str = "NonSemantic.rust.krnl";
        pub const SET_SHORT_NAME: &'static str = "krnl";
        pub fn iter() -> impl Iterator<Item = Self> {
            use KrnlInst::*;
            [
                Safe,
                WorkgroupSize,
                Spec,
                Input,
                DataType,
                Item,
                GroupSlice,
                Panic,
            ]
            .into_iter()
        }
        pub fn from_u32(x: u32) -> Option<Self> {
            Self::iter().find(|b| *b as u32 == x)
        }
        pub fn operand_names(&self) -> &'static [&'static str] {
            match self {
                Self::Panic => &[],
                Self::Safe => &[],
                Self::WorkgroupSize => &["workgroup_size"],
                Self::Spec => &["var"],
                Self::Input => &["var"],
                Self::DataType => &["var", "type_name"],
                Self::Item => &[],
                Self::GroupSlice => &["var", "len"],
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __safe() {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%result = OpExtInst %void %ext {inst}",
                inst = const KrnlInst::Safe as u32,
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __spec<V>(var: *const V) {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%u32 = OpTypeInt 32 0",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%result = OpExtInst %void %ext {inst} {var}",
                inst = const KrnlInst::Spec as u32,
                var = in(reg) var,
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __input<V>(var: *const V) {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%u32 = OpTypeInt 32 0",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%result = OpExtInst %void %ext {inst} {var}",
                inst = const KrnlInst::Input as u32,
                var = in(reg) var,
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __data_type_f16<V>(var: *const V) {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%type_name = OpString \"f16\"",
                "%result = OpExtInst %void %ext {inst} {var} %type_name",
                inst = const KrnlInst::DataType as u32,
                var = in(reg) var,
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __data_type_bf16<V>(var: *const V) {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%type_name = OpString \"bf16\"",
                "%result = OpExtInst %void %ext {inst} {var} %type_name",
                inst = const KrnlInst::DataType as u32,
                var = in(reg) var,
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __item<V>(var: *const V) {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%result = OpExtInst %void %ext {inst} {var}",
                inst = const KrnlInst::Item as u32,
                var = in(reg) var,
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __group_slice<V>(var: *const V, len: usize) {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%u32 = OpTypeInt 32 0",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%result = OpExtInst %void %ext {inst} {var} {len}",
                inst = const KrnlInst::GroupSlice as u32,
                var = in(reg) var,
                len = in(reg) len,
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __threads() -> u32 {
        let mut result_slot = MaybeUninit::uninit();
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%u32 = OpTypeInt 32 0",
                "%one = OpConstant %u32 1",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%threads = OpSpecConstant %u32 1",
                "OpDecorate %threads SpecId 0",
                "OpName %threads \"krnl::threads\"",
                "OpStore {result_slot} %threads",
                result_slot = in(reg) result_slot.as_mut_ptr()
            }
            result_slot.assume_init()
        }
    }
}
