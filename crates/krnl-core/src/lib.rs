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

    /*
    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __spec_name<T, const ID: u32, const NAME: u32>(var: T) {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%u32 = OpTypeInt 32 0",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%id = OpConstant %u32 {id}",
                "%name = OpConstant %u32 {name}",
                "%result = OpExtInst %void %ext {inst} {var} %id %name",
                inst = const KrnlInst::SpecName as u32,
                id = const ID,
                name = const NAME,
                var = in(reg) &var,
            }
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __push_name<V, const NAME: u32>(var: *const V) {
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%u32 = OpTypeInt 32 0",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%name = OpConstant %u32 {name}",
                "%result = OpExtInst %void %ext {inst} {var} %name",
                inst = const KrnlInst::PushName as u32,
                var = in(reg) var,
                name = const NAME,
            }
        }
    }
    */

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

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __threads() -> u32 {
        let mut result_slot = MaybeUninit::uninit();
        unsafe {
            asm! {
                "%void = OpTypeVoid",
                "%u32 = OpTypeInt 32 0",
                //"%uvec3 = OpTypeVector %u32 3",
                "%one = OpConstant %u32 1",
                "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                "%threads = OpSpecConstant %u32 1",
                "OpDecorate %threads SpecId 0",
                "OpName %threads \"krnl::threads\"",
                //"%workgroup_size = OpSpecConstantComposite %uvec3 %threads %one %one",
                //"OpDecorate %workgroup_size BuiltIn WorkgroupSize",
                //"%result = OpExtInst %void %ext {inst} %workgroup_size",
                "OpStore {result_slot} %threads",
                //inst = const KrnlInst::WorkgroupSize as u32,
                result_slot = in(reg) result_slot.as_mut_ptr()
            }
            result_slot.assume_init()
        }
    }

    /*
    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __spec_constant_u32<T, const ID: u32>() -> T {
        let mut result_slot = MaybeUninit::uninit();
        unsafe {
            asm! {
                "%spec = OpSpecConstant typeof*{result_slot} 0",
                "OpDecorate %spec SpecId {id}",
                "OpStore {result_slot} %spec",
                id = const ID,
                result_slot = in(reg) result_slot.as_mut_ptr()
            }
            result_slot.assume_init()
        }
    }

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub unsafe fn __spec_constant_u64<T, const ID: u32>() -> T {
        let mut result_slot = MaybeUninit::uninit();
        unsafe {
            asm! {
                "%spec = OpSpecConstant typeof*{result_slot} 0 0",
                "OpDecorate %spec SpecId {id}",
                "OpStore {result_slot} %spec",
                id = const ID,
                result_slot = in(reg) result_slot.as_mut_ptr()
            }
            result_slot.assume_init()
        }
    }
    */
}

/*
pub mod scalar {
    use derive_more::Display;

    #[derive(Clone, Copy, PartialEq, Eq, Display, Debug)]
    #[cfg_attr(not(target_arch = "spirv"), repr(u8))]
    #[cfg_attr(target_arch = "spirv", repr(u32))]
    pub enum ScalarType {
        #[display("u8")]
        U8 = 1,
        #[display("i8")]
        I8 = 2,
        #[display("u16")]
        U16 = 3,
        #[display("i16")]
        I16 = 4,
        #[display("f16")]
        F16 = 5,
        #[display("bf16")]
        BF16 = 6,
        #[display("u32")]
        U32 = 7,
        #[display("i32")]
        I32 = 8,
        #[display("u64")]
        U64 = 10,
        #[display("i64")]
        I64 = 11,
        #[display("f32")]
        F32 = 9,
        #[display("f64")]
        F64 = 12,
    }
}

#[doc(hidden)]
pub mod builtin {
    #[cfg(target_arch = "spirv")]
    pub mod __private {
        use core::arch::asm;
        use core::mem::MaybeUninit;
        use spirv_std::glam::UVec3;

        /*
        pub unsafe fn __global_threads() -> usize {
            __groups() * __threads()
        }

        pub unsafe fn __global_thread_id() -> usize {
            __group_id() * __threads() + __thread_id()
        }

        pub unsafe fn __groups() -> usize {
            let mut result_slot = MaybeUninit::<UVec3>::uninit();
            unsafe {
                asm! {
                    "%var = OpVariable typeof{result_slot} Input",
                    "OpDecorate %var BuiltIn NumWorkgroups",
                    "OpName %var \"krnl::groups\"",
                    "%result = OpLoad _ %var",
                    "OpStore {result_slot} %result",
                    result_slot = in(reg) result_slot.as_mut_ptr(),
                }
                result_slot.assume_init().x as usize
            }
        }

        pub unsafe fn __group_id() -> usize {
            let mut result_slot = MaybeUninit::<UVec3>::uninit();
            unsafe {
                asm! {
                    "%var = OpVariable typeof{result_slot} Input",
                    "OpDecorate %var BuiltIn WorkgroupId",
                    "OpName %var \"krnl::group_id\"",
                    "%result = OpLoad _ %var",
                    "OpStore {result_slot} %result",
                    result_slot = in(reg) result_slot.as_mut_ptr(),
                }
                result_slot.assume_init().x as usize
            }
        }
        */

        pub fn __threads() -> usize {
            let mut result_slot = MaybeUninit::uninit();
            unsafe {
                asm! {
                    "%u32 = OpTypeInt 32 0",
                    "%spec = OpSpecConstant %u32 1",
                    "OpName %spec \"krnl::threads\"",
                    "OpStore {result_slot} %spec",
                    result_slot = in(reg) result_slot.as_mut_ptr()
                }
                result_slot.assume_init()
            }
        }

        /*
        pub unsafe fn __thread_id() -> usize {
            let mut result_slot = MaybeUninit::uninit();
            unsafe {
                asm! {
                    "%var = OpVariable typeof{result_slot} Input",
                    "OpDecorate %var BuiltIn LocalInvocationIndex",
                    "OpName %var \"krnl::thread_id\"",
                    "%result = OpLoad _ %var",
                    "OpStore {result_slot} %result",
                    result_slot = in(reg) result_slot.as_mut_ptr(),
                }
                result_slot.assume_init()
            }
        }

        pub unsafe fn __items() -> usize {
            use crate::ext_inst::KrnlInst;
            let mut result_slot = MaybeUninit::uninit();
            unsafe {
                asm! {
                    "%void = OpTypeVoid",
                    "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                    "%var = OpVariable typeof{result_slot} Input",
                    "OpName %var \"krnl::items\"",
                    "%_result = OpExtInst %void %ext {inst} %var",
                    "%result = OpLoad _ %var",
                    "OpStore {result_slot} %result",
                    result_slot = in(reg) result_slot.as_mut_ptr(),
                    inst = const KrnlInst::Items as u32,
                }
                result_slot.assume_init()
            }
        }
        */
    }
}

pub mod ext_inst {

    #[cfg(target_arch = "spirv")]
    pub mod __private {
        use super::KrnlInst;
        use core::arch::asm;

        pub unsafe fn __kernel_data<const DATA: u32>() {
            unsafe {
                asm! {
                    "%void = OpTypeVoid",
                    "%u32 = OpTypeInt 32 0",
                    "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                    "%data = OpConstant %u32 {data}",
                    "%result = OpExtInst %void %ext {inst} %data",
                    inst = const KrnlInst::KernelData as u32,
                    data = const DATA,
                }
            }
        }

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

        pub unsafe fn __group_slice<V>(var: *const V, len: usize) {
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
}

#[doc(hidden)]
#[cfg(target_arch = "spirv")]
pub mod __private {
    pub use crate::builtin::__private::__threads;
    pub use crate::ext_inst::__private::*;
}

pub mod kernel {
    pub mod __private {
        pub const __INCLUDE_ENV_VAR_PREFIX: &'static str = "KRNL_INCLUDE_";
    }
}
*/

/*
pub use krnl_macros as macros;
#[cfg(target_arch = "spirv")]
pub use spirv_std;

pub mod scalar {
    pub use half::{bf16, f16};
    pub use krnl_types::scalar::*;
}

pub mod kernel {
    #[cfg(any(not(target_arch = "spirv"), all(krnlc, target_arch = "spirv")))]
    pub mod __private {
        use super::{ItemKernel, Kernel};
        use crate::{
            macros::device_only,
            scalar::{DeviceCopy, Element, Scalar},
        };
        use core::cell::UnsafeCell;

        #[cfg(target_arch = "spirv")]
        use core::{arch::asm, mem::MaybeUninit};
        #[cfg(target_arch = "spirv")]
        use krnl_types::ext_inst::__private::__data_type;

        pub unsafe fn __kernel() -> Kernel {
            #[cfg(target_arch = "spirv")]
            {
                use crate::thread::*;

                Kernel {
                    groups: groups(),
                    group_id: group_id(),
                    threads: threads(),
                    thread_id: thread_id(),
                    /*
                    subgroups: subgroups(),
                    subgroup_threads: subgroup_threads(),
                    subgroup_thread_id: subgroup_thread_id(),
                    */
                }
            }
            #[cfg(not(target_arch = "spirv"))]
            unreachable!()
        }

        pub unsafe fn __item_kernel(items: usize) -> ItemKernel {
            #[cfg(target_arch = "spirv")]
            {
                ItemKernel {
                    items,
                    item_id: crate::thread::global_thread_id(),
                }
            }
            #[cfg(not(target_arch = "spirv"))]
            unreachable!()
        }

        pub unsafe fn __item_kernel_next(kernel: &mut ItemKernel) {
            #[cfg(target_arch = "spirv")]
            {
                kernel.item_id += crate::thread::global_threads();
            }
            #[cfg(not(target_arch = "spirv"))]
            unreachable!()
        }

        /*
        pub fn __item_kernel_for_each(items: usize, mut f: impl FnMut(&ItemKernel)) {
            #[cfg(target_arch = "spirv")]
            {
                use crate::thread::{global_thread_id, global_threads};

                let mut item_id = global_thread_id();
                let global_threads = global_threads();

                while item_id < items {
                    f(&ItemKernel { items, item_id });
                    item_id += global_threads;
                }
            }
            #[cfg(not(target_arch = "spirv"))]
            {
                unreachable!()
            }
        }

        pub struct ItemKernelIter(ItemKernel);

        pub unsafe fn __item_kernel_iter(items: usize) -> ItemKernelIter {
            #[cfg(target_arch = "spirv")]
            {
                use crate::thread::{global_thread_id, global_threads};

                impl ItemKernelIter {
                    pub fn next(&mut self) -> Option<ItemKernel> {
                        if self.0.item_id >= self.0.items {
                            return None;
                        }
                        let item = ItemKernel {
                            items: self.0.items,
                            item_id: self.0.item_id,
                        };
                        self.0.item_id += global_threads();
                        Some(item)
                    }
                }

                ItemKernelIter(ItemKernel {
                    items,
                    item_id: global_thread_id(),
                })
            }
            #[cfg(not(target_arch = "spirv"))]
            {
                unreachable!()
            }
        }
        */

        pub unsafe fn __kernel_data<const DATA: u32>() {
            unsafe {
                asm! {
                    "%void = OpTypeVoid",
                    "%u32 = OpTypeInt 32 0",
                    "%ext = OpExtInstImport \"NonSemantic.rust.krnl\"",
                    "%data = OpConstant %u32 {data}",
                    "%result = OpExtInst %void %ext {inst} %data",
                    inst = const KrnlInst::KernelData as u32,
                    data = const DATA,
                }
            }
        }

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

        pub unsafe fn __data_type<T: DeviceCopy, V>(var: *const V) {
            T::__data_type(var);
        }

        pub(crate) unsafe fn __data_type_f16<V>(var: *const V) {
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

        pub(crate) unsafe fn __data_type_bf16<V>(var: *const V) {
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

        pub unsafe fn __group_slice<V>(var: *const V, len: usize) {
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

    #[derive(Default)]
    pub struct Kernel {
        groups: usize,
        group_id: usize,
        threads: usize,
        thread_id: usize,
        /*
        subgroups: usize,
        subgroup_threads: usize,
        subgroup_thread_id: usize,
        */
    }

    impl Kernel {
        pub fn global_threads(&self) -> usize {
            self.groups * self.threads
        }
        pub fn global_thread_id(&self) -> usize {
            self.group_id * self.threads + self.thread_id
        }
    }

    pub struct ItemKernel {
        items: usize,
        item_id: usize,
    }

    impl ItemKernel {
        pub fn items(&self) -> usize {
            self.items
        }
        pub fn item_id(&self) -> usize {
            self.item_id
        }
    }
}
*/
