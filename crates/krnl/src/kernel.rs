use krnl_macros::host_only;

host_only! {
    #[cfg(feature = "device")]
    use crate::{scalar::DeviceCopy, context::device::{Kernel as RawKernel, BufferBindingVec}};
    use crate::{
        Result,
        buffer::{Slice, SliceMut},
        context::{
            Context,
            device::Features,
        },
        scalar::Element,
    };
    use std::{
        marker::PhantomData,
        collections::BTreeMap,
    };
    #[cfg(feature = "device")]
    use derive_more::Debug;
    #[cfg(feature = "device")]
    use fxhash::FxHashMap;
    #[cfg(feature = "device")]
    use tinyvec::ArrayVec;

    pub enum Safe {}

    pub unsafe trait KernelDef {
        type Safety;
        type BuildArgs;
        type Args<'a>;
        fn builder(
            args: Self::BuildArgs,
        ) -> KernelBuilder<Self>
        where
            Self: Sized,
        {
            Kernel::builder(args)
        }
        fn __visit_build_args<V: BuildArgsVisitor>(
            args: &Self::BuildArgs,
            v: &mut V,
        );
        fn __visit_args<V: ArgsVisitor>(args: &mut Self::Args<'_>, v: &mut V) -> Result<()>;
    }

    pub struct KernelBuilder<T: KernelDef> {
        args: T::BuildArgs,
        threads: Option<usize>,
        subgroup_threads: Option<usize>,
        _m: PhantomData<T>,
    }

    impl<T: KernelDef> KernelBuilder<T> {
        pub fn threads(mut self, threads: usize) -> Self {
            assert!(threads <= u32::MAX as usize);
            self.threads.replace(threads);
            self
        }
        pub fn subgroup_threads(mut self, subgroup_threads: usize) -> Self {
            self.subgroup_threads.replace(subgroup_threads);
            self
        }
        pub fn build(self, context: Context) -> Result<Kernel<T, T::Safety>> {
            match context {
                Context::Host => todo!(),
                #[cfg(not(feature = "device"))]
                Context::Device(device) => unreachable!(),
                #[cfg(feature = "device")]
                Context::Device(device) => {
                    let threads = self.threads.unwrap_or(256) as u32;
                    let subgroup_threads = if let Some(subgroup_threads) = self.subgroup_threads {
                        Some(subgroup_threads as u32)
                    } else {
                        Some(device.default_subgroup_threads() as u32)
                    };
                    let key = KernelKey::new::<T>(&self.args, threads, subgroup_threads);
                    let raw = RawKernel::get_or_create(device, key, || KernelCreateInfo::new::<T>(&self.args, threads as u32, subgroup_threads)).unwrap();
                    Ok(Kernel {
                        builder: self,
                        groups: 0,
                        raw,
                        _m: PhantomData,
                    })
                }
            }
        }
    }

    #[cfg(feature = "device")]
    #[derive(Default, Clone, PartialEq, Eq, Hash)]
    pub(crate) struct KernelKey(Box<[u8]>);

    #[cfg(feature = "device")]
    impl KernelKey {
        fn new<T: KernelDef>(args: &T::BuildArgs, threads: u32, subgroup_threads: Option<u32>) -> Self {
            let mut proto_builder = KernelKeyProtoBuilder::default();
            T::__visit_build_args(args, &mut proto_builder);
            proto_builder.__visit_spec("krnl::threads", &threads);
            if let Some(subgroup_threads) = subgroup_threads {
                proto_builder.__visit_spec("krnl::subgroup_threads", &subgroup_threads);
            }
            let mut builder = KernelKeyBuilder {
                bytes: Vec::with_capacity(proto_builder.byte_count)
            };
            T::__visit_build_args(args, &mut builder);
            builder.__visit_spec("krnl::threads", &threads);
            if let Some(subgroup_threads) = subgroup_threads {
                builder.__visit_spec("krnl::subgroup_threads", &subgroup_threads);
            }
            Self(builder.bytes.into())
        }
    }

    #[cfg(feature = "device")]
    #[derive(Default)]
    struct KernelKeyProtoBuilder {
        byte_count: usize,
    }

    #[cfg(feature = "device")]
    impl __private::__visit::sealed::Sealed for KernelKeyProtoBuilder {}

    #[cfg(feature = "device")]
    impl BuildArgsVisitor for KernelKeyProtoBuilder {
        fn __visit_spirv(&mut self, _spirv: &'static [u32]) {
            self.byte_count += size_of::<usize>();
        }
        fn __visit_spec<T: DeviceCopy>(&mut self, _name: &'static str, _spec: &T) {
            self.byte_count += size_of::<T>();
        }
    }

    #[cfg(feature = "device")]
    struct KernelKeyBuilder {
        bytes: Vec<u8>,
    }

    #[cfg(feature = "device")]
    impl __private::__visit::sealed::Sealed for KernelKeyBuilder {}

    #[cfg(feature = "device")]
    impl BuildArgsVisitor for KernelKeyBuilder {
        fn __visit_spirv(&mut self, spirv: &'static [u32]) {
            self.bytes.extend((spirv.as_ptr() as usize).to_ne_bytes());
        }
        fn __visit_spec<T: DeviceCopy>(&mut self, _name: &'static str, spec: &T) {
            self.bytes.extend(bytemuck::bytes_of(spec));
        }
    }

    #[cfg(feature = "device")]
    #[derive(Default, Debug)]
    pub(crate) struct KernelDesc {
        pub(crate) features: Features,
        pub(crate) threads: u32,
        pub(crate) subgroup_threads: Option<u32>,
        pub(crate) buffers: u32,
        pub(crate) mutability: BufferMutabilityMask,
        pub(crate) push_constant_bytes: u32,
        pub(crate) push_offsets: Vec<u32>,
        buffer_offsets: Vec<u32>,
    }

    #[cfg(feature = "device")]
    #[derive(Default, Clone, Copy)]
    pub(crate) struct BufferMutabilityMask(u32);

    #[cfg(feature = "device")]
    impl BufferMutabilityMask {
        pub(crate) fn get(&self, index: u32) -> bool {
            (self.0 & (1 << index)) != 0
        }
        fn insert(&mut self, index: u32) {
            self.0 |= 1 << index
        }
    }

    #[cfg(feature = "device")]
    impl std::fmt::Debug for BufferMutabilityMask {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            let values: Vec<u32> = (0..32).filter(|i| self.get(*i)).collect();
            values.fmt(f)
        }
    }

    #[cfg(feature = "device")]
    pub(crate) struct KernelCreateInfo {
        pub(crate) spirv: &'static [u32],
        pub(crate) spec_constants: BTreeMap<u32, ArrayVec<[u32; 2]>>,
        pub(crate) desc: KernelDesc,
    }

    #[cfg(feature = "device")]
    impl KernelCreateInfo {
        fn new<T: KernelDef>(args: &T::BuildArgs, threads: u32, subgroup_threads: Option<u32>) -> Self {
            let mut builder = KernelCreateInfoBuilder::default();
            T::__visit_build_args(args, &mut builder);
            builder.desc.threads = threads;
            builder.desc.subgroup_threads = subgroup_threads;
            builder.spec_constants.insert(0, ArrayVec::from_array_len([threads, 0], 1));
            let spirv = builder.spirv.expect("no spirv!");
            let mut desc = builder.desc;
            while desc.push_constant_bytes % 4 != 0 {
                desc.push_constant_bytes += 1;
            }
            Self {
                spirv,
                spec_constants: builder.spec_constants,
                desc,
            }
        }
    }

    #[cfg(feature = "device")]
    #[derive(Default, Debug)]
    struct KernelCreateInfoBuilder {
        spirv: Option<&'static [u32]>,
        spec_ids: FxHashMap<&'static str, u32>,
        spec_constants: BTreeMap<u32, ArrayVec<[u32; 2]>>,
        desc: KernelDesc,
    }

    #[cfg(feature = "device")]
    impl __private::__visit::sealed::Sealed for KernelCreateInfoBuilder {}

    #[cfg(feature = "device")]
    impl BuildArgsVisitor for KernelCreateInfoBuilder {
        fn __visit_spirv(&mut self, spirv: &'static [u32]) {
            self.spirv.replace(spirv);
        }
        fn __visit_features(&mut self, features: Features) {
            self.desc.features.insert(features);
        }
        fn __visit_buffer<T: Element>(&mut self, _name: &'static str) {
            self.desc.buffers += 1;
        }
        fn __visit_buffer_mut<T: Element>(&mut self, _name: &'static str) {
            self.desc.mutability.insert(self.desc.buffers);
            self.__visit_buffer::<T>(_name);
        }
        fn __visit_buffer_offset(&mut self, id: u32, offset: u32) {
            assert_eq!(self.desc.buffer_offsets.len() as u32, id);
            self.desc.buffer_offsets.push(offset);
            let range = offset + size_of::<u32>() as u32;
            self.desc.push_constant_bytes = self.desc.push_constant_bytes.max(range);
        }
        fn __visit_push<T: DeviceCopy>(&mut self, _name: &'static str, offset: u32) {
            self.desc.push_offsets.push(offset);
            let range = offset + size_of::<T>() as u32;
            self.desc.push_constant_bytes = self.desc.push_constant_bytes.max(range);
        }
        fn __visit_spec_id<T: DeviceCopy>(&mut self, name: &'static str, id: u32) {
            self.spec_ids.insert(name, id);
        }
        fn __visit_spec<T: DeviceCopy>(&mut self, name: &'static str, spec: &T) {
            let mut words = ArrayVec::default();
            if const { size_of::<T>() == 1 } {
                let x: u8 = bytemuck::cast(*spec);
                words.push(x as u32);
            } else if const { size_of::<T>() == 2 } {
                let x: u16 = bytemuck::cast(*spec);
                words.push(x as u32);
            } else if const { size_of::<T>() == 4 } {
                words.push(bytemuck::cast(*spec));
            } else {
                words.extend(bytemuck::cast_slice(bytemuck::bytes_of(spec)).iter().copied());
            }
            let id = self.spec_ids.get(name).copied().unwrap();
            self.spec_constants.insert(id, words);
        }
    }

    pub struct Kernel<T: KernelDef, S = ()> {
        builder: KernelBuilder<T>,
        groups: usize,
        #[cfg(feature = "device")]
        raw: RawKernel,
        _m: PhantomData<(T, S)>,
    }

    impl<T: KernelDef> Kernel<T> {
        pub fn builder(
            args: T::BuildArgs,
        ) -> KernelBuilder<T> {
            KernelBuilder {
                args,
                threads: None,
                subgroup_threads: None,
                _m: PhantomData,
            }
        }
        pub unsafe fn exec(&self, args: T::Args<'_>) -> Result<()> {
            unsafe { self.exec_impl(args) }
        }
    }

    impl<T: KernelDef<Safety = Safe>> Kernel<T, Safe> {
        pub fn exec(&self, args: T::Args<'_>) -> Result<()> {
            unsafe { self.exec_impl(args) }
        }
    }

    impl<T: KernelDef, S> Kernel<T, S> {
        pub fn global_threads(self, global_threads: usize) -> Self {
            #[cfg(feature = "device")] {
                let threads = self.raw.desc().threads as usize;
                let groups = global_threads / threads + (global_threads % threads != 0) as usize;
                self.groups(groups)
            }
            #[cfg(not(feature = "device"))] {
                unreachable!()
            }
        }
        pub fn groups(self, groups: usize) -> Self {
            Self {
                groups,
                .. self
            }
        }
        unsafe fn exec_impl(&self, args: T::Args<'_>) -> Result<()> {
            #[cfg(feature = "device")] {
                let mut args = args;
                let desc = self.raw.desc();
                let buffers = BufferBindingVec::with_capacity(desc.buffers as usize);
                let push_constants = vec![0u8; desc.push_constant_bytes as usize];
                let mut visitor = KernelArgsVisitor {
                    desc: self.raw.desc(),
                    buffers,
                    push_constants,
                    push_index: 0,
                    items: None,
                };
                T::__visit_args(&mut args, &mut visitor)?;
                let KernelArgsVisitor {
                    buffers,
                    mut push_constants,
                    items,
                    ..
                } = visitor;
                for (value, offset) in buffers.buffer_offsets().zip(desc.buffer_offsets.iter().copied()) {
                    push_constants[offset as usize..][..4].copy_from_slice(&value.to_ne_bytes());
                }
                let mut groups = self.groups;
                if let Some(items) = items && groups == 0 {
                    let threads = desc.threads as usize;
                    groups = items / threads + (items % threads != 0) as usize;
                }
                if groups == 0 {
                    todo!();
                }
                unsafe { self.raw.exec(groups, &buffers, &push_constants) }
            }
            #[cfg(not(feature = "device"))] {
                unreachable!()
            }
        }
    }

    #[cfg(feature = "device")]
    struct KernelArgsVisitor<'a> {
        desc: &'a KernelDesc,
        buffers: BufferBindingVec,
        push_constants: Vec<u8>,
        push_index: usize,
        items: Option<usize>,
    }

    #[cfg(feature = "device")]
    impl __private::__visit::sealed::Sealed for KernelArgsVisitor<'_> {}

    #[cfg(feature = "device")]
    impl ArgsVisitor for KernelArgsVisitor<'_> {
        fn __visit_slice<T: Element>(
            &mut self,
            _name: &'static str,
            slice: &Slice<T>,
        ) -> Result<()> {
            if let crate::context::Slice::Device(slice) = slice.as_context_slice() {
                self.buffers.push_slice(slice);
                Ok(())
            } else {
                todo!()
            }
        }
        fn __visit_slice_mut<T: Element>(
            &mut self,
            _name: &'static str,
            slice: &mut SliceMut<T>,
        ) -> Result<()> {
            if let crate::context::SliceMut::Device(slice) = slice.as_context_slice_mut() {
                self.buffers.push_slice_mut(slice);
                Ok(())
            } else {
                todo!()
            }
        }
        fn __visit_item<T: Element>(
            &mut self,
            name: &'static str,
            slice: &Slice<T>,
        ) -> Result<()> {
            self.items.replace(slice.len());
            self.__visit_slice(name, slice)
        }
        fn __visit_item_mut<T: Element>(
            &mut self,
            name: &'static str,
            slice: &mut SliceMut<T>,
        ) -> Result<()> {
            self.items.replace(slice.len());
            self.__visit_slice_mut(name, slice)
        }
        fn __visit_push<T: DeviceCopy>(&mut self, _name: &'static str, push: &T) {
            let offset = self.desc.push_offsets[self.push_index] as usize;
            let size = size_of::<T>();
            self.push_constants[offset..offset+size].copy_from_slice(bytemuck::bytes_of(push));
            self.push_index += 1;
        }
    }
}

pub mod __private {
    #[cfg(not(target_arch = "spirv"))]
    pub mod __visit {
        pub use crate::context::device::Features;
        use crate::{
            Result,
            buffer::{Slice, SliceMut},
            scalar::{DeviceCopy, Element},
        };

        pub(in crate::kernel) mod sealed {
            pub trait Sealed {}
        }
        use sealed::Sealed;

        #[allow(unused_variables)]
        pub trait __BuildArgsVisitor: Sealed {
            fn __visit_spirv(&mut self, spirv: &'static [u32]) {}
            fn __visit_features(&mut self, features: Features) {}
            fn __visit_buffer<T: Element>(&mut self, name: &'static str) {}
            fn __visit_buffer_mut<T: Element>(&mut self, name: &'static str) {}
            fn __visit_buffer_offset(&mut self, id: u32, offset: u32) {}
            fn __visit_push<T: DeviceCopy>(&mut self, name: &'static str, offset: u32) {}
            fn __visit_spec_id<T: DeviceCopy>(&mut self, name: &'static str, id: u32) {}
            fn __visit_spec<T: DeviceCopy>(&mut self, name: &'static str, spec: &T) {}
        }

        pub trait __ArgsVisitor: Sealed {
            fn __visit_item<T: Element>(
                &mut self,
                name: &'static str,
                slice: &Slice<T>,
            ) -> Result<()>;
            fn __visit_item_mut<T: Element>(
                &mut self,
                name: &'static str,
                slice: &mut SliceMut<T>,
            ) -> Result<()>;
            fn __visit_slice<T: Element>(
                &mut self,
                name: &'static str,
                slice: &Slice<T>,
            ) -> Result<()>;
            fn __visit_slice_mut<T: Element>(
                &mut self,
                name: &'static str,
                slice: &mut SliceMut<T>,
            ) -> Result<()>;
            fn __visit_push<T: DeviceCopy>(&mut self, name: &'static str, push: &T);
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    pub use __visit::*;

    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub mod __intrinsics {
        use crate::scalar::{DeviceCopy, Element};
        use core::cell::UnsafeCell;
        use krnl_core::__private::{__group_slice as group_slice, __input, __item as item, __spec};
        pub use krnl_core::__private::{__safe, __threads};

        unsafe fn __data_type<T: DeviceCopy, V>(var: *const V) {
            unsafe {
                T::__data_type(var);
            }
        }

        pub unsafe fn __item<T: Element>(var: &[T]) {
            let var = var.as_ptr();
            unsafe {
                item(var);
                __input::<T>(var);
                __data_type::<T, T>(var);
            }
        }

        #[cfg(all(krnlc, target_arch = "spirv"))]
        pub unsafe fn __item_mut<T: Element>(var: &[UnsafeCell<T>]) {
            let var = var.as_ptr();
            unsafe {
                item(var);
                __input::<UnsafeCell<T>>(var);
                __data_type::<T, UnsafeCell<T>>(var);
            }
        }

        #[cfg(all(krnlc, target_arch = "spirv"))]
        pub unsafe fn __slice<T: Element>(var: &[T]) {
            let var = var.as_ptr();
            unsafe {
                __input::<T>(var);
                __data_type::<T, T>(var);
            }
        }

        #[cfg(all(krnlc, target_arch = "spirv"))]
        pub unsafe fn __unsafe_slice<T: Element>(var: &[UnsafeCell<T>]) {
            let var = var.as_ptr();
            unsafe {
                __input::<UnsafeCell<T>>(var);
                __data_type::<T, UnsafeCell<T>>(var);
            }
        }

        #[cfg(all(krnlc, target_arch = "spirv"))]
        pub unsafe fn __spec_constant<T: DeviceCopy>(var: &T) {
            let var = var as *const T;
            unsafe {
                __spec::<T>(var);
                __input::<T>(var);
                __data_type::<T, T>(var);
            }
        }

        #[cfg(all(krnlc, target_arch = "spirv"))]
        pub unsafe fn __push_constant<T: DeviceCopy>(var: &T) {
            let var = var as *const T;
            unsafe {
                __input::<T>(var);
                __data_type::<T, T>(var);
            }
        }

        #[cfg(all(krnlc, target_arch = "spirv"))]
        pub unsafe fn __group_array<T: Element, const N: usize>(var: &[UnsafeCell<T>; N]) {
            let var = var.as_ptr();
            unsafe {
                __input::<UnsafeCell<T>>(var);
                __data_type::<T, UnsafeCell<T>>(var);
            }
        }

        #[cfg(all(krnlc, target_arch = "spirv"))]
        pub unsafe fn __group_slice<T: Element>(var: &[UnsafeCell<T>], len: usize) {
            let var = var.as_ptr();
            unsafe {
                group_slice(var, len);
                __input::<UnsafeCell<T>>(var);
                __data_type::<T, UnsafeCell<T>>(var);
            }
        }
    }
    #[cfg(all(krnlc, target_arch = "spirv"))]
    pub use __intrinsics::*;

    #[cfg(not(target_arch = "spirv"))]
    pub mod __marker {
        use core::marker::PhantomData;

        pub struct __Safety<T> {
            _m: PhantomData<T>,
        }

        impl<T> __Safety<T> {
            pub const fn __new() -> Self {
                Self { _m: PhantomData }
            }
        }

        pub struct __Spec<T> {
            _m: PhantomData<T>,
        }

        impl<T> __Spec<T> {
            pub const fn __new() -> Self {
                Self { _m: PhantomData }
            }
        }

        pub struct __Buffer<T: ?Sized> {
            _m: PhantomData<T>,
        }

        impl<T: ?Sized> __Buffer<T> {
            pub const fn __new() -> Self {
                Self { _m: PhantomData }
            }
        }

        pub struct __Item<T> {
            _m: PhantomData<T>,
        }

        impl<T> __Item<T> {
            pub const fn __new() -> Self {
                Self { _m: PhantomData }
            }
        }

        pub struct __ItemMut<T> {
            _m: PhantomData<T>,
        }

        impl<T> __ItemMut<T> {
            pub const fn __new() -> Self {
                Self { _m: PhantomData }
            }
        }

        pub struct __Push<T> {
            _m: PhantomData<T>,
        }

        impl<T> __Push<T> {
            pub const fn __new() -> Self {
                Self { _m: PhantomData }
            }
        }

        pub struct __Group<T: ?Sized> {
            _m: PhantomData<T>,
        }

        impl<T: ?Sized> __Group<T> {
            pub const fn __new() -> Self {
                Self { _m: PhantomData }
            }
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    pub use __marker::*;
}
#[cfg(not(target_arch = "spirv"))]
use __private::__visit::{__ArgsVisitor as ArgsVisitor, __BuildArgsVisitor as BuildArgsVisitor};
