#[cfg(feature = "device")]
use crate::device::KernelCache;
use crate::{device::Device, buffer::ScalarSlice};
use anyhow::Result;
use std::sync::Arc;

#[doc(inline)]
pub use krnl_macros::module;

#[doc(hidden)]
#[path = "../krnl-macros/src/kernel_info.rs"]
pub mod kernel_info;
use kernel_info::{KernelInfo, Spirv};

#[doc(hidden)]
pub struct KernelBase {
    device: Device,
    #[cfg(feature = "device")]
    cache: Arc<KernelCache>,
}

#[doc(hidden)]
pub fn __build(device: Device, kernel_info: Arc<KernelInfo>, spirv: Arc<Spirv>) -> Result<KernelBase> {
    todo!()
}


#[doc(hidden)]
pub unsafe fn __dispatch(kernel_base: &KernelBase, dispatch_dim: DispatchDim<&[u32]>, slices: &[(&'static str, ScalarSlice)], push_consts: &[u8]) -> Result<()> {
    todo!()
}

pub enum DispatchDim<T> {
    GlobalThreads(T),
    Groups(T),
}

#[doc(hidden)]
impl<T: AsRef<[u32]>> DispatchDim<T> {
    pub fn as_ref(&self) -> DispatchDim<&[u32]> {
        match self {
            Self::GlobalThreads(x) => DispatchDim::GlobalThreads(x.as_ref()),
            Self::Groups(x) => DispatchDim::GlobalThreads(x.as_ref()),
        }
    }
}

impl DispatchDim<&[u32]> {
    fn to_dispatch_groups(&self, threads: [u32; 3]) -> [u32; 3] {
        let mut output = [1; 3];
        match self {
            Self::GlobalThreads(global_threads) => {
                for ((gt, t), y) in global_threads.iter().zip(threads.iter()).zip(output.iter_mut()) {
                    *y = *gt / *t + if *gt % *t != 0 { 1 } else { 0 };
                }
            }
            Self::Groups(groups) => {
                for (g, y) in groups.iter().zip(output.iter_mut()) {
                    *y = *g;
                }
            }
        }
        output
    }
}

/*
pub mod error {
    use super::*;
    #[doc(inline)]
    pub use krnl_types::kernel::error::*;
    use spirv::Capability;

    #[derive(Debug, thiserror::Error)]
    #[error("Kernel {:?} validation failed for \"{:?}\"{}!", self.name(), self.device, self.features_msg())]
    pub struct KernelValidationError {
        pub(super) device: Device,
        pub(super) info: KernelInfo,
    }

    impl KernelValidationError {
        fn kernel_info(&self) -> &KernelInfoInner {
            KernelInfoInner::get_from_kernel_info(&self.info)
        }
        fn name(&self) -> &str {
            self.kernel_info().name.as_str()
        }
        fn unenabled_capabilities(&self) -> Cow<[Capability]> {
            let capabilities = self.kernel_info().capabilities.as_slice();
            #[cfg(feature = "device")]
            {
                if let DeviceInner::Device(device) = &self.device.inner {
                    return capabilities
                        .iter()
                        .copied()
                        .filter(|x| !device.capability_enabled(*x))
                        .collect::<Vec<_>>()
                        .into();
                }
            }
            capabilities.into()
        }
        fn features_msg(&self) -> String {
            let caps = self.unenabled_capabilities();
            if caps.is_empty() {
                String::new()
            } else {
                format!(", capabilities {caps:?} are not enabled")
            }
        }
    }
}
use error::*;

pub mod builder {
    use super::*;

    pub struct KernelBuilder {
        device: Device,
        info: KernelInfo,
    }

    impl KernelBuilder {
        pub(super) fn new(device: Device, info: KernelInfo) -> Self {
            Self { device, info }
        }
        pub fn validate(mut self) -> Result<ValidatedKernelBuilder, KernelValidationError> {
            match &self.device.inner {
                DeviceInner::Host => (),
                #[cfg(feature = "device")]
                DeviceInner::Device(device) => {
                    let kernel_info = KernelInfoInner::get_from_kernel_info(&self.info);
                    if device.supports_vulkan_version(kernel_info.vulkan_version)
                        && kernel_info
                            .capabilities
                            .iter()
                            .copied()
                            .all(|x| device.capability_enabled(x))
                        && kernel_info
                            .extensions
                            .iter()
                            .all(|x| device.extension_enabled(x))
                    {
                        return Ok(ValidatedKernelBuilder {
                            device: self.device,
                            info: self.info,
                        });
                    }
                }
            }
            Err(KernelValidationError {
                device: self.device,
                info: self.info,
            })
        }
        pub fn build(self) -> Result<Kernel> {
            self.validate()?.build()
        }
    }

    pub struct ValidatedKernelBuilder {
        device: Device,
        info: KernelInfo,
    }

    impl ValidatedKernelBuilder {
        pub fn build(self) -> Result<Kernel> {
            match self.device.inner {
                DeviceInner::Host => unreachable!(),
                #[cfg(feature = "device")]
                DeviceInner::Device(device) => {
                    let cache = device.kernel_cache(self.info)?;
                    Ok(Kernel { device, cache })
                }
            }
        }
    }

    pub struct DispatchBuilder<'a> {
        #[cfg(feature = "device")]
        device: DeviceBase,
        #[cfg(feature = "device")]
        cache: Arc<KernelCache>,
        dim: Option<DispatchDimKind>,
        slices: Vec<NamedArg<SliceArg>>,
        push_consts: Vec<NamedArg<ScalarElem>>,
        _m: PhantomData<&'a ()>,
    }

    impl<'a> DispatchBuilder<'a> {
        #[cfg(feature = "device")]
        pub(super) fn new(device: DeviceBase, cache: Arc<KernelCache>) -> Self {
            let info = KernelInfoInner::get_from_kernel_info(cache.info());
            let slices = Vec::with_capacity(info.slice_infos.len());
            let push_consts = Vec::with_capacity(info.push_infos.len());
            Self {
                device,
                cache,
                dim: None,
                slices,
                push_consts,
                _m: PhantomData::default(),
            }
        }
        pub fn global_threads(mut self, global_threads: impl Into<DispatchDim>) -> Self {
            self.dim
                .replace(DispatchDimKind::GlobalThreads(global_threads.into()));
            self
        }
        pub fn groups(mut self, groups: impl Into<DispatchDim>) -> Self {
            self.dim.replace(DispatchDimKind::Groups(groups.into()));
            self
        }
        pub fn slice<'b: 'a>(
            mut self,
            name: impl Into<Cow<'static, str>>,
            slice: impl Into<ScalarSlice<'b>>,
        ) -> DispatchBuilder<'b> {
            self.slices.push(NamedArg {
                name: name.into(),
                arg: SliceArg::Slice(slice.into().into_raw_slice()),
            });
            DispatchBuilder {
                #[cfg(feature = "device")]
                device: self.device,
                #[cfg(feature = "device")]
                cache: self.cache,
                dim: self.dim,
                slices: self.slices,
                push_consts: self.push_consts,
                _m: PhantomData::default(),
            }
        }
        pub fn slice_mut<'b: 'a>(
            mut self,
            name: impl Into<Cow<'static, str>>,
            slice: impl Into<ScalarSliceMut<'b>>,
        ) -> DispatchBuilder<'b> {
            self.slices.push(NamedArg {
                name: name.into(),
                arg: SliceArg::SliceMut(slice.into().into_raw_slice_mut()),
            });
            DispatchBuilder {
                #[cfg(feature = "device")]
                device: self.device,
                #[cfg(feature = "device")]
                cache: self.cache,
                dim: self.dim,
                slices: self.slices,
                push_consts: self.push_consts,
                _m: PhantomData::default(),
            }
        }
        pub fn push(
            mut self,
            name: impl Into<Cow<'static, str>>,
            push: impl Into<ScalarElem>,
        ) -> Self {
            self.push_consts.push(NamedArg {
                name: name.into(),
                arg: push.into(),
            });
            self
        }
        pub fn build(self) -> Result<Dispatch<'a>> {
            #[cfg(feature = "device")]
            {
                let kernel_info = KernelInfoInner::get_from_kernel_info(self.cache.info());
                let kernel = &kernel_info.name;
                let module = &ModuleInner::get_from_kernel_info(&self.cache.info()).name;
                if kernel_info.safe {
                    return unsafe { self.build_unsafe() };
                } else {
                    return Err(format_err!("Kernel {kernel:?} in module {module:?} is unsafe, use `.build_unsafe()` instead."));
                }
            }
            unreachable!()
        }
        pub unsafe fn build_unsafe(self) -> Result<Dispatch<'a>> {
            #[cfg(feature = "device")]
            {
                let kernel_info = KernelInfoInner::get_from_kernel_info(self.cache.info());
                let kernel = &kernel_info.name;
                let module = &ModuleInner::get_from_kernel_info(self.cache.info()).name;
                let slice_infos = &kernel_info.slice_infos;
                let elements = if kernel_info.elementwise {
                    if let Some(slice_info) = slice_infos.iter().find(|x| x.elementwise) {
                        if let Some(slice) = self.slices.iter().find(|x| x.name == slice_info.name)
                        {
                            Some(slice.arg.len())
                        } else {
                            Some(0)
                        }
                    } else {
                        Some(0)
                    }
                } else {
                    None
                };
                let threads = &kernel_info.threads;
                let groups = if let Some(dim) = self.dim.as_ref() {
                    if kernel_info.elementwise {
                        let dim_kind = match dim {
                            DispatchDimKind::GlobalThreads(_) => "global_threads",
                            DispatchDimKind::Groups(_) => "groups",
                        };
                        return Err(format_err!("Can not specify `{dim_kind}` in elementwise kernel {kernel:?} in module {module:?}!"));
                    }
                    let ndim = kernel_info.threads.len();
                    match dim {
                        DispatchDimKind::GlobalThreads(global_threads) => {
                            if global_threads.ndim != threads.len() {
                                return Err(format_err!("Expected {ndim} dimensional `global_threads` for threads {threads:?} in kernel {kernel:?} in module {module:?}!"));
                            }
                            let mut groups = [1; 3];
                            for (g, (gt, t)) in groups.iter_mut().zip(
                                global_threads
                                    .dim
                                    .iter()
                                    .map(|x| *x as u32)
                                    .zip(threads.iter().copied()),
                            ) {
                                *g = gt / t + if gt % t != 0 { 1 } else { 0 };
                            }
                            groups
                        }
                        DispatchDimKind::Groups(groups) => {
                            if groups.ndim != threads.len() {
                                return Err(format_err!("Expected {ndim} dimensional `groups` for threads {threads:?} in kernel {kernel:?} in module {module:?}!"));
                            }
                            let [x, y, z] = groups.dim;
                            [x as u32, y as u32, z as u32]
                        }
                    }
                } else {
                    if !kernel_info.elementwise {
                        return Err(format_err!("Expected `.global_threads()` or `.groups()` for kernel {kernel:?} in module {module:?}!"));
                    }
                    // TODO: get active workgroups
                    let n = elements.unwrap() as u32;
                    let t = threads[0] as u32;
                    let g = n / t + if n % t != 0 { 1 } else { 0 };
                    [g, 1, 1]
                };
                if self.slices.len() > slice_infos.len() {
                    for slice in self.slices.iter() {
                        if slice_infos.iter().find(|x| x.name == slice.name).is_none() {
                            return Err(format_err!("Unexpected slice {:?}!", slice.name));
                        }
                    }
                    unreachable!()
                }
                let push_infos = &kernel_info.push_infos;
                let mut push_consts = vec![0u32; kernel_info.num_push_words as usize];
                let mut push_consts_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut push_consts);
                let mut buffers = Vec::with_capacity(slice_infos.len());
                let device = Device {
                    inner: DeviceInner::Device(self.device),
                };
                for slice_info in slice_infos.iter() {
                    if let Some(slice) = self.slices.iter().find(|x| x.name == slice_info.name) {
                        let slice_name = &slice_info.name;
                        let slice = match &slice.arg {
                            SliceArg::Slice(slice) => {
                                if slice_info.mutable {
                                    return Err(format_err!(
                                        "Expected `.slice_mut()` for slice {slice_name:?}!"
                                    ));
                                }
                                slice
                            }
                            SliceArg::SliceMut(slice) => {
                                if !slice_info.mutable {
                                    return Err(format_err!(
                                        "Expected `.slice()` for slice {slice_name:?}!"
                                    ));
                                }
                                slice
                            }
                        };
                        let slice_device = slice.device_ref();
                        if slice_device != &device {
                            return Err(format_err!("Expected device {device:?}, found {slice_device:?} for slice {slice_name:?}!"));
                        }
                        if slice_info.elementwise && slice.len() != elements.unwrap() {
                            return Err(format_err!("Expected elementwise slice {slice_name:?} to have len {}, found {}!", elements.unwrap(), slice.len()));
                        }
                        if slice.is_empty() {
                            if !groups.iter().any(|x| *x == 0) {
                                return Err(format_err!("Slice {slice_name:?} is empty!"));
                            }
                        } else {
                            if !slice_info.elementwise {
                                let len = slice.len() as u32;
                                let offset = push_infos
                                    .iter()
                                    .find(|x| {
                                        let name = &x.name;
                                        name.starts_with("__krnl_len_")
                                            && name.ends_with(&slice_info.name)
                                    })
                                    .unwrap()
                                    .offset as usize;
                                push_consts_bytes[offset..offset + 4]
                                    .copy_from_slice(len.to_ne_bytes().as_slice());
                            }
                            let buffer = slice
                                .as_device_slice()
                                .unwrap()
                                .device_buffer()
                                .unwrap()
                                .inner();
                            buffers.push(buffer);
                        }
                    } else {
                        return Err(format_err!("Expected slice {:?}!", slice_info.name));
                    }
                }
                if let Some(elements) = elements {
                    let offset = push_infos
                        .iter()
                        .find(|x| x.name == "__krnl_elements")
                        .unwrap()
                        .offset as usize;
                    push_consts_bytes[offset..offset + 4]
                        .copy_from_slice((elements as u32).to_ne_bytes().as_slice());
                }
                let num_push_consts = push_infos
                    .iter()
                    .filter(|x| !x.name.starts_with("__krnl"))
                    .count();
                if self.push_consts.len() > num_push_consts {
                    for push_info in push_infos.iter() {
                        if self
                            .push_consts
                            .iter()
                            .find(|x| x.name == push_info.name)
                            .is_none()
                        {
                            return Err(format_err!("Expected push {:?}!", push_info.name));
                        }
                    }
                    unreachable!()
                } else {
                    for push in self.push_consts {
                        if let Some(push_info) =
                            kernel_info.push_infos.iter().find(|x| x.name == push.name)
                        {
                            let push_name = &push.name;
                            let push_ty = push.arg.scalar_type();
                            let push_info_ty = push_info.scalar_type;
                            if push_ty != push_info_ty {
                                return Err(format_err!("Expected push {push_name:?} to have scalar type {push_info_ty:?}, found {push_ty:?}!"));
                            }
                            let offset = push_info.offset as usize;
                            write_scalar_elem_to_bytes(
                                &push.arg,
                                &mut push_consts_bytes[offset..offset + push_ty.size()],
                            );
                        } else {
                            return Err(format_err!("Unexpected push {:?}!", push.name));
                        }
                    }
                }
                let compute = if !groups.iter().any(|x| *x == 0) {
                    Some(Compute {
                        cache: self.cache,
                        groups,
                        buffers,
                        push_consts,
                    })
                } else {
                    None
                };
                let device = device.inner.unwrap_device();
                return Ok(Dispatch {
                    device,
                    compute,
                    _m: PhantomData::default(),
                });
            }
            unreachable!()
        }
    }

    enum DispatchDimKind {
        GlobalThreads(DispatchDim),
        Groups(DispatchDim),
    }

    fn write_scalar_elem_to_bytes(scalar_elem: &ScalarElem, bytes: &mut [u8]) {
        use ScalarElem::*;
        match scalar_elem.to_scalar_bits() {
            U8(x) => {
                bytes.copy_from_slice([x].as_ref());
            }
            U16(x) => {
                bytes.copy_from_slice(x.to_ne_bytes().as_slice());
            }
            U32(x) => {
                bytes.copy_from_slice(x.to_ne_bytes().as_slice());
            }
            U64(x) => {
                bytes.copy_from_slice(x.to_ne_bytes().as_slice());
            }
            _ => unreachable!(),
        }
    }
}
use builder::*;

pub struct Kernel {
    #[cfg(feature = "device")]
    device: DeviceBase,
    #[cfg(feature = "device")]
    cache: Arc<KernelCache>,
}

impl Kernel {
    pub fn builder(device: Device, info: KernelInfo) -> KernelBuilder {
        KernelBuilder::new(device, info)
    }
    pub fn dispatch_builder(&self) -> DispatchBuilder {
        #[cfg(feature = "device")]
        {
            return DispatchBuilder::new(self.device.clone(), self.cache.clone());
        }
        unreachable!()
    }
}

pub struct Dispatch<'a> {
    #[cfg(feature = "device")]
    device: DeviceBase,
    #[cfg(feature = "device")]
    compute: Option<Compute>,
    _m: PhantomData<&'a ()>,
}

impl<'a> Dispatch<'a> {
    pub fn dispatch(self) -> Result<()> {
        #[cfg(feature = "device")]
        {
            if let Some(compute) = self.compute {
                self.device.compute(compute)?;
            }
        }
        Ok(())
    }
}

pub struct DispatchDim {
    dim: [usize; 3],
    ndim: usize,
}

impl From<usize> for DispatchDim {
    fn from(dim: usize) -> Self {
        Self {
            dim: [dim, 1, 1],
            ndim: 1,
        }
    }
}

impl From<[usize; 1]> for DispatchDim {
    fn from(dim: [usize; 1]) -> Self {
        Self {
            dim: [dim[0], 1, 1],
            ndim: 1,
        }
    }
}

impl From<[usize; 2]> for DispatchDim {
    fn from(dim: [usize; 2]) -> Self {
        Self {
            dim: [dim[0], dim[1], 1],
            ndim: 2,
        }
    }
}

impl From<[usize; 3]> for DispatchDim {
    fn from(dim: [usize; 3]) -> Self {
        Self {
            dim: [dim[0], dim[1], dim[2]],
            ndim: 3,
        }
    }
}

struct NamedArg<T> {
    name: Cow<'static, str>,
    arg: T,
}

enum SliceArg {
    Slice(RawSlice),
    SliceMut(RawSlice),
}

impl SliceArg {
    fn len(&self) -> usize {
        match self {
            Self::Slice(x) => x.len(),
            Self::SliceMut(x) => x.len(),
        }
    }
}*/
