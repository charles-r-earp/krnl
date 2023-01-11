#[doc(inline)]
pub use krnl_macros::module;

#[doc(hidden)]
pub mod __private {
    #[cfg(feature = "device")]
    use crate::device::{Compute, DeviceBufferInner, KernelCache};
    use crate::{
        anyhow::{bail, format_err, Result},
        buffer::{RawSlice, ScalarSlice, ScalarSliceMut, Slice, SliceMut},
        device::{Device, DeviceInner, Features},
        krnl_core::glam::{UVec2, UVec3},
        scalar::{Scalar, ScalarElem, ScalarType},
    };
    use std::{
        borrow::Cow,
        collections::HashMap,
        fmt::{self, Debug},
        marker::PhantomData,
        sync::Arc,
    };
    include!("kernel/__private/metadata.rs");

    pub mod builder {
        use super::*;

        pub struct KernelBuilder {
            name: Cow<'static, str>,
            spirv: &'static [u32],
            threads: UVec3,
            spec_consts: Vec<u32>,
            buffer_descs: &'static [BufferDesc],
            push_consts_size: u32,
            features: Features,
        }

        impl KernelBuilder {
            pub(super) fn new(name: Cow<'static, str>, spirv: &'static [u32]) -> Self {
                KernelBuilder {
                    name,
                    spirv,
                    threads: UVec3::new(1, 1, 1),
                    spec_consts: Vec::new(),
                    buffer_descs: &[],
                    push_consts_size: 0,
                    features: Features::default(),
                }
            }
            pub fn threads(mut self, threads: UVec3) -> Self {
                self.threads = threads;
                self
            }
            pub fn specs(mut self, specs: &[ScalarElem]) -> Self {
                self.spec_consts = specs
                    .iter()
                    .flat_map(|x| {
                        use ScalarElem::*;
                        match x.to_scalar_bits() {
                            U8(x) => [u32::from_ne_bytes([x, 0, 0, 0]), 0].into_iter().take(1),
                            U16(x) => {
                                let [x1, x2] = x.to_ne_bytes();
                                [u32::from_ne_bytes([x1, x2, 0, 0]), 0].into_iter().take(1)
                            }
                            U32(x) => [x, 0].into_iter().take(1),
                            U64(x) => {
                                let x = x.to_ne_bytes();
                                [
                                    u32::from_ne_bytes([x[0], x[1], x[2], x[3]]),
                                    u32::from_ne_bytes([x[4], x[5], x[6], x[7]]),
                                ]
                                .into_iter()
                                .take(2)
                            }
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                self
            }
            pub fn buffer_descs(mut self, buffer_descs: &'static [BufferDesc]) -> Self {
                self.buffer_descs = buffer_descs;
                self
            }
            pub fn push_consts_size(mut self, push_consts_size: u32) -> Self {
                self.push_consts_size = push_consts_size;
                self
            }
            pub fn features(mut self, features: Features) -> Self {
                self.features = features;
                self
            }
            pub fn build(self, device: Device) -> Result<Kernel> {
                let kernel_name = &self.name;
                match &device.inner {
                    #[cfg(feature = "device")]
                    DeviceInner::Device(vulkan_device) => {
                        let features = &self.features;
                        let device_features = vulkan_device.features();
                        if !device_features.contains(&self.features) {
                            bail!("Kernel `{kernel_name}` requires {features:?}, device has {device_features:?}!");
                        }
                        let desc = KernelDesc {
                            name: self.name,
                            buffer_descs: self.buffer_descs,
                            threads: self.threads,
                            push_consts_size: self.push_consts_size,
                        };
                        let cache =
                            vulkan_device.kernel_cache(self.spirv, self.spec_consts, desc)?;
                        Ok(Kernel { device, cache })
                    }
                    DeviceInner::Host => Err(format_err!(
                        "Kernel `{kernel_name}` cannot be built for the host!"
                    )),
                }
            }
        }

        pub enum DispatchDim {
            U32(u32),
            UVec2(UVec2),
            UVec3(UVec3),
        }

        impl DispatchDim {
            fn to_array(&self) -> [u32; 3] {
                match self {
                    Self::U32(dim) => [*dim, 1, 1],
                    Self::UVec2(dim) => [dim.x, dim.y, 1],
                    Self::UVec3(dim) => dim.to_array(),
                }
            }
        }

        impl From<u32> for DispatchDim {
            fn from(dim: u32) -> Self {
                Self::U32(dim)
            }
        }

        impl From<UVec2> for DispatchDim {
            fn from(dim: UVec2) -> Self {
                Self::UVec2(dim)
            }
        }

        impl From<UVec3> for DispatchDim {
            fn from(dim: UVec3) -> Self {
                Self::UVec3(dim)
            }
        }

        impl Debug for DispatchDim {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self {
                    Self::U32(dim) => dim.fmt(f),
                    Self::UVec2(dim) => dim.fmt(f),
                    Self::UVec3(dim) => dim.fmt(f),
                }
            }
        }

        pub enum DispatchSlice<'a> {
            Slice(ScalarSlice<'a>),
            SliceMut(ScalarSliceMut<'a>),
        }

        impl DispatchSlice<'_> {
            fn as_raw_slice(&self) -> &RawSlice {
                match self {
                    Self::Slice(slice) => slice.as_raw_slice(),
                    Self::SliceMut(slice) => slice.as_raw_slice(),
                }
            }
        }

        impl<'a, T: Scalar> From<Slice<'a, T>> for DispatchSlice<'a> {
            fn from(slice: Slice<'a, T>) -> Self {
                Self::Slice(ScalarSlice::from(slice))
            }
        }

        impl<'a, T: Scalar> From<SliceMut<'a, T>> for DispatchSlice<'a> {
            fn from(slice: SliceMut<'a, T>) -> Self {
                Self::SliceMut(ScalarSliceMut::from(slice))
            }
        }

        pub struct DispatchBuilder<'a> {
            device: Device,
            #[cfg(feature = "device")]
            cache: Arc<KernelCache>,
            groups: Option<[u32; 3]>,
            #[cfg(feature = "device")]
            buffers: Vec<Arc<DeviceBufferInner>>,
            push_consts: Vec<u32>,
            _m: PhantomData<&'a ()>,
        }

        impl<'a> DispatchBuilder<'a> {
            #[cfg(feature = "device")]
            pub(super) fn new(
                device: Device,
                cache: Arc<KernelCache>,
                slices: &mut [DispatchSlice<'a>],
                push_consts: &[u8],
            ) -> Result<Self> {
                let desc = cache.desc();
                let kernel_name = &desc.name;
                if device.is_host() {
                    bail!("Kernel `{kernel_name}` expected Device(_), found Host!");
                }
                let push_consts_u8 = push_consts;
                let mut push_consts: Vec<u32> = Vec::with_capacity(desc.push_consts_size as usize);
                push_consts.extend(push_consts_u8.chunks(4).map(|x| {
                    u32::from_ne_bytes([
                        x.get(0).copied().unwrap_or_default(),
                        x.get(1).copied().unwrap_or_default(),
                        x.get(2).copied().unwrap_or_default(),
                        x.get(3).copied().unwrap_or_default(),
                    ])
                }));
                let mut items: Option<usize> = None;
                let mut buffers = Vec::with_capacity(slices.len());
                for (buffer_desc, slice) in desc.buffer_descs.iter().zip(slices) {
                    let name = buffer_desc.name;
                    let slice = slice.as_raw_slice();
                    if buffer_desc.item {
                        if let Some(items) = items {
                            if items != slice.len() {
                                bail!("Kernel `{kernel_name}` `{name}` has {} items, expected {items}!", slice.len());
                            }
                        } else {
                            items.replace(slice.len());
                        }
                    }
                    let slice_device = slice.device_ref();
                    if slice_device != &device {
                        bail!("Kernel `{kernel_name}` `{name}` expected {device:?} found {slice_device:?}!");
                    }
                    let device_slice = slice.as_device_slice().unwrap();
                    if let Some(device_buffer) = device_slice.device_buffer() {
                        buffers.push(device_buffer.inner().clone());
                    } else {
                        bail!("Kernel `{kernel_name}` `{name}` is empty!");
                    }
                    push_consts.extend([slice.offset() as u32, slice.len() as u32]);
                }
                let mut builder = Self {
                    device,
                    cache,
                    groups: None,
                    buffers,
                    push_consts,
                    _m: PhantomData::default(),
                };
                if let Some(items) = items {
                    builder = builder.global_threads((items as u32).into())?;
                }
                Ok(builder)
            }
            pub fn global_threads(mut self, global_threads: DispatchDim) -> Result<Self> {
                #[cfg(feature = "device")]
                {
                    let desc = self.cache.desc();
                    let threads = desc.threads;
                    let kernel_name = &desc.name;
                    if global_threads.to_array().iter().any(|x| *x == 0) {
                        bail!("Kernel `{kernel_name}` global_threads cannot be 0, found {global_threads:?}!");
                    }
                    let mut groups = [0; 3];
                    for (g, (gt, t)) in groups
                        .as_mut()
                        .iter_mut()
                        .zip(global_threads.to_array().iter().zip(threads.as_ref()))
                    {
                        *g = gt / t;
                        if gt % t != 0 {
                            *g += 1;
                        }
                    }
                    self.groups.replace(groups);
                    Ok(self)
                }
                #[cfg(not(feature = "device"))]
                {
                    unreachable!()
                }
            }
            pub fn groups(mut self, groups: DispatchDim) -> Result<Self> {
                #[cfg(feature = "device")]
                {
                    let kernel_name = &self.cache.desc().name;
                    if groups.to_array().iter().any(|x| *x == 0) {
                        bail!("Kernel `{kernel_name}` groups cannot be 0, found {groups:?}!");
                    }
                    self.groups.replace(groups.to_array());
                    Ok(self)
                }
                #[cfg(not(feature = "device"))]
                {
                    unreachable!()
                }
            }
            pub unsafe fn dispatch(self) -> Result<()> {
                #[cfg(feature = "device")]
                {
                    let groups = if let Some(groups) = self.groups {
                        groups
                    } else {
                        let kernel_name = &self.cache.desc().name;
                        bail!("Kernel `{kernel_name}` global_threads or groups are required!");
                    };
                    let compute = Compute {
                        cache: self.cache,
                        groups,
                        buffers: self.buffers,
                        push_consts: self.push_consts,
                    };
                    self.device.as_device().unwrap().compute(compute)
                }
                #[cfg(not(feature = "device"))]
                {
                    unreachable!()
                }
            }
        }
    }
    use builder::{DispatchBuilder, DispatchSlice, KernelBuilder};

    /*
    pub struct Version {
        major: u32,
        minor: u32,
        patch: u32,
    }

    // krnl-macros -> krnlc
    struct ModuleData {
        source: String,
        dependencies: Vec<String>,
        build: bool,
        krnl_crate: String,
    }

    pub struct ModuleDesc {
        module_data_hash: u64,
        kernel_descs: &'static [KernelDesc2],
        build: bool,
    }

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
    }*/

    #[derive(Debug)]
    pub(crate) struct KernelDesc {
        name: Cow<'static, str>,
        buffer_descs: &'static [BufferDesc],
        threads: UVec3,
        push_consts_size: u32,
    }

    impl KernelDesc {
        pub(crate) fn name(&self) -> &str {
            &self.name
        }
        pub(crate) fn buffer_descs(&self) -> &[BufferDesc] {
            &self.buffer_descs
        }
        pub(crate) fn threads(&self) -> UVec3 {
            self.threads
        }
        pub(crate) fn push_consts_size(&self) -> u32 {
            self.push_consts_size
        }
    }

    #[derive(Debug)]
    pub struct BufferDesc {
        name: &'static str,
        scalar_type: ScalarType,
        mutable: bool,
        item: bool,
    }

    impl BufferDesc {
        pub const fn new(name: &'static str, scalar_type: ScalarType) -> Self {
            Self {
                name,
                scalar_type,
                mutable: false,
                item: false,
            }
        }
        pub const fn with_mutable(mut self, mutable: bool) -> Self {
            self.mutable = mutable;
            self
        }
        pub(crate) fn mutable(&self) -> bool {
            self.mutable
        }
        pub const fn with_item(mut self, item: bool) -> Self {
            self.item = true;
            self
        }
    }

    pub struct Kernel {
        device: Device,
        #[cfg(feature = "device")]
        cache: Arc<KernelCache>,
    }

    impl Kernel {
        pub fn builder(name: impl Into<Cow<'static, str>>, spirv: &'static [u32]) -> KernelBuilder {
            KernelBuilder::new(name.into(), spirv)
        }
        pub fn dispatch_builder<'a>(
            &self,
            slices: &mut [DispatchSlice<'a>],
            push_consts: &[u8],
        ) -> Result<DispatchBuilder<'a>> {
            #[cfg(feature = "device")]
            {
                DispatchBuilder::new(self.device.clone(), self.cache.clone(), slices, push_consts)
            }
            #[cfg(not(feature = "device"))]
            {
                unreachable!()
            }
        }
    }
}
