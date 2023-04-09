/*!

A [`Device`](crate::device::Device) is used to create [buffers](crate::buffer) and [kernels](crate::_kernel_programming_guide).
[`Device::host()`](crate::device::Device::host) method creates the host, which is merely the lack of a device.

Note: Kernels can not be created for the host.

Creating a device and printing out useful info:
```no_build
# fn main() -> Result<()> {
let device = Device::builder()
    .index(1)
    .build()?;
dbg!(device.info());
# }
```

# Queues
Devices can support multiple compute queues and a dedicated transfer queue.

Dispatching a kernel:
- Waits for immutable access to slice arguments.
- Waits for mutable access to mutable slice arguments.
- Blocks until the kernel is queued.

One kernel can be queued while another is executing on that queue.
*/

use crate::{
    buffer::{ScalarSlice, ScalarSliceMut, Slice, SliceMut},
    scalar::{Scalar, ScalarElem, ScalarType},
};
use anyhow::{bail, Result};
#[cfg(feature = "device")]
use rspirv::{binary::Assemble, dr::Operand};
use serde::Deserialize;
#[cfg(feature = "device")]
use std::{collections::HashMap, hash::Hash, ops::Range};
use std::{
    fmt::{self, Debug},
    sync::Arc,
};

#[cfg(all(not(target_arch = "wasm32"), feature = "device"))]
mod vulkan_engine;
#[cfg(all(not(target_arch = "wasm32"), feature = "device"))]
use vulkan_engine::Engine;

#[cfg(all(target_arch = "wasm32", feature = "device"))]
compile_error!("device feature not supported on wasm");

/// Errors.
pub mod error {
    use std::fmt::{self, Debug, Display};

    /** Device is unavailable.

    - The "device" feature is not enabled.
    - Failed to load the Vulkan library.
    */
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    #[error("DeviceUnavailable")]
    pub struct DeviceUnavailable;

    #[cfg(feature = "device")]
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    #[cfg_attr(
        feature = "device",
        error("Device index {index} is out of range 0..{devices}!")
    )]
    #[cfg_attr(not(feature = "device"), error("unreachable!"))]
    pub struct DeviceIndexOutOfRange {
        #[cfg(feature = "device")]
        pub(super) index: usize,
        #[cfg(feature = "device")]
        pub(super) devices: usize,
    }

    /// The Device was lost.
    #[derive(Clone, Copy, thiserror::Error)]
    pub struct DeviceLost {
        #[cfg(feature = "device")]
        pub(super) index: usize,
        #[cfg(feature = "device")]
        pub(super) handle: u64,
    }

    impl Debug for DeviceLost {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            #[cfg(feature = "device")]
            {
                f.debug_tuple("DeviceLost")
                    .field(&self.index)
                    .field(&(self.handle as *const ()))
                    .finish()
            }
            #[cfg(not(feature = "device"))]
            {
                write!(f, "DeviceLost")
            }
        }
    }

    impl Display for DeviceLost {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(self, f)
        }
    }
}
use error::*;

/// Builders.
pub mod builder {
    use super::*;

    /// Builder for creating a [`Device`].
    pub struct DeviceBuilder {
        #[cfg(feature = "device")]
        pub(super) options: DeviceOptions,
    }

    impl DeviceBuilder {
        /// Index of the device, defaults to 0.
        pub fn index(self, index: usize) -> Self {
            #[cfg(feature = "device")]
            {
                let mut this = self;
                this.options.index = index;
                this
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = index;
                self
            }
        }
        /** Creates a device.
        **errors**
        - [`DeviceUnavailable`](super::error::DeviceUnavailable)
        - [`DeviceIndexOutofRange`](super::error::DeviceIndexOutOfRange)
        - The device could not be created.
        */
        pub fn build(self) -> Result<Device> {
            #[cfg(feature = "device")]
            {
                let raw = RawDevice::new(self.options)?;
                Ok(Device {
                    inner: DeviceInner::Device(raw),
                })
            }
            #[cfg(not(feature = "device"))]
            {
                Err(DeviceUnavailable.into())
            }
        }
    }
}
use builder::*;

#[cfg(feature = "device")]
trait DeviceEngine {
    type DeviceBuffer: DeviceEngineBuffer<Engine = Self>;
    type Kernel: DeviceEngineKernel<Engine = Self, DeviceBuffer = Self::DeviceBuffer>;
    fn new(options: DeviceOptions) -> Result<Arc<Self>>;
    fn handle(&self) -> u64;
    fn info(&self) -> &Arc<DeviceInfo>;
    fn wait(&self) -> Result<(), DeviceLost>;
    //fn performance_metrics(&self) -> PerformanceMetrics;
}

#[cfg(feature = "device")]
struct DeviceOptions {
    index: usize,
    optimal_features: Features,
}

#[cfg(feature = "device")]
trait DeviceEngineBuffer: Sized {
    type Engine;
    unsafe fn uninit(engine: Arc<Self::Engine>, len: usize) -> Result<Self>;
    fn upload(&self, data: &[u8]) -> Result<()>;
    fn download(&self, data: &mut [u8]) -> Result<()>;
    fn transfer(&self, dst: &Self) -> Result<()>;
    fn engine(&self) -> &Arc<Self::Engine>;
    fn offset(&self) -> usize;
    fn len(&self) -> usize;
    fn slice(self: &Arc<Self>, range: Range<usize>) -> Option<Arc<Self>>;
}

#[cfg(feature = "device")]
trait DeviceEngineKernel: Sized {
    type Engine;
    type DeviceBuffer;
    fn cached(
        engine: Arc<Self::Engine>,
        key: KernelKey,
        desc_fn: impl FnOnce() -> Result<Arc<KernelDesc>>,
    ) -> Result<Arc<Self>>;
    unsafe fn dispatch(
        &self,
        groups: [u32; 3],
        buffers: &[Arc<Self::DeviceBuffer>],
        push_consts: Vec<u8>,
    ) -> Result<()>;
    fn engine(&self) -> &Arc<Self::Engine>;
    fn desc(&self) -> &Arc<KernelDesc>;
}

/** A device.

Devices can be cloned, which is equivalent to [`Arc::clone()`].

Devices (other than the host) are unique:
```no_run
# use krnl::{anyhow::Result, device::Device};
# fn main() -> Result<()> {
let a = Device::builder().build()?;
let b = Device::builder().build()?;
assert_ne!(a, b);
# Ok(())
# }
```
*/
#[derive(Clone, Eq, PartialEq)]
pub struct Device {
    inner: DeviceInner,
}

impl Device {
    /// The host.
    pub const fn host() -> Self {
        Self {
            inner: DeviceInner::Host,
        }
    }
    /// A builder for creating a device.
    pub fn builder() -> DeviceBuilder {
        DeviceBuilder {
            #[cfg(feature = "device")]
            options: DeviceOptions {
                index: 0,
                optimal_features: Features::empty()
                    .with_shader_int8(true)
                    .with_shader_int16(true)
                    .with_shader_int64(true)
                    .with_shader_float16(true)
                    .with_shader_float64(true),
            },
        }
    }
    /// Is the host.
    pub fn is_host(&self) -> bool {
        self.inner.is_host()
    }
    /// Is a device.
    pub fn is_device(&self) -> bool {
        self.inner.is_device()
    }
    pub(crate) fn inner(&self) -> &DeviceInner {
        &self.inner
    }
    /** Device info.

    The host returns None. */
    pub fn info(&self) -> Option<&Arc<DeviceInfo>> {
        match self.inner() {
            DeviceInner::Host => None,
            #[cfg(feature = "device")]
            DeviceInner::Device(raw) => Some(raw.info()),
        }
    }
    /** Wait for previous work to finish.

    If host, this does nothing.

    Operations (like kernel dispatches) executed after this method is called
    will not block, and will not be waited on.

    This is primarily for benchmarking, manual synchronization is unnecessary.

    **errors**
    Returns an error if the device was lost while waiting. */
    pub fn wait(&self) -> Result<(), DeviceLost> {
        match self.inner() {
            DeviceInner::Host => Ok(()),
            #[cfg(feature = "device")]
            DeviceInner::Device(raw) => raw.wait(),
        }
    }
}

/** Prints `Device(index, handle)` where handle is a u64 that uniquely
identifies the device.

See [`.info()`](Device::info) for printing device info. */
impl Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "device")]
impl From<RawDevice> for Device {
    fn from(device: RawDevice) -> Self {
        Self {
            inner: DeviceInner::Device(device),
        }
    }
}

#[derive(Clone, Eq, PartialEq, derive_more::Unwrap)]
pub(crate) enum DeviceInner {
    Host,
    #[cfg(feature = "device")]
    Device(RawDevice),
}

impl DeviceInner {
    pub(crate) fn is_host(&self) -> bool {
        matches!(self, Self::Host)
    }
    pub(crate) fn is_device(&self) -> bool {
        !self.is_host()
    }
}

impl Debug for DeviceInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => f.debug_struct("Host").finish(),
            #[cfg(feature = "device")]
            Self::Device(raw_device) => raw_device.fmt(f),
        }
    }
}

#[cfg(feature = "device")]
#[derive(Clone)]
pub(crate) struct RawDevice {
    engine: Arc<Engine>,
}

#[cfg(feature = "device")]
impl RawDevice {
    fn new(options: DeviceOptions) -> Result<Self> {
        let engine = Engine::new(options)?;
        Ok(Self { engine })
    }
    fn info(&self) -> &Arc<DeviceInfo> {
        self.engine.info()
    }
    fn wait(&self) -> Result<(), DeviceLost> {
        self.engine.wait()
    }
}

#[cfg(feature = "device")]
impl PartialEq for RawDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.engine, &other.engine)
    }
}

#[cfg(feature = "device")]
impl Eq for RawDevice {}

#[cfg(feature = "device")]
impl Debug for RawDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let index = self.info().index;
        let handle = self.engine.handle() as *const ();
        f.debug_tuple("Device")
            .field(&index)
            .field(&handle)
            .finish()
    }
}

#[cfg(feature = "device")]
#[repr(transparent)]
#[derive(Clone)]
pub(crate) struct DeviceBuffer {
    inner: Arc<<Engine as DeviceEngine>::DeviceBuffer>,
}

#[cfg(feature = "device")]
impl DeviceBuffer {
    pub(crate) unsafe fn uninit(device: RawDevice, len: usize) -> Result<Self> {
        let inner =
            unsafe { <Engine as DeviceEngine>::DeviceBuffer::uninit(device.engine, len)?.into() };
        Ok(Self { inner })
    }
    pub(crate) fn upload(&self, data: &[u8]) -> Result<()> {
        self.inner.upload(data)
    }
    pub(crate) fn download(&self, data: &mut [u8]) -> Result<()> {
        self.inner.download(data)
    }
    pub(crate) fn transfer(&self, dst: &Self) -> Result<()> {
        self.inner.transfer(&dst.inner)
    }
    pub(crate) fn offset(&self) -> usize {
        self.inner.offset()
    }
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
    pub(crate) fn device(&self) -> RawDevice {
        RawDevice {
            engine: self.inner.engine().clone(),
        }
    }
    pub(crate) fn slice(&self, range: Range<usize>) -> Option<Self> {
        let inner = self.inner.slice(range)?;
        Some(Self { inner })
    }
}

/** Features

Features supported by a device. See [`DeviceInfo::features`].

Kernels can not be compiled unless the device supports all features used.

This is a subset of [vulkano::device::Features](https://docs.rs/vulkano/latest/vulkano/device/struct.Features.html).

Use features to specialize or provide a more helpful error message:
```
# use krnl::device::Features;
# fn main() {
# let features = Features::empty();
if features.shader_int8() {
    /* u8 impl */
} else {
    /* fallback */
}
# }
```

*/
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub struct Features {
    shader_int8: bool,
    shader_int16: bool,
    shader_int64: bool,
    shader_float16: bool,
    shader_float64: bool,
}

impl Features {
    /// No features.
    pub const fn empty() -> Self {
        Self {
            shader_int8: false,
            shader_int16: false,
            shader_int64: false,
            shader_float16: false,
            shader_float64: false,
        }
    }
    /// 8 bit scalars.
    pub const fn shader_int8(&self) -> bool {
        self.shader_int8
    }
    /// Adds `shader_int8`.
    pub const fn with_shader_int8(mut self, shader_int8: bool) -> Self {
        self.shader_int8 = shader_int8;
        self
    }
    /// 16 bit scalars.
    pub const fn shader_int16(&self) -> bool {
        self.shader_int16
    }
    /// Adds `shader_int16`.
    pub const fn with_shader_int16(mut self, shader_int16: bool) -> Self {
        self.shader_int16 = shader_int16;
        self
    }
    /// 64 bit scalars.
    pub const fn shader_int64(&self) -> bool {
        self.shader_int64
    }
    /// Adds `shader_int64`.
    pub const fn with_shader_int64(mut self, shader_int64: bool) -> Self {
        self.shader_int64 = shader_int64;
        self
    }
    /// f16 intrinsics.
    pub const fn shader_float16(&self) -> bool {
        self.shader_float16
    }
    /// Adds `shader_float16`.
    pub const fn with_shader_float16(mut self, shader_float16: bool) -> Self {
        self.shader_float16 = shader_float16;
        self
    }
    /// f64.
    pub const fn shader_float64(&self) -> bool {
        self.shader_float64
    }
    /// Adds `shader_float64`.
    pub const fn with_shader_float64(mut self, shader_float64: bool) -> Self {
        self.shader_float64 = shader_float64;
        self
    }
    /// Contains all features of `other`.
    pub const fn contains(&self, other: &Features) -> bool {
        (self.shader_int8 || !other.shader_int8)
            && (self.shader_int16 || !other.shader_int16)
            && (self.shader_int64 || !other.shader_int64)
            && (self.shader_float16 || !other.shader_float16)
            && (self.shader_float64 || !other.shader_float64)
    }
    /// All features of `self` and `other`.
    pub const fn union(mut self, other: &Features) -> Self {
        self.shader_int8 |= other.shader_int8;
        self.shader_int16 |= other.shader_int16;
        self.shader_int64 |= other.shader_int64;
        self.shader_float16 |= other.shader_float16;
        self.shader_float64 |= other.shader_float64;
        self
    }
}

/// Device info.
#[derive(Debug)]
#[allow(dead_code)]
pub struct DeviceInfo {
    index: usize,
    name: String,
    compute_queues: usize,
    transfer_queues: usize,
    features: Features,
}

impl DeviceInfo {
    /// Device features.
    pub fn features(&self) -> Features {
        self.features
    }
}

/*
#[derive(Clone, Copy, Debug)]
struct TransferMetrics {
    bytes: usize,
    time: Duration,
}

#[derive(Clone, Copy, Debug)]
struct KernelMetrics {
    dispatches: usize,
    time: Duration,
}

#[derive(Clone, Debug)]
pub struct PerformanceMetrics {
    upload: TransferMetrics,
    download: TransferMetrics,
    kernels: HashMap<String, KernelMetrics>,
}*/

/*
#[derive(Default, Clone)]
struct KernelKey {
    inner: Arc<()>,
    spec_consts: Vec<ScalarElem>,
}

impl PartialEq for KernelKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) && self.spec_consts == other.spec_consts
    }
}

impl Eq for KernelKey {}

impl Hash for KernelKey {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (Arc::as_ptr(&self.inner) as usize).hash(hasher);
        for spec in self.spec_consts.iter().copied() {
            use ScalarElem::*;
            match spec {
                U8(x) => x.hash(hasher),
                I8(x) => x.hash(hasher),
                U16(x) => x.hash(hasher),
                I16(x) => x.hash(hasher),
                F16(x) => x.to_bits().hash(hasher),
                BF16(x) => x.to_bits().hash(hasher),
                U32(x) => x.hash(hasher),
                I32(x) => x.hash(hasher),
                F32(x) => x.to_bits().hash(hasher),
                F64(x) => x.to_bits().hash(hasher),
                _ => unreachable!(),
            }
        }
    }
}*/

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Deserialize, Debug)]
struct KernelDesc {
    name: String,
    hash: u64,
    spirv: Vec<u32>,
    features: Features,
    threads: Vec<u32>,
    safe: bool,
    spec_descs: Vec<SpecDesc>,
    slice_descs: Vec<SliceDesc>,
    push_descs: Vec<PushDesc>,
}

#[cfg(feature = "device")]
impl KernelDesc {
    fn push_consts_range(&self) -> u32 {
        let mut size: usize = self.push_descs.iter().map(|x| x.scalar_type.size()).sum();
        while size % 4 != 0 {
            size += 1;
        }
        size += self.slice_descs.len() * 2 * 4;
        size.try_into().unwrap()
    }
    fn specialize(&self, threads: Vec<u32>, spec_consts: &[ScalarElem]) -> Result<Self> {
        use rspirv::spirv::{Decoration, Op};
        let mut module = rspirv::dr::load_words(&self.spirv).unwrap();
        let mut spec_ids = HashMap::<u32, u32>::with_capacity(spec_consts.len());
        for inst in module.annotations.iter() {
            if inst.class.opcode == Op::Decorate {
                if let [Operand::IdRef(id), Operand::Decoration(Decoration::SpecId), Operand::LiteralInt32(spec_id)] =
                    inst.operands.as_slice()
                {
                    spec_ids.insert(*id, *spec_id);
                }
            }
        }
        for inst in module.types_global_values.iter_mut() {
            if inst.class.opcode == Op::SpecConstant {
                if let Some(result_id) = inst.result_id {
                    if let Some(spec_id) = spec_ids.get(&result_id) {
                        if let Some(value) = spec_consts.get(*spec_id as usize) {
                            match inst.operands.as_mut_slice() {
                                [Operand::LiteralInt32(a)] => {
                                    bytemuck::bytes_of_mut(a).copy_from_slice(value.as_bytes());
                                }
                                [Operand::LiteralInt32(a), Operand::LiteralInt32(b)] => {
                                    bytemuck::bytes_of_mut(a)
                                        .copy_from_slice(&value.as_bytes()[..8]);
                                    bytemuck::bytes_of_mut(b)
                                        .copy_from_slice(&value.as_bytes()[9..]);
                                }
                                _ => unreachable!("{:?}", inst.operands),
                            }
                        }
                    }
                }
            }
        }
        let spirv = module.assemble();
        Ok(Self {
            spirv,
            spec_descs: Vec::new(),
            threads,
            ..self.clone()
        })
    }
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Deserialize, Debug)]
struct SpecDesc {
    #[allow(unused)]
    name: String,
    scalar_type: ScalarType,
    thread_dim: Option<usize>,
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Deserialize, Debug)]
struct SliceDesc {
    name: String,
    scalar_type: ScalarType,
    mutable: bool,
    item: bool,
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Deserialize, Debug)]
struct PushDesc {
    #[allow(unused)]
    name: String,
    scalar_type: ScalarType,
}

#[cfg(feature = "device")]
#[derive(PartialEq, Eq, Hash, Debug)]
struct KernelKey {
    id: usize,
    spec_bytes: Vec<u8>,
}

#[doc(hidden)]
pub mod __private {
    use super::*;

    #[doc(hidden)]
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    #[derive(Clone)]
    pub struct KernelBuilder {
        id: usize,
        desc: Arc<KernelDesc>,
        spec_consts: Vec<ScalarElem>,
        threads: [u32; 3],
    }

    impl KernelBuilder {
        pub fn from_bytes(bytes: &'static [u8]) -> Result<Self> {
            let desc: Arc<KernelDesc> = Arc::new(bincode2::deserialize(bytes)?);
            let mut threads = [1, 1, 1];
            threads[..desc.threads.len()].copy_from_slice(&desc.threads);
            Ok(Self {
                id: bytes.as_ptr() as usize,
                desc,
                spec_consts: Vec::new(),
                threads,
            })
        }
        pub fn specialize(mut self, spec_consts: &[ScalarElem]) -> Result<Self> {
            assert_eq!(spec_consts.len(), self.desc.spec_descs.len());
            for (spec_const, spec_desc) in
                spec_consts.iter().copied().zip(self.desc.spec_descs.iter())
            {
                assert_eq!(spec_const.scalar_type(), spec_desc.scalar_type);
                if let Some(dim) = spec_desc.thread_dim {
                    if let ScalarElem::U32(value) = spec_const {
                        if value == 0 {
                            bail!("threads.{} cannot be zero!", ["x", "y", "z"][dim],);
                        }
                        self.threads[dim] = value;
                    } else {
                        unreachable!()
                    }
                }
            }
            self.spec_consts.clear();
            self.spec_consts.extend_from_slice(spec_consts);
            Ok(self)
        }
        pub fn features(&self) -> Features {
            self.desc.features
        }
        pub fn hash(&self) -> u64 {
            self.desc.hash
        }
        pub fn safe(&self) -> bool {
            self.desc.safe
        }
        pub fn build(&self, device: Device) -> Result<Kernel> {
            match device.inner {
                DeviceInner::Host => {
                    bail!("Kernel `{}` expected device, found host!", self.desc.name);
                }
                #[cfg(feature = "device")]
                DeviceInner::Device(device) => {
                    let desc = &self.desc;
                    let name = &desc.name;
                    let features = desc.features;
                    let device_features = device.info().features();
                    if !device_features.contains(&features) {
                        bail!("Kernel {name} requires {features:?}, {device:?} has {device_features:?}!");
                    }
                    let spec_bytes = if !self.desc.spec_descs.is_empty() {
                        if self.spec_consts.is_empty() {
                            bail!("Kernel `{name}` must be specialized!");
                        }
                        self.spec_consts
                            .iter()
                            .flat_map(|x| x.as_bytes())
                            .copied()
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let key = KernelKey {
                        id: self.id,
                        spec_bytes,
                    };
                    let inner = if !desc.spec_descs.is_empty() {
                        <<Engine as DeviceEngine>::Kernel>::cached(device.engine, key, || {
                            desc.specialize(
                                self.threads[..self.desc.threads.len()].to_vec(),
                                &self.spec_consts,
                            )
                            .map(Arc::new)
                        })?
                    } else {
                        <<Engine as DeviceEngine>::Kernel>::cached(device.engine, key, || {
                            Ok(desc.clone())
                        })?
                    };
                    Ok(Kernel {
                        inner,
                        groups: None,
                    })
                }
            }
        }
    }

    #[doc(hidden)]
    #[derive(Clone)]
    pub struct Kernel {
        #[cfg(feature = "device")]
        inner: Arc<<Engine as DeviceEngine>::Kernel>,
        #[cfg(feature = "device")]
        groups: Option<[u32; 3]>,
    }

    #[cfg(feature = "device")]
    fn global_threads_to_groups(global_threads: &[u32], threads: &[u32]) -> [u32; 3] {
        debug_assert_eq!(global_threads.len(), threads.len());
        let mut groups = [1; 3];
        for (gt, (g, t)) in global_threads
            .iter()
            .copied()
            .zip(groups.iter_mut().zip(threads.iter().copied()))
        {
            *g = gt / t + u32::from(gt % t != 0);
        }
        groups
    }

    impl Kernel {
        pub fn global_threads(
            #[cfg_attr(not(feature = "device"), allow(unused_mut))] mut self,
            global_threads: &[u32],
        ) -> Self {
            #[cfg(feature = "device")]
            {
                let desc = &self.inner.desc();
                let groups = global_threads_to_groups(global_threads, &desc.threads);
                self.groups.replace(groups);
                self
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = global_threads;
                unreachable!()
            }
        }
        pub fn groups(
            #[cfg_attr(not(feature = "device"), allow(unused_mut))] mut self,
            groups: &[u32],
        ) -> Self {
            #[cfg(feature = "device")]
            {
                debug_assert_eq!(groups.len(), self.inner.desc().threads.len());
                let mut new_groups = [1; 3];
                new_groups[..groups.len()].copy_from_slice(groups);
                self.groups.replace(new_groups);
                self
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = groups;
                unreachable!()
            }
        }
        pub unsafe fn dispatch(
            &self,
            slices: &[KernelSliceArg],
            push_consts: &[ScalarElem],
        ) -> Result<()> {
            #[cfg(feature = "device")]
            {
                let desc = &self.inner.desc();
                let kernel_name = &desc.name;
                let mut buffers = Vec::with_capacity(desc.slice_descs.len());
                let mut items: Option<usize> = None;
                for (slice, slice_desc) in slices.iter().zip(desc.slice_descs.iter()) {
                    debug_assert_eq!(slice.scalar_type(), slice_desc.scalar_type);
                    debug_assert!(!slice_desc.mutable || slice.mutable());
                    let slice_name = &slice_desc.name;
                    let buffer = if let Some(buffer) = slice.device_buffer() {
                        buffer
                    } else {
                        bail!("Kernel `{kernel_name}`.`{slice_name}` expected device, found host!");
                    };
                    if !Arc::ptr_eq(buffer.inner.engine(), self.inner.engine()) {
                        let device = RawDevice {
                            engine: self.inner.engine().clone(),
                        };
                        let buffer_device = buffer.device();
                        bail!(
                            "Kernel `{kernel_name}`.`{slice_name}`, expected `{device:?}`, found {buffer_device:?}!"
                        );
                    }
                    buffers.push(buffer.inner.clone());
                    if slice_desc.item {
                        items.replace(if let Some(items) = items {
                            items.min(slice.len())
                        } else {
                            slice.len()
                        });
                    }
                }
                let groups = if let Some(groups) = self.groups {
                    groups
                } else if let Some(items) = items {
                    if desc.threads.iter().skip(1).any(|t| *t > 1) {
                        bail!("Kernel `{kernel_name}` cannot infer global_threads if threads.y > 1 or threads.z > 1, threads = {threads:?}!", threads = desc.threads);
                    }
                    global_threads_to_groups(&[items as u32], &[desc.threads[0]])
                } else {
                    bail!("Kernel `{kernel_name}` global_threads or groups not provided!");
                };
                let mut push_bytes = Vec::with_capacity(desc.push_consts_range() as usize);
                for (push, push_desc) in push_consts.iter().zip(desc.push_descs.iter()) {
                    debug_assert_eq!(push.scalar_type(), push_desc.scalar_type);
                    push_bytes.extend_from_slice(push.as_bytes());
                }
                unsafe { self.inner.dispatch(groups, &buffers, push_bytes) }
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = (slices, push_consts);
                unreachable!()
            }
        }
        pub fn threads(&self) -> &[u32] {
            #[cfg(feature = "device")]
            {
                return self.inner.desc().threads.as_ref();
            }
            #[cfg(not(feature = "device"))]
            {
                unreachable!()
            }
        }
        pub fn features(&self) -> Features {
            #[cfg(feature = "device")]
            {
                return self.inner.desc().features;
            }
            #[cfg(not(feature = "device"))]
            {
                unreachable!()
            }
        }
    }

    #[doc(hidden)]
    pub enum KernelSliceArg<'a> {
        Slice(ScalarSlice<'a>),
        SliceMut(ScalarSliceMut<'a>),
    }

    #[cfg(feature = "device")]
    impl KernelSliceArg<'_> {
        fn scalar_type(&self) -> ScalarType {
            match self {
                Self::Slice(x) => x.scalar_type(),
                Self::SliceMut(x) => x.scalar_type(),
            }
        }
        fn mutable(&self) -> bool {
            match self {
                Self::Slice(_) => false,
                Self::SliceMut(_) => true,
            }
        }
        /*fn device(&self) -> Device {
            match self {
                Self::Slice(x) => x.device(),
                Self::SliceMut(x) => x.device(),
            }
        }*/
        fn device_buffer(&self) -> Option<&DeviceBuffer> {
            match self {
                Self::Slice(x) => x.device_buffer(),
                Self::SliceMut(x) => x.device_buffer_mut(),
            }
        }
        fn len(&self) -> usize {
            match self {
                Self::Slice(x) => x.len(),
                Self::SliceMut(x) => x.len(),
            }
        }
    }

    impl<'a, T: Scalar> From<Slice<'a, T>> for KernelSliceArg<'a> {
        fn from(slice: Slice<'a, T>) -> Self {
            Self::Slice(slice.into())
        }
    }

    impl<'a, T: Scalar> From<SliceMut<'a, T>> for KernelSliceArg<'a> {
        fn from(slice: SliceMut<'a, T>) -> Self {
            Self::SliceMut(slice.into())
        }
    }
}
