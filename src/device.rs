/*!

A [`Device`](crate::device::Device) is used to create [buffers](crate::buffer) and [kernels](crate::kernel).
[`Device::host()`](crate::device::Device::host) method creates the host, which is merely the lack of a device.

Note: Kernels can not be created for the host.

Creating a device and printing out useful info:
```no_run
# use krnl::{anyhow::Result, device::Device};
# fn main() -> Result<()> {
let device = Device::builder()
    .index(1)
    .build()?;
dbg!(device.info());
Ok(())
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

#[cfg(feature = "device")]
use crate::kernel::{KernelDesc, KernelKey, SliceDesc};
use anyhow::Result;
use serde::Deserialize;
#[cfg(feature = "device")]
use std::ops::Range;
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
    #[cfg(feature = "device")]
    use super::DeviceId;
    #[cfg(feature = "device")]
    pub(super) use crate::buffer::error::{DeviceBufferTooLarge, OutOfDeviceMemory};

    use std::fmt::{self, Debug, Display};

    /** Device is unavailable.

    - The "device" feature is not enabled.
    - Failed to load the Vulkan library.
    */
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    #[error("DeviceUnavailable")]
    pub struct DeviceUnavailable;

    /// The device index is greater than or equal to the number of devices.
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
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    pub struct DeviceLost(#[cfg(feature = "device")] pub(super) DeviceId);

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
        /// Creates a device.
        ///
        /// **errors**
        ///
        /// - [`DeviceUnavailable`](super::error::DeviceUnavailable)
        /// - [`DeviceIndexOutofRange`](super::error::DeviceIndexOutOfRange)
        /// - The device could not be created.
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
    fn id(&self) -> DeviceId;
    fn info(&self) -> &Arc<DeviceInfo>;
    fn wait(&self) -> Result<(), DeviceLost>;
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

/// See [`Device::host()`].
impl Default for Device {
    fn default() -> Self {
        Self::host()
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
    pub(crate) fn info(&self) -> &Arc<DeviceInfo> {
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
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct DeviceId {
    index: usize,
    handle: usize,
}

#[cfg(feature = "device")]
impl Debug for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Device({}@{:x})", self.index, self.handle)
    }
}

#[cfg(feature = "device")]
impl Debug for RawDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.engine.id().fmt(f)
    }
}

#[cfg(feature = "device")]
#[repr(transparent)]
#[derive(Clone)]
pub(crate) struct DeviceBuffer {
    inner: Arc<<Engine as DeviceEngine>::DeviceBuffer>,
}

#[cfg(feature = "device")]
fn cast_device_buffers(buffers: &[DeviceBuffer]) -> &[Arc<<Engine as DeviceEngine>::DeviceBuffer>] {
    // # Safety
    // Safe because transparent
    unsafe { std::slice::from_raw_parts(buffers.as_ptr() as _, buffers.len()) }
}

#[cfg(feature = "device")]
impl DeviceBuffer {
    const MAX_SIZE: usize = i32::MAX as usize;
    pub(crate) unsafe fn uninit(device: RawDevice, len: usize) -> Result<Self> {
        if len > Self::MAX_SIZE {
            return Err(DeviceBufferTooLarge { bytes: len }.into());
        }
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

#[cfg(feature = "device")]
#[derive(Clone)]
pub(crate) struct RawKernel {
    inner: Arc<<Engine as DeviceEngine>::Kernel>,
}

#[cfg(feature = "device")]
impl RawKernel {
    pub(crate) fn cached(
        device: RawDevice,
        key: KernelKey,
        desc_fn: impl FnOnce() -> Result<Arc<KernelDesc>>,
    ) -> Result<Self> {
        Ok(Self {
            inner: <Engine as DeviceEngine>::Kernel::cached(device.engine, key, desc_fn)?,
        })
    }
    pub(crate) unsafe fn dispatch(
        &self,
        groups: [u32; 3],
        buffers: &[DeviceBuffer],
        push_consts: Vec<u8>,
    ) -> Result<()> {
        unsafe {
            self.inner
                .dispatch(groups, cast_device_buffers(buffers), push_consts)
        }
    }
    pub(crate) fn device(&self) -> RawDevice {
        RawDevice {
            engine: self.inner.engine().clone(),
        }
    }
    pub(crate) fn desc(&self) -> &Arc<KernelDesc> {
        self.inner.desc()
    }
}
