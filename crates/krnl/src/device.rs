use anyhow::Result;
use std::{
    collections::HashMap,
    fmt::{self, Debug},
    mem::{forget, size_of},
    ops::RangeBounds,
    sync::Arc,
    time::Duration,
};

#[cfg(feature = "device")]
mod vulkan_engine;
#[cfg(feature = "device")]
use vulkan_engine::Engine;

mod error {
    use std::fmt::{self, Debug, Display};

    #[derive(Clone, Copy, Debug, thiserror::Error)]
    #[error("DeviceUnavailable")]
    pub(super) struct DeviceUnavailable;

    #[cfg(feature = "device")]
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    pub(super) struct DeviceIndexOutOfRange {
        pub(super) index: usize,
        pub(super) devices: usize,
    }

    #[cfg(feature = "device")]
    impl Display for DeviceIndexOutOfRange {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(self, f)
        }
    }
    #[cfg(feature = "device")]
    pub(super) struct DeviceNotSupported;

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

pub mod builder {
    use super::*;

    pub struct DeviceBuilder {
        #[cfg(feature = "device")]
        pub(super) options: DeviceOptions,
    }

    impl DeviceBuilder {
        pub fn index(self, index: usize) -> Self {
            #[cfg(feature = "device")]
            {
                let mut this = self;
                this.options.index = index;
                this
            }
            #[cfg(not(feature = "device"))]
            {
                self
            }
        }
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
    fn engine(&self) -> &Arc<Self::Engine>;
    unsafe fn uninit(engine: Arc<Self::Engine>, len: usize) -> Result<Arc<Self>>;
    fn upload(engine: Arc<Self::Engine>, data: &[u8]) -> Result<Arc<Self>>;
    fn download(&self, data: &mut [u8]) -> Result<(), DeviceLost>;
    fn len(&self) -> usize;
    fn slice(self: &Arc<Self>, bounds: impl RangeBounds<usize>) -> Option<Arc<Self>>;
}

#[derive(Clone, Eq, PartialEq)]
pub struct Device {
    inner: DeviceInner,
}

impl Device {
    pub const fn host() -> Self {
        Self {
            inner: DeviceInner::Host,
        }
    }
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
    pub fn is_host(&self) -> bool {
        self.inner.is_host()
    }
    pub fn is_device(&self) -> bool {
        self.inner.is_device()
    }
    pub(crate) fn inner(&self) -> &DeviceInner {
        &self.inner
    }
    pub fn info(&self) -> Option<&Arc<DeviceInfo>> {
        match self.inner() {
            DeviceInner::Host => None,
            #[cfg(feature = "device")]
            DeviceInner::Device(raw) => Some(raw.info()),
        }
    }
    pub fn wait(&self) -> Result<(), DeviceLost> {
        match self.inner() {
            DeviceInner::Host => Ok(()),
            #[cfg(feature = "device")]
            DeviceInner::Device(raw) => raw.wait(),
        }
    }
}

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

#[derive(Clone, Eq, PartialEq)]
pub(crate) enum DeviceInner {
    Host,
    #[cfg(feature = "device")]
    Device(RawDevice),
}

impl DeviceInner {
    pub(crate) fn is_host(&self) -> bool {
        if let Self::Host = self {
            true
        } else {
            false
        }
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
#[derive(Clone)]
pub(crate) struct DeviceBuffer {
    inner: Arc<<Engine as DeviceEngine>::DeviceBuffer>,
}

#[cfg(feature = "device")]
impl DeviceBuffer {
    pub(crate) unsafe fn uninit(device: RawDevice, len: usize) -> Result<Self> {
        let inner = unsafe { <Engine as DeviceEngine>::DeviceBuffer::uninit(device.engine, len)? };
        Ok(Self { inner })
    }
    pub(crate) fn upload(device: RawDevice, data: &[u8]) -> Result<Self> {
        let inner = <Engine as DeviceEngine>::DeviceBuffer::upload(device.engine, data)?;
        Ok(Self { inner })
    }
    pub(crate) fn download(&self, data: &mut [u8]) -> Result<(), DeviceLost> {
        self.inner.download(data)
    }
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
    pub(crate) fn device(&self) -> RawDevice {
        RawDevice {
            engine: self.inner.engine().clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    shader_int8: bool,
    shader_int16: bool,
    shader_int64: bool,
    shader_float16: bool,
    shader_float64: bool,
}

impl Features {
    pub const fn empty() -> Self {
        Self {
            shader_int8: false,
            shader_int16: false,
            shader_int64: false,
            shader_float16: false,
            shader_float64: false,
        }
    }
    pub fn shader_int8(&self) -> bool {
        self.shader_int8
    }
    pub const fn with_shader_int8(mut self, shader_int8: bool) -> Self {
        self.shader_int8 = shader_int8;
        self
    }
    pub fn shader_int16(&self) -> bool {
        self.shader_int16
    }
    pub const fn with_shader_int16(mut self, shader_int16: bool) -> Self {
        self.shader_int16 = shader_int16;
        self
    }
    pub fn shader_int64(&self) -> bool {
        self.shader_int64
    }
    pub const fn with_shader_int64(mut self, shader_int64: bool) -> Self {
        self.shader_int64 = shader_int64;
        self
    }
    pub fn shader_float16(&self) -> bool {
        self.shader_float16
    }
    pub const fn with_shader_float16(mut self, shader_float16: bool) -> Self {
        self.shader_float16 = shader_float16;
        self
    }
    pub fn shader_float64(&self) -> bool {
        self.shader_float64
    }
    pub const fn with_shader_float64(mut self, shader_float64: bool) -> Self {
        self.shader_float64 = shader_float64;
        self
    }
    pub fn contains(&self, other: &Features) -> bool {
        (self.shader_int8 || !other.shader_int8)
            && (self.shader_int16 || !other.shader_int16)
            && (self.shader_int64 || !other.shader_int64)
            && (self.shader_float16 || !other.shader_float16)
            && (self.shader_float64 || !other.shader_float64)
    }
    pub fn union(mut self, other: &Features) -> Self {
        self.shader_int8 |= other.shader_int8;
        self.shader_int16 |= other.shader_int16;
        self.shader_int64 |= other.shader_int64;
        self.shader_float16 |= other.shader_float16;
        self.shader_float64 |= other.shader_float64;
        self
    }
}

#[derive(Debug)]
pub struct DeviceInfo {
    index: usize,
    name: String,
    compute_queues: usize,
    transfer_queues: usize,
    features: Features,
}

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
}
