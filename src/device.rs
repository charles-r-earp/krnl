use anyhow::Result;
use spirv::Capability;
use std::{
    fmt::{self, Debug},
    future::Future,
    sync::Arc,
};

#[cfg(feature = "device")]
pub(crate) mod engine;
#[cfg(feature = "device")]
pub(crate) use engine::{
    Compute, DeviceBuffer, DeviceBufferInner, Engine, HostBuffer, KernelCache,
};

pub mod error {

    /// The "device" feature is not enabled.
    #[derive(Debug, thiserror::Error)]
    #[error("DeviceUnavailable")]
    pub struct DeviceUnavailable {}

    impl DeviceUnavailable {
        pub(crate) fn new() -> Self {
            Self {}
        }
    }
}
use error::*;

pub(crate) mod future {
    use super::{Device, Result};
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll},
    };

    #[cfg(feature = "device")]
    pub(crate) use super::engine::HostBufferFuture;

    #[cfg(feature = "device")]
    use super::engine::SyncFuture as SyncFutureInner;

    pub(super) struct SyncFuture {
        #[cfg(feature = "device")]
        inner: Option<SyncFutureInner>,
    }

    impl SyncFuture {
        pub(super) fn new(device: &Device) -> Result<Self> {
            #[cfg(feature = "device")]
            {
                if let Some(device) = device.as_device() {
                    Ok(Self {
                        inner: Some(device.sync()?),
                    })
                } else {
                    Ok(Self { inner: None })
                }
            }
            #[cfg(not(feature = "device"))]
            {
                Ok(Self {})
            }
        }
    }

    impl Future for SyncFuture {
        type Output = Result<()>;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            #[cfg(feature = "device")]
            {
                if self.inner.is_some() {
                    let inner = unsafe {
                        Pin::map_unchecked_mut(self, |this| this.inner.as_mut().unwrap())
                    };
                    return Future::poll(inner, cx);
                }
            }
            Poll::Ready(Ok(()))
        }
    }
}
use future::SyncFuture;

pub mod builder {
    use super::*;
    use anyhow::bail;
    use dry::macro_wrap;
    use std::{env::var, str::FromStr};

    fn get_env_device_indices() -> Result<Vec<usize>> {
        let mut output = Vec::new();
        if let Ok(indices) = var("KRNL_DEVICE") {
            let indices = indices.trim().split(',');
            let (lower, upper) = indices.size_hint();
            output.reserve(upper.unwrap_or(lower));
            for index in indices {
                output.push(usize::from_str(index.trim())?);
            }
        }
        Ok(output)
    }

    fn get_env_device_features() -> Result<Option<Features>> {
        if let Ok(features) = var("KRNL_DEVICE_FEATURES") {
            let features = features.trim();
            let mut output = Features::default();
            if !features.is_empty() {
                let features = features.split(',');
                for feature in features {
                    let feature = feature.trim();
                    macro_wrap!(match feature {
                        macro_for!($FEATURE in [shader_int8, shader_int16, shader_int64, shader_float16, shader_float64] {
                            stringify!($FEATURE) => {
                                output.$FEATURE = true;
                            }
                        })
                        _ => {
                            bail!("KRNL_DEVICE_FEATURES unexpected feature `{feature}`");
                        }
                    });
                }
            }
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    pub struct DeviceBuilder {
        indices: Vec<usize>,
        features: Option<Features>,
        max_vulkan_version: Option<(u32, u32)>,
    }

    impl DeviceBuilder {
        pub(super) fn new() -> Self {
            Self {
                indices: Vec::new(),
                features: None,
                max_vulkan_version: None,
            }
        }
        pub fn index(mut self, index: usize) -> Self {
            self.indices.push(index);
            self
        }
        pub fn indices(mut self, indices: impl IntoIterator<Item = usize>) -> Self {
            self.indices.extend(indices);
            self
        }
        pub fn features(mut self, features: Features) -> Self {
            self.features.replace(features);
            self
        }
        pub fn max_vulkan_version(mut self, max_vulkan_version: (u32, u32)) -> Self {
            self.max_vulkan_version.replace(max_vulkan_version);
            self
        }
        pub fn build(self) -> Result<Device> {
            #[cfg(feature = "device")]
            {
                use std::str::FromStr;
                let mut indices = self.indices;
                if indices.is_empty() {
                    indices = get_env_device_indices()?;
                }
                let index = indices.first().copied().unwrap_or_default();
                let mut features = self.features;
                if features.is_none() {
                    features = get_env_device_features()?;
                }
                Ok(Device {
                    inner: DeviceInner::Device(VulkanDevice::new(
                        index,
                        features,
                        self.max_vulkan_version,
                    )?),
                })
            }
            #[cfg(not(feature = "device"))]
            {
                Err(DeviceUnavailable::new().into())
            }
        }
        pub fn build_iter(self) -> Result<impl Iterator<Item = Device>> {
            todo!();
            Ok([].into_iter())
        }
    }
}
use builder::DeviceBuilder;

#[cfg(feature = "device")]
#[derive(Clone, derive_more::Deref)]
pub(crate) struct VulkanDevice {
    #[deref]
    engine: Arc<Engine>,
}
#[cfg(feature = "device")]
pub(crate) use VulkanDevice as DeviceBase;

#[cfg(feature = "device")]
impl VulkanDevice {
    fn new(
        index: usize,
        features: Option<Features>,
        max_api_version: Option<(u32, u32)>,
    ) -> Result<Self> {
        Ok(Self {
            engine: Arc::new(Engine::new(index, features, max_api_version)?),
        })
    }
}

#[cfg(feature = "device")]
impl PartialEq for VulkanDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.engine, &other.engine)
    }
}

#[cfg(feature = "device")]
impl Eq for VulkanDevice {}

#[cfg(feature = "device")]
impl Debug for VulkanDevice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Device({})", self.index())
    }
}

#[derive(Clone, derive_more::IsVariant, derive_more::Unwrap, Eq, PartialEq)]
pub(crate) enum DeviceInner {
    Host,
    #[cfg(feature = "device")]
    Device(VulkanDevice),
}

impl DeviceInner {
    #[cfg(feature = "device")]
    pub(crate) fn device(&self) -> Option<&DeviceBase> {
        if let Self::Device(device) = self {
            Some(device)
        } else {
            None
        }
    }
    pub(crate) fn kind(&self) -> DeviceKind {
        match self {
            Self::Host => DeviceKind::Host,
            #[cfg(feature = "device")]
            Self::Device(_) => DeviceKind::Device,
        }
    }
}

pub(crate) enum DeviceKind {
    Host,
    #[cfg(feature = "device")]
    Device,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Device {
    pub(crate) inner: DeviceInner,
}

impl Device {
    pub fn host() -> Self {
        Self {
            inner: DeviceInner::Host,
        }
    }
    pub fn builder() -> DeviceBuilder {
        DeviceBuilder::new()
    }
    pub fn new(index: usize) -> Result<Self> {
        Self::builder().index(index).build()
    }
    pub(crate) fn kind(&self) -> DeviceKind {
        self.inner.kind()
    }
    pub fn is_host(&self) -> bool {
        self.inner.is_host()
    }
    pub(crate) fn is_device(&self) -> bool {
        !self.is_host()
    }
    #[cfg(feature = "device")]
    pub(crate) fn as_device(&self) -> Option<&DeviceBase> {
        match &self.inner {
            DeviceInner::Host => None,
            DeviceInner::Device(device) => Some(device),
        }
    }
    pub fn sync(&self) -> Result<impl Future<Output = Result<()>>> {
        SyncFuture::new(self)
    }
    pub fn features(&self) -> Option<&Features> {
        #[cfg(feature = "device")]
        {
            self.inner.device().map(|x| x.features())
        }
        #[cfg(not(feature = "device"))]
        {
            None
        }
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.inner {
            #[cfg(feature = "device")]
            DeviceInner::Device(device) => {
                write!(f, "Device({})", device.index())
            }
            DeviceInner::Host => write!(f, "Host"),
        }
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    shader_int8: bool,
    shader_int16: bool,
    shader_int64: bool,
    shader_float16: bool,
    shader_float64: bool,
}

impl Features {
    pub const fn new() -> Self {
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

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use crate::future::BlockableFuture;

    #[cfg(feature = "device")]
    #[test]
    fn device_new() -> Result<()> {
        let device = Device::new(0)?;
        Ok(())
    }

    #[test]
    fn sync_host() -> Result<()> {
        Device::host().sync()?.block()
    }

    #[cfg(feature = "device")]
    #[test]
    fn sync_device() -> Result<()> {
        Device::new(0)?.sync()?.block()
    }
}
