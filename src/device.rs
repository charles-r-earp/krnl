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
    Engine, Compute, DeviceBuffer, DeviceBufferInner, HostBuffer, KernelCache,
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

/*
mod options {
    use super::*;

    pub(super) struct DeviceOptions {
        pub(super) optimal_capabilities: Vec<Capability>,
    }

    impl Default for DeviceOptions {
        fn default() -> Self {
            use spirv::Capability::*;
            Self {
                optimal_capabilities: vec![VulkanMemoryModel],
            }
        }
    }
}
use options::DeviceOptions;
*/

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
    fn new(index: usize) -> Result<Self> {
        Ok(Self {
            engine: Arc::new(Engine::new(index)?)
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
    pub fn new(index: usize) -> Result<Self, anyhow::Error> {
        #[cfg(test)]
        {
            #[cfg(test)] {
                use once_cell::sync::OnceCell;
                static DEVICE: OnceCell<Device> = OnceCell::new();
                if index == 0 {
                    return DEVICE
                        .get_or_try_init(|| Device::new_impl(index))
                        .map(|x| x.clone());
                }
            }
        }
        Self::new_impl(index)
    }
    #[cfg_attr(not(feature = "device"), allow(unused_variables))]
    fn new_impl(index: usize) -> Result<Self, anyhow::Error> {
        #[cfg(feature = "device")]
        {
            return Ok(Self {
                inner: DeviceInner::Device(VulkanDevice::new(index)?),
            });
        }
        #[cfg(not(feature = "device"))]
        {
            Err(DeviceUnavailable::new().into())
        }
    }
    pub(crate) fn kind(&self) -> DeviceKind {
        self.inner.kind()
    }
    pub(crate) fn is_host(&self) -> bool {
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
        #[cfg(feature = "device")] {
            self.inner.device().map(|x| x.features())
        }
        #[cfg(not(feature = "device"))] {
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
    shader_float64: bool,
}

impl Features {
    pub fn shader_int8(&self) -> bool {
        self.shader_int8
    }
    pub fn set_shader_int8(&mut self, shader_int8: bool) -> &mut Self {
        self.shader_int8 = shader_int8;
        self
    }
    pub fn shader_int16(&self) -> bool {
        self.shader_int16
    }
    pub fn set_shader_int16(&mut self, shader_int16: bool) -> &mut Self {
        self.shader_int16 = shader_int16;
        self
    }
    pub fn shader_int64(&self) -> bool {
        self.shader_int64
    }
    pub fn set_shader_int64(&mut self, shader_int64: bool) -> &mut Self {
        self.shader_int64 = shader_int64;
        self
    }
    pub fn shader_float64(&self) -> bool {
        self.shader_float64
    }
    pub fn set_shader_float64(&mut self, shader_float64: bool) -> &mut Self {
        self.shader_float64 = shader_float64;
        self
    }
    pub fn contains(&self, other: &Features) -> bool {
        (self.shader_int8 || !other.shader_int8)
        && (self.shader_int16 || !other.shader_int16)
        && (self.shader_int64 || !other.shader_int64)
        && (self.shader_float64 || !other.shader_float64)
    }
    pub fn union(mut self, other: &Features) -> Self {
        self.shader_int8 |= other.shader_int8;
        self.shader_int16 |= other.shader_int16;
        self.shader_int64 |= other.shader_int64;
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
