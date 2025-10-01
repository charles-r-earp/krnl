use crate::Result;
#[cfg(feature = "device")]
use crate::kernel::{KernelCreateInfo, KernelDesc, KernelKey};
use std::sync::Arc;

#[cfg(all(feature = "device", not(target_family = "wasm")))]
pub(super) mod vulkan;
#[cfg(all(feature = "device", not(target_family = "wasm")))]
pub(super) use vulkan as backend_impl;

#[cfg(all(feature = "device", target_family = "wasm"))]
pub(super) mod web;
#[cfg(all(feature = "device", target_family = "wasm"))]
pub(super) use web as backend_impl;

pub enum DeviceSpecifier {
    Index(usize),
}

pub(super) trait Backend {
    type Device: Device<Backend = Self>;
    type Event: Event<Device = Self::Device>;
    type Buffer: Buffer<Device = Self::Device>;
    type Slice: Slice<Device = Self::Device>;
    type Kernel: Kernel<Device = Self::Device, Slice = Self::Slice>;
    fn create() -> Result<Arc<Self>>;
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Properties {
    min_subgroup_threads: u32,
    max_subgroup_threads: u32,
}

impl Properties {
    pub(super) fn min_subgroup_threads(&self) -> usize {
        self.min_subgroup_threads as usize
    }
    pub(super) fn max_subgroup_threads(&self) -> usize {
        self.max_subgroup_threads as usize
    }
}

pub(super) trait Device {
    type Backend;
    type Event: Event<Device = Self>;
    #[cfg(not(target_family = "wasm"))]
    fn create(backend: Arc<Self::Backend>, specifier: DeviceSpecifier) -> Result<Arc<Self>>;
    #[cfg(target_family = "wasm")]
    async fn create_async(backend: Arc<Self::Backend>) -> Result<Arc<Self>>;
    fn event(self: &Arc<Self>) -> Arc<Self::Event>;
    fn properties(&self) -> &Properties;
}

pub(super) trait DeviceOwned {
    type Device: Device;
    fn device(&self) -> &Arc<Self::Device>;
}

pub(super) trait Event: DeviceOwned {
    #[cfg(not(target_family = "wasm"))]
    fn wait(&self) -> Result<()>;
    #[cfg(target_family = "wasm")]
    async fn wait_async(&self) -> Result<()>;
}

pub(super) trait Buffer: DeviceOwned {
    type Slice: Slice;
    unsafe fn uninit(device: Arc<Self::Device>, len: usize) -> Result<Arc<Self>>;
    fn len(&self) -> usize;
    fn into_slice(self: Arc<Self>) -> Arc<Self::Slice>;
}

pub(super) trait Slice: DeviceOwned {
    fn len(&self) -> usize;
    fn range(&self) -> BufferRange;
    fn slice(self: Arc<Self>, range: BufferRange) -> Arc<Self>;
    unsafe fn upload(&self, bytes: &[u8]) -> Result<()>;
    #[cfg(not(target_family = "wasm"))]
    unsafe fn download(&self, bytes: &mut [u8]) -> Result<()>;
    #[cfg(target_family = "wasm")]
    async unsafe fn download_async(&self, bytes: &mut [u8]) -> Result<()>;
}

pub(super) trait Kernel: DeviceOwned {
    type Slice: Slice;
    fn get_or_create(
        device: Arc<Self::Device>,
        key: KernelKey,
        f: impl FnOnce() -> KernelCreateInfo,
    ) -> Result<Arc<Self>>;
    fn desc(&self) -> &KernelDesc;
    unsafe fn exec(
        self: &Arc<Self>,
        groups: u32,
        buffers: &[BufferBinding<Self::Slice>],
        push_constants: &[u8],
    ) -> Result<()>;
}

#[derive(Clone, Copy)]
pub(super) struct BufferRange {
    pub(super) start: usize,
    pub(super) end: usize,
}

impl BufferRange {
    pub(super) fn len(&self) -> usize {
        self.end - self.start
    }
    pub(super) fn is_aligned_to(&self, size: usize) -> bool {
        self.start % size == 0 && self.end % size == 0
    }
}

pub(super) struct BufferBinding<T> {
    pub(super) slice: Arc<T>,
    pub(super) mutable: bool,
}
