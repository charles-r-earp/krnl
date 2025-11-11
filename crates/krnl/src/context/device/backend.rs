#[cfg(feature = "device")]
use crate::kernel::{KernelCreateInfo, KernelDesc, KernelKey};
use crate::{Result, context::device::Features};
use core::ops::{Bound, RangeBounds};
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
    min_buffer_align: u32,
    max_buffer_size: u32,
    min_subgroup_threads: u32,
    max_subgroup_threads: u32,
}

impl Properties {
    pub(super) fn min_buffer_align(&self) -> usize {
        self.min_buffer_align as usize
    }
    pub(super) fn max_buffer_size(&self) -> usize {
        self.max_buffer_size as usize
    }
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
    fn features(&self) -> Features;
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

impl RangeBounds<usize> for BufferRange {
    fn start_bound(&self) -> Bound<&usize> {
        Bound::Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&usize> {
        Bound::Excluded(&self.end)
    }
}

impl BufferRange {
    pub(super) fn len(&self) -> usize {
        self.end - self.start
    }
    pub(super) fn is_aligned_to(&self, size: usize) -> bool {
        self.start % size == 0 && self.end % size == 0
    }
    pub(super) fn aligned_offset(&self, align: usize) -> (Self, usize) {
        let rem = self.start % align;
        let aligned = self.start - rem;
        (
            Self {
                start: aligned,
                end: self.end,
            },
            rem,
        )
    }
    pub(super) fn slice(self, bounds: impl RangeBounds<usize>) -> Self {
        let start = match bounds.start_bound() {
            Bound::Unbounded => self.start,
            Bound::Included(x) => self.start + x,
            Bound::Excluded(x) => self.start + x + 1,
        };
        let end = match bounds.end_bound() {
            Bound::Excluded(x) => self.start + x,
            Bound::Included(x) => self.start + x + 1,
            Bound::Unbounded => self.end,
        };
        assert!(start < self.end);
        assert!(start < end);
        assert!(end <= self.end);
        Self { start, end }
    }
}

pub(super) struct BufferBinding<T> {
    slice: Arc<T>,
    mutable: bool,
    offset: u32,
}

impl<T: Slice> BufferBinding<T> {
    pub(super) fn new(slice: Arc<T>, mutable: bool, elem_size: usize) -> Self {
        let min_buffer_align = slice.device().properties().min_buffer_align();
        let offset = (slice.range().aligned_offset(min_buffer_align).1 / elem_size) as u32;
        Self {
            slice,
            mutable,
            offset,
        }
    }
    pub(super) fn offset(&self) -> u32 {
        self.offset
    }
}
