#![allow(unused)]
use crate::{device::Features, kernel::__private::KernelDesc, scalar::Scalar};
use anyhow::{bail, format_err, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use once_cell::sync::OnceCell;
use parking_lot::{Mutex, RwLock};
use spirv::Capability;
use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{HashMap, VecDeque},
    fmt::{self, Debug},
    future::Future,
    hash::{Hash, Hasher},
    iter::{once, Peekable},
    mem::ManuallyDrop,
    ops::Deref,
    ops::Range,
    pin::Pin,
    ptr::NonNull,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};
use vulkano::{
    buffer::{
        cpu_access::ReadLock,
        cpu_pool::CpuBufferPoolChunk,
        device_local::DeviceLocalBuffer,
        sys::{Buffer, BufferCreateInfo, RawBuffer},
        BufferAccess, BufferError, BufferInner, BufferSlice, BufferUsage, CpuAccessibleBuffer,
        CpuBufferPool,
    },
    command_buffer::{
        pool::{CommandPool, CommandPoolAlloc, CommandPoolCreateInfo},
        sys::{
            CommandBufferBeginInfo,
            //    UnsafeCommandBufferBuilderPipelineBarrier,
            UnsafeCommandBuffer,
            UnsafeCommandBufferBuilder,
        },
        CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
    },
    descriptor_set::{
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        pool::{DescriptorPool, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo},
        sys::UnsafeDescriptorSet,
        WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Features as VulkanoFeatures,
        Queue, QueueCreateInfo,
    },
    instance::{
        Instance, InstanceCreateInfo, InstanceCreationError, InstanceExtensions,
        Version as InstanceVersion,
    },
    library::VulkanLibrary,
    memory::{
        allocator::MemoryAlloc, DeviceMemory, DeviceMemoryError, MappedDeviceMemory,
        MemoryAllocateInfo, MemoryType,
    },
    pipeline::{
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        ComputePipeline, PipelineBindPoint, PipelineLayout,
    },
    shader::{
        spirv::{Capability as VulkanoCapability, ExecutionModel},
        DescriptorRequirements, EntryPointInfo, ShaderExecution, ShaderInterface, ShaderModule,
        ShaderStages, SpecializationConstantRequirements, SpecializationConstants,
        SpecializationMapEntry,
    },
    sync::{
        AccessFlags, BufferMemoryBarrier, DependencyInfo, Fence, FenceCreateInfo, PipelineStages,
        Semaphore,
    },
    DeviceSize, OomError, Version,
};

#[cfg(any(target_os = "ios", target_os = "macos"))]
mod molten {
    use ash::vk::Instance;
    use std::os::raw::{c_char, c_void};
    use vulkano::instance::loader::Loader;

    pub(super) struct AshMoltenLoader;

    unsafe impl Loader for AshMoltenLoader {
        fn get_instance_proc_addr(&self, instance: Instance, name: *const c_char) -> *const c_void {
            let entry = ash_molten::load();
            let ptr = unsafe { entry.get_instance_proc_addr(std::mem::transmute(instance), name) };
            if let Some(ptr) = ptr {
                unsafe { std::mem::transmute(ptr) }
            } else {
                std::ptr::null()
            }
        }
    }
}
#[cfg(any(target_os = "ios", target_os = "macos"))]
use molten::AshMoltenLoader;

fn create_instance(max_api_version: Option<InstanceVersion>) -> Result<Arc<Instance>> {
    let engine_name = Some("krnl".to_string());
    let engine_version = InstanceVersion::from_str(env!("CARGO_PKG_VERSION"))?;
    let mut library = VulkanLibrary::new();
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    {
        if library.is_err() {
            library = VulkanLibrary::with_loader(AshMoltenLoader)
        }
    };
    let library = library?;
    let instance_create_info = InstanceCreateInfo {
        engine_name: engine_name.clone(),
        engine_version,
        max_api_version,
        enumerate_portability: true,
        ..InstanceCreateInfo::default()
    };
    Ok(Instance::new(library, instance_create_info)?)
}

pub(crate) struct Engine {
    index: usize,
    features: Features,
    buffer_allocator: BufferAllocator,
    kernel_cache_map: KernelCacheMap,
    device: Arc<Device>,
    runner: Arc<Runner>,
}

impl Engine {
    pub(super) fn new(
        index: usize,
        features: Option<Features>,
        max_api_version: Option<(u32, u32)>,
    ) -> Result<Self> {
        let max_api_version =
            max_api_version.map(|(major, minor)| InstanceVersion::major_minor(major, minor));
        let instance = create_instance(max_api_version)?;
        let physical_devices = instance.enumerate_physical_devices()?;
        let physical_device_count = physical_devices.len();
        let physical_device = physical_devices.skip(index).next().ok_or_else(|| {
            format_err!(
                "Cannot create device at index {index}, only {physical_device_count} devices!",
            )
        })?;
        let vulkan_version = physical_device.api_version();
        if vulkan_version < InstanceVersion::major_minor(1, 1) {
            bail!("Device({index}) supports vulkan {vulkan_version:?}, >= 1.1 required!");
        }
        let supported_device_features = physical_device.supported_features();
        let mut optimal_device_features = VulkanoFeatures {
            shader_int8: true,
            shader_int16: true,
            shader_int64: true,
            shader_float64: true,
            vulkan_memory_model: true,
            ..VulkanoFeatures::none()
        };
        if let Some(features) = features.as_ref() {
            let supported_features = Features {
                shader_int8: supported_device_features.shader_int8,
                shader_int16: supported_device_features.shader_int16,
                shader_int64: supported_device_features.shader_int64,
                shader_float16: supported_device_features.shader_float16,
                shader_float64: supported_device_features.shader_float64,
            };
            if !supported_features.contains(features) {
                bail!(
                    "Device({index}) has {supported_features:?}, requested features {features:?}!"
                );
            }
            optimal_device_features.shader_int8 &= features.shader_int8;
            optimal_device_features.shader_int16 &= features.shader_int16;
            optimal_device_features.shader_int64 &= features.shader_int64;
            optimal_device_features.shader_float16 &= features.shader_float16;
            optimal_device_features.shader_float64 &= features.shader_float64;
        }
        let device_features = physical_device
            .supported_features()
            .intersection(&optimal_device_features);
        let features = Features {
            shader_int8: device_features.shader_int8,
            shader_int16: device_features.shader_int16,
            shader_int64: device_features.shader_int64,
            shader_float16: device_features.shader_float16,
            shader_float64: device_features.shader_float64,
        };
        let queue_family_properties = physical_device.queue_family_properties();
        let queue_family_index = queue_family_properties
            .iter()
            .map(|x| x.queue_flags)
            .position(|x| x.compute && !x.graphics)
            .or_else(|| {
                queue_family_properties
                    .iter()
                    .map(|x| x.queue_flags)
                    .position(|x| x.compute && !x.graphics)
            })
            .map(|x| x.try_into().unwrap())
            .ok_or_else(|| format_err!("Device {index} doesn't support compute!"))?;
        let device_extensions = DeviceExtensions::none();
        let queue_create_info = QueueCreateInfo {
            queue_family_index,
            queues: vec![1f32],
            ..QueueCreateInfo::default()
        };
        let device_create_info = DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: device_features,
            queue_create_infos: vec![queue_create_info],
            ..DeviceCreateInfo::default()
        };
        let (device, mut queues) = Device::new(physical_device, device_create_info)?;
        let queue = queues.next().unwrap();
        let buffer_allocator = BufferAllocator::new(device.clone())?;
        let kernel_cache_map = KernelCacheMap::new(device.clone());
        let runner = Runner::new(queue, index)?;
        Ok(Self {
            index,
            device,
            features,
            buffer_allocator,
            kernel_cache_map,
            runner,
        })
    }
    pub(crate) fn index(&self) -> usize {
        self.index
    }
    /*pub(crate) fn vulkan_version(&self) -> Version {
        let version = self.device.instance().api_version();
        Version {
            major: version.major,
            minor: version.minor,
            patch: version.patch,
        }
    }*/
    /*pub(crate) fn supports_vulkan_version(&self, vulkan_version: Version) -> bool {
        vulkan_version <= self.vulkan_version()
    }
    pub(crate) fn enabled_capabilities(&self) -> &[Capability] {
        &self.enabled_capabilities
    }
    pub(crate) fn capability_enabled(&self, capability: Capability) -> bool {
        self.enabled_capabilities
            .iter()
            .copied()
            .any(|x| x == capability)
    }
    pub(crate) fn enabled_extensions(&self) -> &[&'static str] {
        &self.enabled_extensions
    }
    pub(crate) fn extension_enabled(&self, extension: &str) -> bool {
        self.enabled_extensions
            .iter()
            .copied()
            .any(|x| x == extension)
    }*/
    pub(crate) fn features(&self) -> &Features {
        &self.features
    }
    fn send_op(&self, op: Op) -> Result<()> {
        use std::{
            error::Error,
            marker::{Send, Sync},
        };
        let result = self.runner.result.read().clone();
        if let Err(e) = result {
            return Err(anyhow::Error::msg(e));
        }
        self.runner.op_sender.send(op).unwrap();
        Ok(())
    }
    // # Safety
    // Uninitialized.
    pub(crate) unsafe fn alloc(&self, len: usize) -> Result<Option<Arc<DeviceBuffer>>> {
        if len == 0 {
            Ok(None)
        } else if len > u32::MAX as usize {
            anyhow::bail!(
                "Device buffer size {}B is too large, max is {}B!",
                len,
                u32::MAX
            );
        } else {
            let buffer = self.buffer_allocator.alloc_device(len as u32)?;
            Ok(Some(buffer))
        }
    }
    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<Option<Arc<DeviceBuffer>>> {
        let buffer = unsafe { self.alloc(bytes.len())? };
        if let Some(buffer) = buffer.as_ref() {
            let mut offset = 0;
            for bytes in bytes.chunks(64_000_000) {
                let mut src = self.buffer_allocator.alloc_host(bytes.len() as u32)?;
                Arc::get_mut(&mut src).unwrap().write_slice(bytes)?;
                let len = src.alloc.block.len();
                let upload = Upload {
                    src,
                    dst: buffer.inner.slice_offset_len(offset as u32, len),
                };
                self.send_op(Op::Upload(upload))?;
                offset += bytes.len();
            }
        }
        Ok(buffer)
    }
    pub(crate) fn download(
        &self,
        buffer: Arc<DeviceBuffer>,
        offset: usize,
        len: usize,
    ) -> Result<HostBufferFuture> {
        debug_assert_eq!(offset, 0);
        let src = buffer.inner.clone();
        let dst = self.buffer_allocator.alloc_host(len as u32)?;
        let download = Download {
            src,
            dst: dst.clone(),
        };
        self.send_op(Op::Download(download))?;
        Ok(HostBufferFuture {
            host_buffer: Some(dst),
            runner_result: self.runner.result.clone(),
        })
        /* TODO: break up host buffers into chunks?
        // large copies are expensive, could be done in parallel?
        // idea is to overlap host and device execution
        use crate::future::BlockableFuture;
        for _ in 0 .. 10 {
            let chunk_size = 64_000_000;
            let mut output = vec![0u8; buffer.len()];
            let mut fut: Option<(&mut [u8], HostBufferFuture)> = None;
            let mut offset = 0;
            for bytes in output.chunks_mut(chunk_size) {
                let mut dst = self.buffer_allocator.alloc_host(bytes.len() as u32)?;
                let download = Download {
                    src: buffer.inner.slice_offset_len(offset as u32, dst.len),
                    dst: dst.clone(),
                };
                offset += bytes.len();
                self.send_op(Op::Download(download))?;
                if let Some((bytes, fut)) = fut.take() {
                    bytes.copy_from_slice(fut.block()?.read()?);
                }
                fut.replace((bytes, HostBufferFuture {
                    host_buffer: Some(dst),
                    runner_result: self.runner.result.clone(),
                }));
            }
            let (bytes, fut) = fut.take().unwrap();
            bytes.copy_from_slice(fut.block()?.read()?);
        }

        let start = std::time::Instant::now();
        let chunk_size = 64_000_000;
        let mut output = vec![0u8; buffer.len()];
        let mut fut: Option<(&mut [u8], HostBufferFuture)> = None;
        let mut offset = 0;
        for bytes in output.chunks_mut(chunk_size) {
            let mut dst = self.buffer_allocator.alloc_host(bytes.len() as u32)?;
            let download = Download {
                src: buffer.inner.slice_offset_len(offset as u32, dst.len),
                dst: dst.clone(),
            };
            offset += bytes.len();
            self.send_op(Op::Download(download))?;
            if let Some((bytes, fut)) = fut.take() {
                bytes.copy_from_slice(fut.block()?.read()?);
                dbg!(start.elapsed());
            }
            fut.replace((bytes, HostBufferFuture {
                host_buffer: Some(dst),
                runner_result: self.runner.result.clone(),
            }));

        }
        let (bytes, fut) = fut.take().unwrap();
        bytes.copy_from_slice(fut.block()?.read()?);
        dbg!(start.elapsed());
        todo!()*/
    }
    pub(crate) fn kernel_cache(
        &self,
        spirv: &'static [u32],
        spec_consts: Vec<u32>,
        desc: KernelDesc,
    ) -> Result<Arc<KernelCache>> {
        self.kernel_cache_map.kernel(spirv, spec_consts, desc)
    }
    pub(crate) fn compute(&self, compute: Compute) -> Result<()> {
        self.send_op(Op::Compute(compute))?;
        Ok(())
    }
    pub(super) fn sync(&self) -> Result<SyncFuture> {
        let fut = SyncFuture::new(self.runner.result.clone());
        self.send_op(Op::SyncGuard(fut.guard()))?;
        Ok(fut)
    }
}

#[derive(Debug)]
pub(crate) struct HostBuffer {
    alloc: Arc<ChunkAlloc>,
    buffer: Arc<Buffer>,
    memory_alloc: MemoryAlloc,
    len: u32,
}

impl HostBuffer {
    fn new(alloc: Arc<ChunkAlloc>, len: u32) -> Result<Arc<Self>> {
        let chunk_id = Arc::as_ptr(&alloc.chunk);
        let device = alloc.chunk.memory_alloc.device().clone();
        let usage = BufferUsage {
            transfer_src: true,
            transfer_dst: true,
            ..BufferUsage::default()
        };
        let buffer = RawBuffer::new(
            device,
            BufferCreateInfo {
                size: alloc.block.len() as DeviceSize,
                usage,
                ..Default::default()
            },
        )?;
        let memory_alloc = unsafe {
            let mut memory_alloc = alloc.memory_alloc().alias().unwrap();
            memory_alloc.set_offset(alloc.block.start as DeviceSize);
            memory_alloc.set_size(alloc.block.len() as DeviceSize);
            memory_alloc
        };
        let memory_alloc2 = unsafe { memory_alloc.alias().unwrap() };
        let buffer = Arc::new(buffer.bind_memory(memory_alloc2).map_err(|(e, _, _)| e)?);
        Ok(Arc::new(Self {
            alloc,
            buffer,
            memory_alloc,
            len,
        }))
    }
    pub(crate) fn read(&self) -> Result<&[u8]> {
        let data = unsafe {
            self.memory_alloc
                .invalidate_range(0..self.memory_alloc.size())?;
            self.memory_alloc
                .mapped_slice()
                .unwrap()
                .get(..self.len as usize)
                .unwrap()
        };
        Ok(data)
    }
    fn write_slice(&mut self, slice: &[u8]) -> Result<()> {
        let data = unsafe {
            self.memory_alloc
                .mapped_slice_mut()
                .unwrap()
                .get_mut(..self.len as usize)
                .unwrap()
        };
        data.copy_from_slice(slice);
        unsafe {
            self.memory_alloc.flush_range(0..self.memory_alloc.size())?;
        }
        Ok(())
    }
    fn chunk_id(&self) -> usize {
        Arc::as_ptr(&self.alloc.chunk) as usize
    }
    fn start(&self) -> u32 {
        self.alloc.block.start
    }
    fn barrier_range(&self) -> Range<DeviceSize> {
        0..self.buffer.size()
    }
}

unsafe impl DeviceOwned for HostBuffer {
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

unsafe impl BufferAccess for HostBuffer {
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.buffer,
            offset: 0,
        }
    }
    fn size(&self) -> DeviceSize {
        self.buffer.size()
    }
}

#[derive(Debug)]
pub(crate) struct HostBufferFuture {
    host_buffer: Option<Arc<HostBuffer>>,
    runner_result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
}

impl Future for HostBufferFuture {
    type Output = Result<HostBuffer>;
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let host_buffer = self.host_buffer.take().unwrap();
        match Arc::try_unwrap(host_buffer) {
            Ok(host_buffer) => {
                let result = self.runner_result.read().clone();
                if let Err(e) = result {
                    Poll::Ready(Err(anyhow::Error::msg(e)))
                } else {
                    Poll::Ready(Ok(host_buffer))
                }
            }
            Err(host_buffer) => {
                self.host_buffer.replace(host_buffer);
                Poll::Pending
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DeviceBufferInner {
    buffer: Arc<Buffer>,
    chunk: Arc<Chunk>,
    usage: BufferUsage,
    start: u32,
    offset: u32,
    len: u32,
}

impl DeviceBufferInner {
    fn chunk_id(&self) -> usize {
        Arc::as_ptr(&self.chunk) as usize
    }
    fn slice_offset_len(&self, offset: u32, len: u32) -> Arc<Self> {
        let offset = self.offset + offset;
        debug_assert!(offset < self.offset + self.len);
        debug_assert!(len <= self.len);
        Arc::new(Self {
            offset,
            len,
            ..self.clone()
        })
    }
    fn barrier_range(&self) -> Range<DeviceSize> {
        let start = self.offset as _;
        let end = (self.offset + self.len) as DeviceSize;
        start..end
    }
}

unsafe impl DeviceOwned for DeviceBufferInner {
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

unsafe impl BufferAccess for DeviceBufferInner {
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.buffer,
            offset: self.offset as _,
        }
    }
    fn size(&self) -> DeviceSize {
        self.len as _
    }
    fn usage(&self) -> &BufferUsage {
        &self.usage
    }
}

#[derive(Debug)]
pub(crate) struct DeviceBuffer {
    inner: Arc<DeviceBufferInner>,
    alloc: Arc<ChunkAlloc>,
}

impl DeviceBuffer {
    fn new(device: Arc<Device>, alloc: Arc<ChunkAlloc>) -> Result<Arc<Self>> {
        let usage = BufferUsage {
            transfer_src: true,
            transfer_dst: true,
            storage_buffer: true,
            ..BufferUsage::default()
        };
        let start = alloc.block.start;
        let len = alloc.block.len();
        let buffer = RawBuffer::new(
            device,
            BufferCreateInfo {
                size: len as DeviceSize,
                usage,
                ..Default::default()
            },
        )?;
        let memory_alloc = unsafe {
            let mut memory_alloc = alloc.memory_alloc().alias().unwrap();
            memory_alloc.set_offset(start as DeviceSize);
            memory_alloc.set_size(len as DeviceSize);
            memory_alloc
        };
        let buffer = Arc::new(buffer.bind_memory(memory_alloc).map_err(|(e, _, _)| e)?);
        let inner = Arc::new(DeviceBufferInner {
            buffer,
            chunk: alloc.chunk.clone(),
            usage,
            start,
            offset: 0,
            len,
        });
        Ok(Arc::new(Self { alloc, inner }))
    }
    pub(crate) fn inner(&self) -> &Arc<DeviceBufferInner> {
        &self.inner
    }
}

#[derive(Debug, Clone, Copy)]
struct Block {
    start: u32,
    end: u32,
}

impl Block {
    fn len(&self) -> u32 {
        self.end - self.start
    }
}

#[derive(Debug)]
struct ChunkAlloc {
    chunk: Arc<Chunk>,
    block: Block,
}

impl ChunkAlloc {
    fn memory_alloc(&self) -> &MemoryAlloc {
        &self.chunk.memory_alloc
    }
}

impl Drop for ChunkAlloc {
    fn drop(&mut self) {
        let mut blocks = self.chunk.blocks.lock();
        if let Some(i) = blocks.iter().position(|x| x.start == self.block.start) {
            blocks.remove(i);
        }
    }
}

/*
#[derive(Debug)]
struct HostMemory {
    buffer: Arc<RawBuffer>,
    memory: MappedDeviceMemory,
    usage: BufferUsage,
}

trait ChunkMemory: Sized {
    fn oom_error() -> OomError;
    fn from_device_memory(device_memory: DeviceMemory) -> Result<Self>;
}

impl ChunkMemory for DeviceMemory {
    fn oom_error() -> OomError {
        OomError::OutOfDeviceMemory
    }
    fn from_device_memory(device_memory: Self) -> Result<Self> {
        Ok(device_memory)
    }
}

impl ChunkMemory for HostMemory {
    fn oom_error() -> OomError {
        OomError::OutOfHostMemory
    }
    fn from_device_memory(device_memory: DeviceMemory) -> Result<Self> {
        let usage = BufferUsage::transfer_src() | BufferUsage::transfer_dst();
        let buffer = RawBuffer::new(
            device_memory.device().clone(),
            BufferCreateInfo {
                size: device_memory.allocation_size(),
                usage,
                ..Default::default()
            },
        )?;
        unsafe { buffer.bind_memory(&device_memory, 0)? };
        let memory = MappedDeviceMemory::new(device_memory, 0..buffer.size())?;
        Ok(Self {
            buffer,
            memory,
            usage,
        })
    }
}*/

const CHUNK_ALIGN: u32 = 256;
const CHUNK_SIZE_MULTIPLE: usize = 256_000_000;

#[derive(Copy, Clone, Debug)]
enum MemoryKind {
    Host,
    Device,
}

#[derive(Debug)]
struct Chunk {
    kind: MemoryKind,
    memory_alloc: MemoryAlloc,
    len: usize,
    blocks: Mutex<Vec<Block>>,
}

impl Chunk {
    fn new(device: Arc<Device>, kind: MemoryKind, len: usize, ids: &[u32]) -> Result<Arc<Self>> {
        let len = CHUNK_SIZE_MULTIPLE * (1 + (len - 1) / CHUNK_SIZE_MULTIPLE);
        for id in ids {
            let result = DeviceMemory::allocate(
                device.clone(),
                MemoryAllocateInfo {
                    allocation_size: len as DeviceSize,
                    memory_type_index: *id,
                    ..Default::default()
                },
            );
            match result {
                Ok(device_memory) => {
                    let memory_alloc = MemoryAlloc::new(device_memory)?;
                    return Ok(Arc::new(Self {
                        kind,
                        memory_alloc,
                        len,
                        blocks: Mutex::default(),
                    }));
                }
                Err(DeviceMemoryError::OomError(e)) => continue,
                Err(e) => {
                    return Err(e.into());
                }
            }
        }
        match kind {
            MemoryKind::Host => Err(DeviceMemoryError::OomError(OomError::OutOfHostMemory).into()),
            MemoryKind::Device => {
                Err(DeviceMemoryError::OomError(OomError::OutOfDeviceMemory).into())
            }
        }
    }
    fn alloc(self: &Arc<Self>, len: u32) -> Option<Arc<ChunkAlloc>> {
        if len as usize > self.len {
            return None;
        }
        let block_len = CHUNK_ALIGN * (1 + (len - 1) / CHUNK_ALIGN);
        let mut blocks = self.blocks.lock();
        if !blocks.is_empty() {
            return None;
        }
        let mut start = match self.kind {
            MemoryKind::Host => 256,
            MemoryKind::Device => 0,
        };
        for (i, block) in blocks.iter().enumerate() {
            if start + len <= block.start {
                let block = Block {
                    start,
                    end: start + block_len,
                };
                blocks.insert(i, block);
                return Some(Arc::new(ChunkAlloc {
                    chunk: self.clone(),
                    block,
                }));
            } else {
                start = block.end;
            }
        }
        if (start + len) as usize <= self.len {
            let block = Block {
                start,
                end: start + block_len,
            };
            blocks.push(block);
            Some(Arc::new(ChunkAlloc {
                chunk: self.clone(),
                block,
            }))
        } else {
            None
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct HostMemoryTypeOrderKey {
    not_cached: bool,
    neg_size: i64,
}

#[derive(Debug)]
struct BufferAllocator {
    device: Arc<Device>,
    host_ids: Vec<u32>,
    device_ids: Vec<u32>,
    host_chunks: Vec<OnceCell<Arc<Chunk>>>,
    device_chunks: Vec<OnceCell<Arc<Chunk>>>,
}

impl BufferAllocator {
    fn new(device: Arc<Device>) -> Result<Self> {
        let physical_device = device.physical_device();
        let mut max_host_chunks = 0;
        let mut max_device_chunks = 0;
        let mut host_ids = Vec::new();
        let mut device_ids = Vec::new();
        let memory_properties = physical_device.memory_properties();
        for (id, memory_type) in memory_properties.memory_types.iter().enumerate() {
            let heap = &memory_properties.memory_heaps[memory_type.heap_index as usize];
            let id = id as u32;
            if memory_type.property_flags.host_visible {
                max_host_chunks += (heap.size / CHUNK_SIZE_MULTIPLE as u64) as usize;
                host_ids.push(id);
            }
            if memory_type.property_flags.host_visible {
                max_device_chunks += (heap.size / CHUNK_SIZE_MULTIPLE as u64) as usize;
                device_ids.push(id);
            }
        }
        // sort largest heap first
        host_ids.sort_by_key(|x| {
            let t = &memory_properties.memory_types[*x as usize];
            let heap = &memory_properties.memory_heaps[t.heap_index as usize];
            HostMemoryTypeOrderKey {
                not_cached: !t.property_flags.host_cached,
                neg_size: -(heap.size as i64),
            }
        });
        device_ids.sort_by_key(|x| {
            let t = &memory_properties.memory_types[*x as usize];
            let heap = &memory_properties.memory_heaps[t.heap_index as usize];
            -(heap.size as i64)
        });
        let host_chunks = (0..max_host_chunks)
            .into_iter()
            .map(|_| OnceCell::default())
            .collect();
        let device_chunks = (0..max_device_chunks)
            .into_iter()
            .map(|_| OnceCell::default())
            .collect();
        Ok(Self {
            device,
            host_ids,
            device_ids,
            host_chunks,
            device_chunks,
        })
    }
    fn alloc_host(&self, len: u32) -> Result<Arc<HostBuffer>> {
        for chunk in self.host_chunks.iter() {
            let chunk = chunk.get_or_try_init(|| {
                Chunk::new(
                    self.device.clone(),
                    MemoryKind::Host,
                    len as usize,
                    &self.host_ids,
                )
            })?;
            if let Some(alloc) = chunk.alloc(len) {
                return HostBuffer::new(alloc, len);
            }
        }
        Err(OomError::OutOfHostMemory.into())
    }
    fn alloc_device(&self, len: u32) -> Result<Arc<DeviceBuffer>> {
        for chunk in self.device_chunks.iter() {
            let chunk = chunk.get_or_try_init(|| {
                Chunk::new(
                    self.device.clone(),
                    MemoryKind::Device,
                    len as usize,
                    &self.device_ids,
                )
            })?;
            if let Some(alloc) = chunk.alloc(len) {
                return DeviceBuffer::new(self.device.clone(), alloc);
            }
        }
        Err(OomError::OutOfDeviceMemory.into())
    }
}

struct KernelCacheKey {
    spirv: &'static [u32],
    spec_consts: Vec<u32>,
}

impl Debug for KernelCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("KernelCacheKey")
            .field("spirv", &self.spirv.as_ptr())
            .field("spec_consts", &self.spec_consts)
            .finish()
    }
}

impl PartialEq for KernelCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.spirv.as_ptr() == other.spirv.as_ptr() && self.spec_consts == other.spec_consts
    }
}

impl Eq for KernelCacheKey {}

impl Hash for KernelCacheKey {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        state.write_usize(self.spirv.as_ptr() as usize);
        for spec in self.spec_consts.iter() {
            spec.hash(state);
        }
    }
}

#[derive(Debug)]
struct KernelCacheMap {
    device: Arc<Device>,
    kernels: DashMap<KernelCacheKey, Arc<KernelCache>>,
}

impl KernelCacheMap {
    fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            kernels: DashMap::new(),
        }
    }
    fn kernel(
        &self,
        spirv: &'static [u32],
        spec_consts: Vec<u32>,
        desc: KernelDesc,
    ) -> Result<Arc<KernelCache>> {
        let key = KernelCacheKey { spirv, spec_consts };
        use dashmap::mapref::entry::Entry;
        match self.kernels.entry(key) {
            Entry::Occupied(occupied) => {
                let cache = occupied.get();
                Ok(cache.clone())
            }
            Entry::Vacant(vacant) => {
                let key = vacant.key();
                let cache = Arc::new(KernelCache::new(
                    self.device.clone(),
                    &key.spirv,
                    &key.spec_consts,
                    desc,
                )?);
                vacant.insert(cache.clone());
                Ok(cache)
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct KernelCache {
    desc: KernelDesc,
    compute_pipeline: Arc<ComputePipeline>,
}

fn specialize(spirv: &[u32], mut spec_consts: &[u32]) -> Result<Vec<u32>> {
    use rspirv::{
        binary::Assemble,
        dr::{Instruction, Operand},
        spirv::{Decoration, Op},
    };
    let mut module = rspirv::dr::load_words(spirv).map_err(|e| format_err!("{e}"))?;
    let mut spec_ids = HashMap::<u32, Vec<u32>>::new();
    for inst in module.annotations.iter() {
        let op = inst.class.opcode;
        if op == Op::Decorate {
            if let [Operand::IdRef(id), Operand::Decoration(Decoration::SpecId), Operand::LiteralInt32(spec_id)] =
                inst.operands.as_slice()
            {
                spec_ids.entry(*spec_id).or_default().push(*id);
            }
        }
    }
    let mut spec_ops = HashMap::<u32, &mut Instruction>::new();
    for inst in module.types_global_values.iter_mut() {
        let op = inst.class.opcode;
        if op == Op::SpecConstant {
            if let Some(id) = inst.result_id {
                spec_ops.insert(id, inst);
            }
        }
    }
    let mut spec_id = 0;
    while !spec_consts.is_empty() {
        let mut len = 1;
        for id in spec_ids.get(&spec_id).unwrap() {
            if let Some(inst) = spec_ops.get_mut(id) {
                if let [Operand::LiteralInt32(a)] = inst.operands.as_mut_slice() {
                    *a = spec_consts.get(0).copied().unwrap_or_default();
                } else if let [Operand::LiteralInt32(a), Operand::LiteralInt32(b)] =
                    inst.operands.as_mut_slice()
                {
                    *a = spec_consts.get(0).copied().unwrap_or_default();
                    *b = spec_consts.get(1).copied().unwrap_or_default();
                    len = 2;
                }
            }
        }
        spec_consts = spec_consts.get(len..).unwrap_or(&[]);
    }
    Ok(module.assemble())
}

impl KernelCache {
    fn new(
        device: Arc<Device>,
        spirv: &[u32],
        spec_consts: &[u32],
        desc: KernelDesc,
    ) -> Result<Self> {
        use vulkano::descriptor_set::layout::DescriptorType;
        let spirv: Cow<[u32]> = if spec_consts.is_empty() {
            spirv.into()
        } else {
            specialize(spirv, spec_consts)?.into()
        };
        let stages = ShaderStages {
            compute: true,
            ..ShaderStages::none()
        };
        let descriptor_requirements = desc
            .buffer_descs()
            .iter()
            .enumerate()
            .map(|(i, desc)| {
                let set = 0u32;
                let binding = i as u32;
                let storage_write = if desc.mutable() { Some(binding) } else { None };
                let descriptor_requirements = DescriptorRequirements {
                    descriptor_types: vec![DescriptorType::StorageBuffer],
                    descriptor_count: Some(1),
                    stages,
                    storage_write: storage_write.into_iter().collect(),
                    ..DescriptorRequirements::default()
                };
                ((set, binding), descriptor_requirements)
            })
            .collect();
        let push_consts_size = desc.push_consts_size();
        let push_constant_range = if push_consts_size > 0 {
            Some(PushConstantRange {
                stages,
                offset: 0,
                size: push_consts_size,
            })
        } else {
            None
        };
        let entry_point_info = EntryPointInfo {
            execution: ShaderExecution::Compute,
            descriptor_requirements,
            push_constant_requirements: push_constant_range,
            specialization_constant_requirements: Default::default(),
            input_interface: ShaderInterface::empty(),
            output_interface: ShaderInterface::empty(),
        };
        let version = Version::major_minor(1, 2);
        let entry_point = "main";
        let shader_module = unsafe {
            ShaderModule::from_words_with_data(
                device.clone(),
                &spirv,
                version,
                [],
                [],
                [(
                    entry_point.to_string(),
                    ExecutionModel::GLCompute,
                    entry_point_info,
                )],
            )?
        };
        let bindings = (0..desc.buffer_descs().len())
            .map(|(binding)| {
                let descriptor_set_layout_binding = DescriptorSetLayoutBinding {
                    descriptor_count: 1,
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                };
                (binding as u32, descriptor_set_layout_binding)
            })
            .collect();
        let descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings,
            ..DescriptorSetLayoutCreateInfo::default()
        };
        let descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), descriptor_set_layout_create_info)?;
        let pipeline_layout_create_info = PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout],
            push_constant_ranges: push_constant_range.into_iter().collect(),
            ..PipelineLayoutCreateInfo::default()
        };
        let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_create_info)?;
        let cache = None;
        let specialization_constants = ();
        let compute_pipeline = ComputePipeline::with_pipeline_layout(
            device,
            shader_module.entry_point(entry_point).unwrap(),
            &specialization_constants,
            pipeline_layout,
            cache,
        )?;
        Ok(Self {
            desc,
            compute_pipeline,
        })
    }
    pub(crate) fn desc(&self) -> &KernelDesc {
        &self.desc
    }
}

trait Encode {
    unsafe fn encode(&self, encoder: &mut Encoder) -> Result<bool>;
}

#[derive(Debug)]
struct Upload {
    src: Arc<HostBuffer>,
    dst: Arc<DeviceBufferInner>,
}

impl Upload {
    fn barrier_key(&self) -> (usize, u32) {
        (self.dst.chunk_id(), self.dst.start)
    }
    fn barrier(&self) -> BufferMemoryBarrier {
        let src_stages = PipelineStages {
            copy: true,
            compute_shader: true,
            ..Default::default()
        };
        let src_access = AccessFlags {
            transfer_read: true,
            transfer_write: true,
            shader_read: true,
            shader_write: true,
            ..Default::default()
        };
        let dst_stages = PipelineStages {
            copy: true,
            ..Default::default()
        };
        let dst_access = AccessFlags {
            transfer_write: true,
            ..Default::default()
        };
        BufferMemoryBarrier {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            range: self.dst.barrier_range(),
            ..BufferMemoryBarrier::buffer(self.dst.buffer.clone())
        }
    }
    fn copy_buffer_info(&self) -> CopyBufferInfo {
        CopyBufferInfo::buffers(self.src.clone(), self.dst.clone())
    }
}

impl Encode for Upload {
    unsafe fn encode(&self, encoder: &mut Encoder) -> Result<bool> {
        let barrier = self.barrier();
        let prev_access = encoder
            .barriers
            .insert(self.barrier_key(), barrier.dst_access)
            .unwrap_or(AccessFlags::none());
        if prev_access != AccessFlags::none() {
            unsafe {
                encoder.cb_builder.pipeline_barrier(&DependencyInfo {
                    buffer_memory_barriers: [barrier].into_iter().collect(),
                    ..Default::default()
                });
            }
        }
        unsafe {
            encoder.cb_builder.copy_buffer(&self.copy_buffer_info());
        }
        Ok(true)
    }
}

#[derive(Debug)]
struct Download {
    src: Arc<DeviceBufferInner>,
    dst: Arc<HostBuffer>,
}

impl Download {
    fn barrier_key(&self) -> (usize, u32) {
        (self.src.chunk_id(), self.src.start)
    }
    fn barrier(&self) -> BufferMemoryBarrier {
        let src_stages = PipelineStages {
            copy: true,
            compute_shader: true,
            ..Default::default()
        };
        let src_access = AccessFlags {
            transfer_write: true,
            transfer_read: true,
            shader_write: true,
            shader_read: true,
            ..Default::default()
        };
        let dst_stages = PipelineStages {
            copy: true,
            ..Default::default()
        };
        let dst_access = AccessFlags {
            transfer_read: true,
            ..Default::default()
        };
        BufferMemoryBarrier {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            range: self.src.barrier_range(),
            ..BufferMemoryBarrier::buffer(self.src.buffer.clone())
        }
    }
    fn copy_buffer_info(&self) -> CopyBufferInfo {
        CopyBufferInfo::buffers(self.src.clone(), self.dst.clone())
    }
}

impl Encode for Download {
    unsafe fn encode(&self, encoder: &mut Encoder) -> Result<bool> {
        let barrier = self.barrier();
        let prev_access = encoder
            .barriers
            .insert(self.barrier_key(), barrier.dst_access)
            .unwrap_or(barrier.dst_access);
        if prev_access != barrier.dst_access {
            unsafe {
                encoder.cb_builder.pipeline_barrier(&DependencyInfo {
                    buffer_memory_barriers: [barrier].into_iter().collect(),
                    ..Default::default()
                });
            }
        }
        unsafe {
            encoder.cb_builder.copy_buffer(&self.copy_buffer_info());
        }
        Ok(true)
    }
}

#[derive(Debug)]
pub(crate) struct Compute {
    pub(crate) cache: Arc<KernelCache>,
    pub(crate) groups: [u32; 3],
    pub(crate) buffers: Vec<Arc<DeviceBufferInner>>,
    pub(crate) push_consts: Vec<u32>,
}

impl Compute {
    fn barrier(buffer: &Arc<DeviceBufferInner>, mutable: bool) -> BufferMemoryBarrier {
        let src_stages = PipelineStages {
            copy: true,
            compute_shader: true,
            ..Default::default()
        };
        let src_access = AccessFlags {
            transfer_write: true,
            transfer_read: true,
            shader_write: true,
            shader_read: true,
            ..Default::default()
        };
        let dst_stages = PipelineStages {
            compute_shader: true,
            ..Default::default()
        };
        let dst_access = AccessFlags {
            shader_read: true,
            shader_write: mutable,
            ..Default::default()
        };
        BufferMemoryBarrier {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            range: buffer.barrier_range(),
            ..BufferMemoryBarrier::buffer(buffer.buffer.clone())
        }
    }
}

impl Encode for Compute {
    unsafe fn encode(&self, encoder: &mut Encoder) -> Result<bool> {
        use vulkano::pipeline::Pipeline;
        let n_descriptors = self.buffers.len();
        if encoder.n_descriptors + n_descriptors > Encoder::MAX_DESCRIPTORS {
            return Ok(false);
        }
        let cache = &self.cache;
        let pipeline_layout = cache.compute_pipeline.layout();
        let layout = &pipeline_layout.set_layouts()[0];
        let descriptor_set_allocate_info = DescriptorSetAllocateInfo {
            layout,
            variable_descriptor_count: 0,
        };
        let mut descriptor_set = unsafe {
            encoder
                .frame
                .descriptor_pool
                .allocate_descriptor_sets([descriptor_set_allocate_info])?
        }
        .next()
        .unwrap();
        let writes = self
            .buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| WriteDescriptorSet::buffer(i as u32, buffer.clone()))
            .collect::<Vec<_>>();
        unsafe {
            descriptor_set.write(layout, writes.iter());
        }
        let mut cb_builder = &mut encoder.cb_builder;
        for (buffer, desc) in self.buffers.iter().zip(cache.desc.buffer_descs()) {
            let barrier = Self::barrier(buffer, desc.mutable());
            let barrier_key = (buffer.chunk_id(), buffer.start);
            let prev_access = encoder
                .barriers
                .insert(barrier_key, barrier.dst_access)
                .unwrap_or(barrier.dst_access);
            if prev_access != barrier.dst_access {
                unsafe {
                    cb_builder.pipeline_barrier(&DependencyInfo {
                        buffer_memory_barriers: [barrier].into_iter().collect(),
                        ..Default::default()
                    });
                }
            }
        }
        unsafe {
            cb_builder.bind_pipeline_compute(&cache.compute_pipeline);
        }
        let first_set = 0;
        let sets = [descriptor_set];
        let dynamic_offsets: &[u32] = &[];
        unsafe {
            cb_builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_layout,
                first_set,
                sets.iter(),
                [],
            );
        }
        let stages = ShaderStages {
            compute: true,
            ..ShaderStages::none()
        };
        if !self.push_consts.is_empty() {
            let offset = 0;
            let size = self.cache.desc.push_consts_size();
            let push_consts: &[u32] = self.push_consts.as_slice();
            unsafe {
                cb_builder.push_constants(pipeline_layout, stages, offset, size, push_consts);
            }
        }
        unsafe {
            cb_builder.dispatch(self.groups);
        }
        Ok(true)
    }
}

pub(super) struct SyncFuture {
    inner: Option<Arc<()>>,
    runner_result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
}

impl SyncFuture {
    fn new(runner_result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>) -> Self {
        Self {
            inner: Some(Arc::default()),
            runner_result,
        }
    }
    fn guard(&self) -> SyncGuard {
        SyncGuard {
            inner: self.inner.as_ref().map_or(Arc::default(), Arc::clone),
        }
    }
}

impl Future for SyncFuture {
    type Output = Result<()>;
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let inner = self.inner.take().unwrap();
        match Arc::try_unwrap(inner) {
            Ok(_) => {
                let result = self.runner_result.read().clone();
                if let Err(e) = result {
                    Poll::Ready(Err(anyhow::Error::msg(e)))
                } else {
                    Poll::Ready(Ok(()))
                }
            }
            Err(inner) => {
                self.inner.replace(inner);
                Poll::Pending
            }
        }
    }
}

#[derive(Debug)]
struct SyncGuard {
    inner: Arc<()>,
}

#[derive(Debug)]
enum Op {
    Upload(Upload),
    Download(Download),
    Compute(Compute),
    SyncGuard(SyncGuard),
}

/*
impl Op {
    fn name(&self) -> &'static str {
        match self {
            Self::Upload(_) => "Upload",
            Self::Download(_) => "Download",
            Self::Compute(_) => "Compute",
            Self::SyncGuard(_) => "SyncGuard",
        }
    }
}
*/

impl Encode for Op {
    unsafe fn encode(&self, encoder: &mut Encoder) -> Result<bool> {
        match self {
            Op::Upload(x) => unsafe { x.encode(encoder) },
            Op::Download(x) => unsafe { x.encode(encoder) },
            Op::Compute(x) => unsafe { x.encode(encoder) },
            Op::SyncGuard(_) => Ok(true),
        }
    }
}

struct Frame {
    queue: Arc<Queue>,
    device_index: usize,
    done: Arc<AtomicBool>,
    result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
    ops: Vec<Op>,
    command_pool: CommandPool,
    command_pool_alloc: Option<CommandPoolAlloc>,
    command_buffer: Option<UnsafeCommandBuffer>,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<UnsafeDescriptorSet>,
    semaphore: Arc<Semaphore>,
    fence: Fence,
}

impl Frame {
    const MAX_OPS: usize = 1_000;
    fn new(
        queue: Arc<Queue>,
        device_index: usize,
        done: Arc<AtomicBool>,
        result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
    ) -> Result<Self, anyhow::Error> {
        let device = queue.device();
        let command_pool_info = CommandPoolCreateInfo {
            queue_family_index: queue.queue_family_index(),
            transient: true,
            reset_command_buffer: false,
            ..CommandPoolCreateInfo::default()
        };
        let command_pool = CommandPool::new(device.clone(), command_pool_info)?;
        let descriptor_pool_create_info = DescriptorPoolCreateInfo {
            max_sets: Encoder::MAX_SETS as u32,
            pool_sizes: [(
                DescriptorType::StorageBuffer,
                Encoder::MAX_DESCRIPTORS as u32,
            )]
            .into_iter()
            .collect(),
            ..DescriptorPoolCreateInfo::default()
        };
        let descriptor_pool = DescriptorPool::new(device.clone(), descriptor_pool_create_info)?;
        let descriptor_sets = Vec::with_capacity(Encoder::MAX_DESCRIPTORS);
        let semaphore = Arc::new(Semaphore::from_pool(device.clone())?);
        let fence = Fence::new(
            device.clone(),
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )?;
        let ops = Vec::with_capacity(Self::MAX_OPS);
        Ok(Self {
            queue,
            device_index,
            done,
            result,
            ops,
            command_pool,
            command_pool_alloc: None,
            command_buffer: None,
            descriptor_pool,
            descriptor_sets,
            semaphore,
            fence,
        })
    }
    fn submit(&mut self, wait_semaphore: Option<&Semaphore>) -> Result<()> {
        debug_assert!(self.fence.is_signaled()?);
        self.fence.reset()?;
        use ash::vk;
        use vulkano::VulkanObject;

        let device = self.queue.device();
        let wait_semaphores = if let Some(semaphore) = wait_semaphore {
            Some([semaphore.handle()])
        } else {
            None
        };
        let wait_semaphores = if let Some(semaphores) = wait_semaphores.as_ref() {
            semaphores.as_ref()
        } else {
            &[]
        };
        let command_buffer = self.command_buffer.as_ref().unwrap().handle();
        let command_buffers = &[command_buffer];
        let semaphore = self.semaphore.handle();
        let semaphores = &[semaphore];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .command_buffers(command_buffers)
            .signal_semaphores(semaphores);
        unsafe {
            (device.fns().v1_0.queue_submit)(
                self.queue.handle(),
                1,
                &*submit_info as *const _,
                self.fence.handle(),
            )
            .result()?;
        }
        Ok(())
    }
    fn poll(&mut self) -> Result<()> {
        self.fence.wait(None)?;
        self.ops.clear();
        self.command_pool_alloc = None;
        self.command_buffer = None;
        self.descriptor_sets.clear();
        unsafe {
            self.descriptor_pool.reset()?;
        }
        Ok(())
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if !self.done.load(Ordering::SeqCst) {
            let mut result = self.result.write();
            if result.is_ok() {
                *result = Err(Arc::new(format_err!(
                    "Device({}) panicked!",
                    self.device_index
                )));
            }
        }
        if !self.ops.is_empty() {
            self.poll().unwrap();
        }
    }
}

struct Encoder {
    frame: Frame,
    cb_builder: UnsafeCommandBufferBuilder,
    barriers: HashMap<(usize, u32), AccessFlags>,
    n_descriptors: usize,
}

impl Encoder {
    const MAX_SETS: usize = Frame::MAX_OPS;
    const MAX_DESCRIPTORS: usize = Self::MAX_SETS * 2;
    fn new(mut frame: Frame) -> Result<Self> {
        debug_assert!(frame.fence.is_signaled()?);
        frame.command_pool_alloc = None;
        frame.command_buffer = None;
        let release_resources = false;
        unsafe {
            frame.command_pool.reset(release_resources)?;
        }
        let command_pool_alloc = frame
            .command_pool
            .allocate_command_buffers(Default::default())?
            .next()
            .unwrap();
        let cb_builder = unsafe {
            UnsafeCommandBufferBuilder::new(
                &command_pool_alloc,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::OneTimeSubmit,
                    ..Default::default()
                },
            )?
        };
        frame.command_pool_alloc.replace(command_pool_alloc);
        let barriers = HashMap::default();
        Ok(Self {
            frame,
            cb_builder,
            barriers,
            n_descriptors: 0,
        })
    }
    fn try_encode(&mut self, op: Op) -> Result<Option<Op>> {
        if !self.is_full() {
            let push_op = unsafe { op.encode(self)? };
            if push_op {
                self.frame.ops.push(op);
                return Ok(None);
            }
        }
        Ok(Some(op))
    }
    fn finish(mut self) -> Result<Frame> {
        self.frame.command_buffer.replace(self.cb_builder.build()?);
        Ok(self.frame)
    }
    fn is_full(&self) -> bool {
        self.frame.ops.len() >= Frame::MAX_OPS
            || self.frame.descriptor_sets.len() >= Self::MAX_SETS
            || self.n_descriptors >= Self::MAX_DESCRIPTORS
    }
    fn is_empty(&self) -> bool {
        self.frame.ops.is_empty()
    }
}

fn write_runner_result(
    runner_result: &Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
    result: Result<()>,
) {
    if let Err(e) = result {
        let mut runner_result = runner_result.write();
        if runner_result.is_ok() {
            *runner_result = Err(Arc::new(e));
        }
    }
}

struct Runner {
    done: Arc<AtomicBool>,
    op_sender: ManuallyDrop<Sender<Op>>,
    result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
    device: ManuallyDrop<Arc<Device>>,
}

impl Runner {
    fn new(queue: Arc<Queue>, index: usize) -> Result<Arc<Self>> {
        let done = Arc::new(AtomicBool::new(false));
        let (op_sender, op_reciever) = bounded(Frame::MAX_OPS);
        let n_frames = 3;
        let (encode_sender, encode_receiver) = bounded(n_frames);
        let (submit_sender, submit_receiver) = bounded(n_frames);
        let (poll_sender, poll_receiver) = bounded(n_frames);
        let result = Arc::new(RwLock::new(Ok(())));
        let mut ready_frames = VecDeque::with_capacity(n_frames);
        for _ in 0..n_frames {
            let frame = Frame::new(queue.clone(), index, done.clone(), result.clone())?;
            ready_frames.push_back(frame);
        }
        let result2 = result.clone();
        let done2 = done.clone();
        std::thread::Builder::new()
            .name(format!("krnl-device{}-encode", index))
            .spawn(move || {
                write_runner_result(
                    &result2,
                    encode(
                        &done2,
                        ready_frames,
                        op_reciever,
                        encode_receiver,
                        submit_sender,
                    ),
                );
            })?;
        let result2 = result.clone();
        std::thread::Builder::new()
            .name(format!("krnl-device{}-submit", index))
            .spawn(move || {
                write_runner_result(&result2, submit(submit_receiver, poll_sender));
            })?;
        let result2 = result.clone();
        std::thread::Builder::new()
            .name(format!("krnl-device{}-poll", index))
            .spawn(move || {
                write_runner_result(&result2, poll(poll_receiver, encode_sender));
            })?;
        Ok(Arc::new(Self {
            done,
            op_sender: ManuallyDrop::new(op_sender),
            result,
            device: ManuallyDrop::new(queue.device().clone()),
        }))
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        unsafe {
            ManuallyDrop::drop(&mut self.op_sender);
        }
        let mut device = unsafe { ManuallyDrop::take(&mut self.device) };
        loop {
            match Arc::try_unwrap(device) {
                Ok(device) => {
                    break;
                }
                Err(e) => {
                    device = e;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }
}

fn encode(
    done: &AtomicBool,
    mut ready_frames: VecDeque<Frame>,
    op_reciever: Receiver<Op>,
    encode_receiver: Receiver<Frame>,
    submit_sender: Sender<Frame>,
) -> Result<()> {
    let mut queued_op: Option<Op> = None;
    'outer: while !done.load(Ordering::Relaxed) {
        if queued_op.is_none() {
            loop {
                if let Ok(op) = op_reciever.recv() {
                    queued_op.replace(op);
                    break;
                } else {
                    break 'outer;
                }
            }
        }
        if ready_frames.is_empty() {
            if let Ok(frame) = encode_receiver.recv() {
                ready_frames.push_back(frame);
            } else {
                break 'outer;
            }
        }
        let mut encoder = Encoder::new(ready_frames.pop_front().unwrap())?;
        loop {
            while !encoder.is_full() {
                if queued_op.is_none() {
                    if let Ok(op) = op_reciever.try_recv() {
                        queued_op.replace(op);
                    } else {
                        break;
                    }
                }
                if let Some(op) = queued_op.take() {
                    queued_op = encoder.try_encode(op)?;
                } else {
                    break;
                }
            }
            ready_frames.extend(encode_receiver.try_iter());
            if done.load(Ordering::Relaxed) {
                break 'outer;
            }
            if ready_frames.len() >= 2 || (ready_frames.len() >= 1 && encoder.is_full()) {
                let frame = encoder.finish()?;
                if submit_sender.send(frame).is_err() {
                    break 'outer;
                }
                break;
            }
        }
    }
    Ok(())
}

fn submit(submit_receiver: Receiver<Frame>, poll_sender: Sender<Frame>) -> Result<()> {
    let mut wait_semaphore: Option<Arc<Semaphore>> = None;
    while let Ok(mut frame) = submit_receiver.recv() {
        frame.submit(wait_semaphore.take().as_deref())?;
        wait_semaphore.replace(frame.semaphore.clone());
        if poll_sender.send(frame).is_err() {
            break;
        }
    }
    Ok(())
}

fn poll(poll_receiver: Receiver<Frame>, encode_sender: Sender<Frame>) -> Result<()> {
    while let Ok(mut frame) = poll_receiver.recv() {
        frame.poll()?;
        if encode_sender.send(frame).is_err() {
            break;
        }
    }
    Ok(())
}
