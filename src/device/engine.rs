#![allow(unused)]
use krnl_types::{__private::raw_module::{RawKernelInfo, Mutability}, kernel::{VulkanVersion, KernelInfo}};
use crate::{device::DeviceOptions, scalar::Scalar};
use anyhow::{format_err, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use once_cell::sync::OnceCell;
use parking_lot::{Mutex, RwLock};
use spirv::Capability;
use std::{
    collections::{HashMap, VecDeque},
    future::Future,
    iter::{once, Peekable},
    pin::Pin,
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
    str::FromStr,
    ops::Deref,
};
use dashmap::DashMap;
use vulkano::{
    buffer::{
        cpu_access::ReadLock,
        cpu_pool::CpuBufferPoolChunk,
        device_local::DeviceLocalBuffer,
        sys::{UnsafeBuffer, UnsafeBufferCreateInfo},
        BufferAccess, BufferCreationError, BufferInner, BufferSlice, BufferUsage,
        CpuAccessibleBuffer, CpuBufferPool,
    },
    command_buffer::{
        pool::{
            CommandPool, UnsafeCommandPool, UnsafeCommandPoolAlloc, UnsafeCommandPoolCreateInfo,
        },
        submit::SubmitCommandBufferBuilder,
        sys::{
            CommandBufferBeginInfo,
            //    UnsafeCommandBufferBuilderPipelineBarrier,
            UnsafeCommandBuffer,
            UnsafeCommandBufferBuilder,
        },
        CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
    },
    descriptor_set::{WriteDescriptorSet, layout::{DescriptorSetLayout, DescriptorType, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo}, pool::{standard::StdDescriptorPoolAlloc, DescriptorPool, DescriptorPoolAlloc, UnsafeDescriptorPool, UnsafeDescriptorPoolCreateInfo, DescriptorSetAllocateInfo}, sys::UnsafeDescriptorSet},
    device::{
        physical::{MemoryType, PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Features, Queue, QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo, InstanceCreationError, InstanceExtensions, Version},
    memory::{
        pool::StdMemoryPool, DeviceMemory, DeviceMemoryAllocationError, MappedDeviceMemory,
        MemoryAllocateInfo,
    },
    pipeline::{ComputePipeline, PipelineBindPoint, PipelineLayout, layout::{PipelineLayoutCreateInfo, PushConstantRange}},
    shader::{
        spirv::{ExecutionModel, Capability as VulkanoCapability}, DescriptorRequirements, EntryPointInfo, ShaderExecution,
        ShaderInterface, ShaderModule, ShaderStages,
    },
    sync::{
        AccessFlags, BufferMemoryBarrier, DependencyInfo, Fence, FenceCreateInfo, PipelineStages,
        Semaphore,
    },
    DeviceSize,
    OomError,
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

struct Backend {
    instance: Arc<Instance>,
    engines: Vec<Mutex<Weak<Engine>>>,
}

impl Backend {
    fn get_or_try_init() -> Result<&'static Self, InstanceCreationError> {
        static BACKEND: OnceCell<Backend> = OnceCell::new();
        BACKEND.get_or_try_init(|| {
            let engine_name = Some("krnl".to_string());
            let engine_version = Version {
                major: u32::from_str(env!("CARGO_PKG_VERSION_MAJOR")).unwrap(),
                minor: u32::from_str(env!("CARGO_PKG_VERSION_MINOR")).unwrap(),
                patch: u32::from_str(env!("CARGO_PKG_VERSION_PATCH")).unwrap(),
            };
            #[allow(unused_mut)]
            let mut instance = Instance::new(InstanceCreateInfo {
                engine_name: engine_name.clone(),
                engine_version,
                enumerate_portability: true,
                .. InstanceCreateInfo::default()
            });
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            {
                use vulkano::instance::loader::FunctionPointers;
                if instance.is_err() {
                    let info = InstanceCreateInfo {
                        engine_name,
                        engine_version,
                        function_pointers: Some(FunctionPointers::new(Box::new(AshMoltenLoader))),
                        enumerate_portability: true,
                        ..InstanceCreateInfo::default()
                    };
                    instance = Instance::new(info);
                }
            }
            let instance = instance?;
            let engines = PhysicalDevice::enumerate(&instance)
                .map(|_| Mutex::default())
                .collect();
            Ok(Self { instance, engines })
        })
    }
}

fn spirv_capability_to_vulkano_capability(input: Capability) -> Result<VulkanoCapability> {
    todo!()
}

fn capabilites_to_features(capabilites: &[Capability]) -> Features {
    use Capability::*;
    let mut f = Features::none();
    for cap in capabilites {
        match cap {
            VulkanMemoryModel => {
                f.vulkan_memory_model = true;
            }
            StorageBuffer8BitAccess => {
                f.storage_buffer8_bit_access = true;
            }
            StorageBuffer16BitAccess => {
                f.storage_buffer16_bit_access = true;
            }
            Int8 => {
                f.shader_int8 = true;
            }
            Int16 => {
                f.shader_int16 = true;
            }
            Int64 => {
                f.shader_int64 = true;
            }
            _ => todo!(),
        }
    }
    f
}

fn features_to_capabilites(features: &Features) -> Vec<Capability> {
    use Capability::*;
    let f = features;
    let mut caps = Vec::new();
    if f.vulkan_memory_model {
        caps.push(VulkanMemoryModel);
    }
    if f.storage_buffer8_bit_access {
        caps.push(StorageBuffer8BitAccess);
    }
    if f.storage_buffer16_bit_access {
        caps.push(StorageBuffer16BitAccess);
    }
    if f.shader_int8 {
        caps.push(Int8);
    }
    if f.shader_int16 {
        caps.push(Int16);
    }
    if f.shader_int64 {
        caps.push(Int64);
    }
    caps
}

fn get_compute_family<'a>(
    physical_device: &'a PhysicalDevice,
) -> Result<QueueFamily<'a>, anyhow::Error> {
    physical_device
        .queue_families()
        .find(|x| !x.supports_graphics() && x.supports_compute())
        .or_else(|| {
            physical_device
                .queue_families()
                .find(|x| x.supports_compute())
        })
        .ok_or_else(|| {
            format_err!(
                "Device {} doesn't support compute!",
                physical_device.index()
            )
        })
}

#[derive(Clone, derive_more::Deref)]
pub(crate) struct ArcEngine {
    #[deref]
    engine: Arc<Engine>,
}

impl ArcEngine {
    pub(super) fn new(index: usize, options: &DeviceOptions) -> Result<Self, anyhow::Error> {
        Ok(Self {
            engine: Engine::new(index, options)?,
        })
    }
}

impl PartialEq for ArcEngine {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.engine, &other.engine)
    }
}

impl Eq for ArcEngine {}

pub(crate) struct Engine {
    device: Arc<Device>,
    buffer_allocator: BufferAllocator,
    kernel_cache_map: KernelCacheMap,
    runner: Arc<Runner>,
}

impl Engine {
    fn new(index: usize, options: &DeviceOptions) -> Result<Arc<Self>, anyhow::Error> {
        let backend = Backend::get_or_try_init()?;
        let physical_device =
            PhysicalDevice::from_index(&backend.instance, index).ok_or_else(|| {
                format_err!(
                    "Cannot create device at index {}, only {} devices!",
                    index,
                    backend.engines.len()
                )
            })?;
        let mut engine_guard = backend.engines[index].lock();
        if let Some(engine) = Weak::upgrade(&engine_guard) {
            return Ok(engine);
        }
        let compute_family = get_compute_family(&physical_device)?;
        let device_extensions = DeviceExtensions::none();
        let optimal_device_features = capabilites_to_features(&options.optimal_capabilities);
        let device_features = physical_device
            .supported_features()
            .intersection(&optimal_device_features);
        let mut queue_create_info = QueueCreateInfo::family(compute_family);
        queue_create_info.queues = vec![1f32];
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
        let runner = Runner::new(queue)?;
        let engine = Arc::new(Self {
            device,
            buffer_allocator,
            kernel_cache_map,
            runner,
        });
        *engine_guard = Arc::downgrade(&engine);
        Ok(engine)
    }
    pub(crate) fn index(&self) -> usize {
        self.device.physical_device().index()
    }
    pub(crate) fn vulkan_version(&self) -> VulkanVersion {
        let version = self.device.instance().api_version();
        VulkanVersion {
            major: version.major,
            minor: version.minor,
            patch: version.patch,
        }
    }
    pub(crate) fn supports_vulkan_version(&self, vulkan_version: VulkanVersion) -> bool {
        vulkan_version <= self.vulkan_version()
    }
    pub(crate) fn enabled_capabilities(&self) -> impl Iterator<Item=Capability> {
        todo!();
        [].into_iter()
    }
    pub(crate) fn capability_enabled(&self, capability: Capability) -> bool {
        self.enabled_capabilities().any(|x| x == capability)
    }
    pub(crate) fn enabled_extensions(&self) -> impl Iterator<Item=&'static str> {
        todo!();
        [].into_iter()
    }
    pub(crate) fn extension_enabled(&self, extension: &str) -> bool {
        self.enabled_extensions().any(|x| x == extension)
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
        self.runner.op_channel.sender.send(op).unwrap();
        Ok(())
    }
    // # Safety
    // Uninitialized.
    #[forbid(unsafe_op_in_unsafe_fn)]
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
        let len = bytes.len();
        if len == 0 {
            Ok(None)
        } else if len > u32::MAX as usize {
            anyhow::bail!(
                "Device buffer size {}B is too large, max is {}B!",
                len,
                u32::MAX
            );
        } else {
            let mut src = self.buffer_allocator.alloc_host(len as u32)?;
            Arc::get_mut(&mut src).unwrap().write_slice(bytes)?;
            let buffer = self.buffer_allocator.alloc_device(len as u32)?;
            let upload = Upload {
                src,
                dst: buffer.inner.clone(),
            };
            self.send_op(Op::Upload(upload))?;
            Ok(Some(buffer))
        }
    }
    pub(crate) fn download(&self, buffer: Arc<DeviceBuffer>) -> Result<HostBufferFuture> {
        let src = buffer.inner.clone();
        let dst = self.buffer_allocator.alloc_host(buffer.len() as u32)?;
        let download = Download {
            src,
            dst: dst.clone(),
        };
        self.send_op(Op::Download(download))?;
        Ok(HostBufferFuture {
            host_buffer: Some(dst),
            runner_result: self.runner.result.clone(),
        })
    }
    pub(crate) fn kernel_cache(&self, info: KernelInfo) -> Result<Arc<KernelCache>> {
        self.kernel_cache_map.kernel(info)
    }
    pub(crate) fn compute(&self, compute: Compute) -> Result<()> {
        self.send_op(Op::Compute(compute))?;
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct HostBuffer {
    alloc: Arc<ChunkAlloc<HostMemory>>,
    len: u32,
}

impl HostBuffer {
    fn new(alloc: Arc<ChunkAlloc<HostMemory>>, len: u32) -> Result<Arc<Self>> {
        Ok(Arc::new(Self { alloc, len }))
    }
    pub(crate) fn read(&self) -> Result<&[u8]> {
        let start = self.alloc.block.start as DeviceSize;
        let end = start + self.len as DeviceSize;
        Ok(unsafe { self.alloc.memory().memory.read(start..end)? })
    }
    fn write_slice(&mut self, slice: &[u8]) -> Result<()> {
        let start = self.alloc.block.start as DeviceSize;
        let end = start + self.len as DeviceSize;
        let data = unsafe { self.alloc.memory().memory.write(start..end)? };
        data.copy_from_slice(slice);
        Ok(())
    }
    fn chunk_id(&self) -> usize {
        Arc::as_ptr(&self.alloc.chunk) as usize
    }
    fn start(&self) -> DeviceSize {
        self.alloc.block.start as DeviceSize
    }
}

unsafe impl DeviceOwned for HostBuffer {
    fn device(&self) -> &Arc<Device> {
        self.alloc.memory().buffer.device()
    }
}

unsafe impl BufferAccess for HostBuffer {
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.alloc.memory().buffer,
            offset: self.alloc.block.start as DeviceSize,
        }
    }
    fn size(&self) -> DeviceSize {
        self.alloc.block.len() as DeviceSize
    }
    fn usage(&self) -> &BufferUsage {
        &self.alloc.memory().usage
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

#[derive(Debug)]
pub(crate) struct DeviceBufferInner {
    chunk: Arc<Chunk<DeviceMemory>>,
    buffer: Arc<UnsafeBuffer>,
    usage: BufferUsage,
    buffer_start: u32,
    len: u32,
    offset: u8,
    pad: u8,
}

impl DeviceBufferInner {
    fn chunk_id(&self) -> usize {
        Arc::as_ptr(&self.chunk) as usize
    }
    fn start(&self) -> DeviceSize {
        self.buffer_start as DeviceSize
    }
    pub(crate) fn offset_pad(&self) -> u32 {
        ((self.offset as u32) << 8) | (self.pad as u32)
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
            offset: 0,
        }
    }
    fn size(&self) -> DeviceSize {
        self.buffer.size()
    }
    fn usage(&self) -> &BufferUsage {
        &self.usage
    }
}

#[derive(Debug)]
pub(crate) struct DeviceBuffer {
    alloc: Arc<ChunkAlloc<DeviceMemory>>,
    inner: Arc<DeviceBufferInner>,
}

impl DeviceBuffer {
    fn new(
        device: Arc<Device>,
        alloc: Arc<ChunkAlloc<DeviceMemory>>,
        len: u32,
    ) -> Result<Arc<Self>> {
        let usage = BufferUsage::transfer_src()
            | BufferUsage::transfer_dst()
            | BufferUsage::storage_buffer();
        let align = device
            .physical_device()
            .properties()
            .min_storage_buffer_offset_alignment as u32;
        let pad = len % align;
        let buffer_len = len + pad;
        let buffer = UnsafeBuffer::new(
            device,
            UnsafeBufferCreateInfo {
                size: buffer_len as DeviceSize,
                usage,
                ..Default::default()
            },
        )?;
        unsafe { buffer.bind_memory(alloc.memory(), 0)? };
        let inner = Arc::new(DeviceBufferInner {
            chunk: alloc.chunk.clone(),
            buffer,
            usage,
            buffer_start: 0,
            len,
            offset: 0,
            pad: pad as u8,
        });
        Ok(Arc::new(Self { alloc, inner }))
    }
    pub(crate) fn len(&self) -> usize {
        self.inner.len as usize
    }
    pub(crate) fn inner(&self) -> Arc<DeviceBufferInner> {
        self.inner.clone()
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
struct ChunkAlloc<M> {
    chunk: Arc<Chunk<M>>,
    block: Block,
}

impl<M> ChunkAlloc<M> {
    fn memory(&self) -> &M {
        &self.chunk.memory
    }
}

impl<M> Drop for ChunkAlloc<M> {
    fn drop(&mut self) {
        let mut blocks = self.chunk.blocks.lock();
        if let Some(i) = blocks.iter().position(|x| x.start == self.block.start) {
            blocks.remove(i);
        }
    }
}

#[derive(Debug)]
struct HostMemory {
    memory: MappedDeviceMemory,
    buffer: Arc<UnsafeBuffer>,
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
        let buffer = UnsafeBuffer::new(
            device_memory.device().clone(),
            UnsafeBufferCreateInfo {
                size: device_memory.allocation_size(),
                usage,
                ..Default::default()
            },
        )?;
        unsafe { buffer.bind_memory(&device_memory, 0)? };
        let memory = MappedDeviceMemory::new(device_memory, 0..buffer.size())?;
        Ok(Self {
            memory,
            buffer,
            usage,
        })
    }
}

const CHUNK_ALIGN: u32 = 256;
const CHUNK_SIZE_MULTIPLE: usize = 256_000_000;

#[derive(Debug)]
struct Chunk<M> {
    memory: M,
    len: usize,
    blocks: Mutex<Vec<Block>>,
}

impl<M> Chunk<M> {
    fn new(device: Arc<Device>, len: usize, ids: &[u32]) -> Result<Arc<Self>>
    where
        M: ChunkMemory,
    {
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
                    let memory = M::from_device_memory(device_memory)?;
                    return Ok(Arc::new(Self {
                        memory,
                        len,
                        blocks: Mutex::default(),
                    }));
                }
                Err(DeviceMemoryAllocationError::OomError(e)) => continue,
                Err(e) => {
                    return Err(e.into());
                }
            }
        }
        Err(M::oom_error().into())
    }
    fn alloc(self: &Arc<Self>, len: u32) -> Option<Arc<ChunkAlloc<M>>> {
        if len as usize > self.len {
            return None;
        }
        let block_len = CHUNK_ALIGN * (1 + (len - 1) / CHUNK_ALIGN);
        let mut blocks = self.blocks.lock();
        let mut start = 0;
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

#[derive(Debug)]
struct BufferAllocator {
    device: Arc<Device>,
    host_ids: Vec<u32>,
    device_ids: Vec<u32>,
    host_chunks: Vec<Mutex<Weak<Chunk<HostMemory>>>>,
    device_chunks: Vec<Mutex<Weak<Chunk<DeviceMemory>>>>,
}

impl BufferAllocator {
    fn new(device: Arc<Device>) -> Result<Self> {
        let physical_device = device.physical_device();
        let mut max_host_chunks = 0;
        let mut max_device_chunks = 0;
        let mut host_ids = Vec::new();
        let mut device_ids = Vec::new();
        for memory_type in physical_device.memory_types() {
            let heap = memory_type.heap();
            if memory_type.is_host_visible() {
                max_host_chunks += (heap.size() / CHUNK_SIZE_MULTIPLE as u64) as usize;
                host_ids.push(memory_type.id());
            }
            if heap.is_device_local() {
                max_device_chunks += (heap.size() / CHUNK_SIZE_MULTIPLE as u64) as usize;
                device_ids.push(memory_type.id());
            }
        }
        // sort largest heap first
        host_ids.sort_by_key(|x| {
            -(physical_device.memory_type_by_id(*x).unwrap().heap().size() as i64)
        });
        device_ids.sort_by_key(|x| {
            -(physical_device.memory_type_by_id(*x).unwrap().heap().size() as i64)
        });
        let host_chunks = (0..max_host_chunks)
            .into_iter()
            .map(|_| Mutex::default())
            .collect();
        let device_chunks = (0..max_device_chunks)
            .into_iter()
            .map(|_| Mutex::default())
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
            let mut chunk = chunk.lock();
            if let Some(chunk) = Weak::upgrade(&chunk) {
                if let Some(alloc) = chunk.alloc(len) {
                    return HostBuffer::new(alloc, len);
                }
            } else {
                let new_chunk = Chunk::new(self.device.clone(), len as usize, &self.host_ids)?;
                let alloc = new_chunk.alloc(len).unwrap();
                *chunk = Arc::downgrade(&new_chunk);
                return HostBuffer::new(alloc, len);
            }
        }
        Err(OomError::OutOfHostMemory.into())
    }
    fn alloc_device(&self, len: u32) -> Result<Arc<DeviceBuffer>> {
        for chunk in self.device_chunks.iter() {
            let mut chunk = chunk.lock();
            if let Some(chunk) = Weak::upgrade(&chunk) {
                if let Some(alloc) = chunk.alloc(len) {
                    return DeviceBuffer::new(self.device.clone(), alloc, len);
                }
            } else {
                let new_chunk = Chunk::new(self.device.clone(), len as usize, &self.device_ids)?;
                let alloc = new_chunk.alloc(len).unwrap();
                *chunk = Arc::downgrade(&new_chunk);
                return DeviceBuffer::new(self.device.clone(), alloc, len);
            }
        }
        Err(OomError::OutOfHostMemory.into())
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
    fn kernel(&self, info: KernelInfo) -> Result<Arc<KernelCache>> {
        let key = KernelCacheKey {
            module_id: Arc::as_ptr(info.__module()) as usize,
            kernel: info.__info().name.to_string(),
        };
        let kernel_cache = self.kernels.entry(key)
            .or_try_insert_with(|| KernelCache::new(self.device.clone(), info))?
            .value()
            .clone();
        Ok(kernel_cache)
    }
}

#[derive(Eq, PartialEq, Hash, Debug)]
struct KernelCacheKey {
    module_id: usize,
    kernel: String,
}

#[derive(Debug)]
pub(crate) struct KernelCache {
    info: KernelInfo,
    shader_module: Arc<ShaderModule>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    pipeline_layout: Arc<PipelineLayout>,
    compute_pipeline: Arc<ComputePipeline>,
}

impl KernelCache {
    fn new(device: Arc<Device>, info: KernelInfo) -> Result<Arc<Self>> {
        use vulkano::descriptor_set::layout::DescriptorType;
        let kernel_info = info.__info();
        let vulkan_version = &kernel_info.vulkan_version;
        let version = Version {
            major: vulkan_version.major,
            minor: vulkan_version.minor,
            patch: vulkan_version.patch,
        };
        let stages = ShaderStages {
            compute: true,
            .. ShaderStages::none()
        };
        let storage_write = kernel_info.slice_infos.iter()
            .enumerate()
            .filter(|(_, x)| x.mutability.is_mutable())
            .map(|(i, _)| i as u32)
            .collect();
        let descriptor_count = kernel_info.slice_infos.len() as u32;
        let descriptor_requirements = [((0, 0), DescriptorRequirements {
            descriptor_types: vec![DescriptorType::StorageBuffer],
            descriptor_count,
            stages,
            storage_write,
            .. DescriptorRequirements::default()
        })].into_iter().collect();
        let push_constant_range = if kernel_info.num_push_words > 0 {
            Some(PushConstantRange {
                stages,
                offset: 0,
                size: kernel_info.num_push_words * 4,
            })
        } else {
            None
        };
        let specialization_constant_requirements = HashMap::new();
        let entry_point_info = EntryPointInfo {
            execution: ShaderExecution::Compute,
            descriptor_requirements,
            push_constant_requirements: push_constant_range,
            specialization_constant_requirements,
            input_interface: ShaderInterface::empty(),
            output_interface: ShaderInterface::empty(),
        };
        let mut capabilities = Vec::with_capacity(kernel_info.capabilities.len());
        for cap in kernel_info.capabilities.iter().copied() {
            capabilities.push(spirv_capability_to_vulkano_capability(cap)?);
        }
        let shader_module = unsafe {
            ShaderModule::from_words_with_data(
                device.clone(),
                &kernel_info.spirv.as_ref().unwrap().words,
                version,
                capabilities.iter(),
                kernel_info.extensions.iter().map(Deref::deref),
                [(kernel_info.name.clone(), ExecutionModel::GLCompute, entry_point_info)],
            )?
        };
        let descriptor_set_layout_binding = DescriptorSetLayoutBinding {
            descriptor_count,
            stages,
            .. DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
        };
        let descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings: [(0, descriptor_set_layout_binding)].into_iter().collect(),
            .. DescriptorSetLayoutCreateInfo::default()
        };
        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            descriptor_set_layout_create_info,
        )?;
        let pipeline_layout_create_info = PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout.clone()],
            push_constant_ranges: push_constant_range.into_iter().collect(),
            .. PipelineLayoutCreateInfo::default()
        };
        let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_create_info)?;
        let specialization_constants = ();
        let cache = None;
        let compute_pipeline = ComputePipeline::with_pipeline_layout(
            device.clone(),
            shader_module.entry_point(&kernel_info.name).unwrap(),
            &specialization_constants,
            pipeline_layout.clone(),
            cache,
        )?;
        Ok(Arc::new(Self {
            info,
            shader_module,
            pipeline_layout,
            descriptor_set_layout,
            compute_pipeline,
        }))
    }
    pub(crate) fn info(&self) -> &KernelInfo {
        &self.info
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
    fn barrier_key(&self) -> (usize, DeviceSize) {
        (self.dst.chunk_id(), self.dst.start())
    }
    fn barrier(&self) -> BufferMemoryBarrier {
        let source_stages = PipelineStages {
            transfer: true,
            compute_shader: true,
            ..Default::default()
        };
        let source_access = AccessFlags {
            transfer_read: true,
            transfer_write: true,
            shader_read: true,
            shader_write: true,
            ..Default::default()
        };
        let destination_stages = PipelineStages {
            transfer: true,
            ..Default::default()
        };
        let destination_access = AccessFlags {
            transfer_write: true,
            ..Default::default()
        };
        BufferMemoryBarrier {
            source_stages,
            source_access,
            destination_stages,
            destination_access,
            range: 0..self.dst.buffer.size(),
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
            .insert(self.barrier_key(), barrier.destination_access)
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
    fn barrier_key(&self) -> (usize, DeviceSize) {
        (self.src.chunk_id(), self.src.start())
    }
    fn barrier(&self) -> BufferMemoryBarrier {
        let source_stages = PipelineStages {
            transfer: true,
            compute_shader: true,
            ..Default::default()
        };
        let source_access = AccessFlags {
            transfer_write: true,
            transfer_read: true,
            shader_write: true,
            shader_read: true,
            ..Default::default()
        };
        let destination_stages = PipelineStages {
            transfer: true,
            ..Default::default()
        };
        let destination_access = AccessFlags {
            transfer_read: true,
            ..Default::default()
        };
        BufferMemoryBarrier {
            source_stages,
            source_access,
            destination_stages,
            destination_access,
            range: 0..self.src.buffer.size(),
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
            .insert(self.barrier_key(), barrier.destination_access)
            .unwrap_or(barrier.destination_access);
        if prev_access != barrier.destination_access {
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
    fn barrier(buffer: &Arc<DeviceBufferInner>, mutability: Mutability) -> BufferMemoryBarrier {
        let source_stages = PipelineStages {
            transfer: true,
            compute_shader: true,
            ..Default::default()
        };
        let source_access = AccessFlags {
            transfer_write: true,
            transfer_read: true,
            shader_write: true,
            shader_read: true,
            ..Default::default()
        };
        let destination_stages = PipelineStages {
            compute_shader: true,
            ..Default::default()
        };
        let destination_access = AccessFlags {
            shader_read: true,
            shader_write: mutability.is_mutable(),
            ..Default::default()
        };
        BufferMemoryBarrier {
            source_stages,
            source_access,
            destination_stages,
            destination_access,
            range: buffer.start() .. buffer.start() + buffer.size(),
            ..BufferMemoryBarrier::buffer(buffer.buffer.clone())
        }
    }
}


impl Encode for Compute {
    unsafe fn encode(&self, encoder: &mut Encoder) -> Result<bool> {
        let n_descriptors = self.buffers.len();
        if encoder.n_descriptors + n_descriptors > Encoder::MAX_DESCRIPTORS {
            return Ok(false);
        }
        let cache = &self.cache;
        let layout = &cache.descriptor_set_layout;
        let descriptor_set_allocate_info = DescriptorSetAllocateInfo {
            layout,
            variable_descriptor_count: 0,
        };
        let mut descriptor_set = unsafe {
            encoder.frame.descriptor_pool.allocate_descriptor_sets(
                [descriptor_set_allocate_info]
            )?
        }.next().unwrap();
        let writes = self.buffers.iter().enumerate()
            .map(|(i, x)| WriteDescriptorSet::buffer(i as u32, x.clone()))
            .collect::<Vec<_>>();
        unsafe {
            descriptor_set.write(layout, writes.iter());
        }
        let slice_infos = &cache.info.__info().slice_infos;
        let mut cb_builder = &mut encoder.cb_builder;
        for (buffer, slice_info) in self.buffers.iter().zip(slice_infos.iter()) {
            let barrier = Self::barrier(buffer, slice_info.mutability);
            let barrier_key = (buffer.chunk_id(), buffer.start());
            let prev_access = encoder
                .barriers
                .insert(barrier_key, barrier.destination_access)
                .unwrap_or(barrier.destination_access);
            if prev_access != barrier.destination_access {
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
                &cache.pipeline_layout,
                first_set,
                sets.iter(),
                []
            );
        }
        let stages = ShaderStages {
            compute: true,
            ..ShaderStages::none()
        };
        if !self.push_consts.is_empty() {
            let offset = 0;
            let size = (self.push_consts.len() * 4) as u32;
            unsafe {
                cb_builder.push_constants(
                    &cache.pipeline_layout,
                    stages,
                    offset,
                    size,
                    self.push_consts.as_slice(),
                );
            }
        }
        unsafe {
            cb_builder.dispatch(self.groups);
        }
        Ok(true)
    }
}

#[derive(Debug)]
enum Op {
    Upload(Upload),
    Download(Download),
    Compute(Compute),
}


impl Encode for Op {
    unsafe fn encode(&self, encoder: &mut Encoder) -> Result<bool> {
        match self {
            Op::Upload(x) => unsafe {
                x.encode(encoder)
            }
            Op::Download(x) => unsafe {
                x.encode(encoder)
            }
            Op::Compute(x) => unsafe {
                x.encode(encoder)
            }
        }
    }
}

struct Frame {
    queue: Arc<Queue>,
    done: Arc<AtomicBool>,
    result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
    command_pool: UnsafeCommandPool,
    command_pool_alloc: Option<UnsafeCommandPoolAlloc>,
    command_buffer: Option<UnsafeCommandBuffer>,
    descriptor_pool: UnsafeDescriptorPool,
    descriptor_sets: Vec<UnsafeDescriptorSet>,
    semaphore: Arc<Semaphore>,
    fence: Fence,
    ops: Vec<Op>,
}

impl Frame {
    const MAX_OPS: usize = 1_000;
    fn new(
        queue: Arc<Queue>,
        done: Arc<AtomicBool>,
        result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
    ) -> Result<Self, anyhow::Error> {
        let device = queue.device();
        let command_pool_info = UnsafeCommandPoolCreateInfo {
            queue_family_index: queue.family().id(),
            transient: true,
            reset_command_buffer: false,
            ..UnsafeCommandPoolCreateInfo::default()
        };
        let command_pool = UnsafeCommandPool::new(device.clone(), command_pool_info)?;
        let descriptor_pool_create_info = UnsafeDescriptorPoolCreateInfo {
            max_sets: Encoder::MAX_SETS as u32,
            pool_sizes: [(DescriptorType::StorageBuffer, Encoder::MAX_DESCRIPTORS as u32)].into_iter().collect(),
            .. UnsafeDescriptorPoolCreateInfo::default()
        };
        let descriptor_pool = UnsafeDescriptorPool::new(
            device.clone(),
            descriptor_pool_create_info,
        )?;
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
            done,
            result,
            command_pool,
            command_pool_alloc: None,
            command_buffer: None,
            descriptor_pool,
            descriptor_sets,
            semaphore,
            fence,
            ops,
        })
    }
    fn submit(&mut self, wait_semaphore: Option<&Semaphore>) -> Result<()> {
        debug_assert!(self.fence.is_signaled()?);
        self.fence.reset()?;
        let mut submit_builder = SubmitCommandBufferBuilder::new();
        if let Some(wait_semaphore) = wait_semaphore {
            unsafe {
                submit_builder.add_wait_semaphore(
                    &wait_semaphore,
                    PipelineStages {
                        bottom_of_pipe: true,
                        ..Default::default()
                    },
                );
            }
        }
        unsafe {
            submit_builder.add_command_buffer(&self.command_buffer.as_ref().unwrap());
        }
        self.semaphore = Arc::new(Semaphore::from_pool(self.queue.device().clone())?);
        unsafe {
            submit_builder.add_signal_semaphore(&self.semaphore);
            submit_builder.set_fence_signal(&self.fence);
        }
        submit_builder.submit(&self.queue)?;
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
            let index = self.queue.device().physical_device().index();
            let mut result = self.result.write();
            if result.is_ok() {
                *result = Err(Arc::new(format_err!("Device({}) panicked!", index)));
            }
        }
        self.queue.wait().unwrap();
    }
}

struct Encoder {
    frame: Frame,
    cb_builder: UnsafeCommandBufferBuilder,
    barriers: HashMap<(usize, DeviceSize), AccessFlags>,
    n_descriptors: usize,
}

impl Encoder {
    const MAX_SETS: usize = 1_000;
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
    fn extend(&mut self, op_iter: &mut Peekable<impl Iterator<Item = Op>>) -> Result<()> {
        while let Some(op) = op_iter.peek() {
            if self.is_full() {
                break;
            }
            let push_op = unsafe {
                op.encode(self)?
            };
            if push_op {
                self.frame.ops.push(op_iter.next().unwrap());
            }
        }
        Ok(())
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

struct Channel<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T> Channel<T> {
    fn bounded(len: usize) -> Self {
        let (sender, receiver) = bounded(len);
        Self { sender, receiver }
    }
}

struct Runner {
    queue: Arc<Queue>,
    done: Arc<AtomicBool>,
    op_channel: Channel<Op>,
    encode_channel: Channel<Frame>,
    submit_channel: Channel<Frame>,
    poll_channel: Channel<Frame>,
    result: Arc<RwLock<Result<(), Arc<anyhow::Error>>>>,
}

impl Runner {
    fn new(queue: Arc<Queue>) -> Result<Arc<Self>> {
        let index = queue.device().physical_device().index();
        let op_channel = Channel::bounded(1_000);
        let encode_channel = Channel::bounded(3);
        let submit_channel = Channel::bounded(1);
        let poll_channel = Channel::bounded(1);
        let runner = Arc::new(Self {
            queue,
            done: Arc::new(AtomicBool::new(false)),
            op_channel,
            encode_channel,
            submit_channel,
            poll_channel,
            result: Arc::new(RwLock::new(Ok(()))),
        });
        let mut ready_frames = VecDeque::with_capacity(3);
        for _ in 0..3 {
            let frame = Frame::new(
                runner.queue.clone(),
                runner.done.clone(),
                runner.result.clone(),
            )?;
            ready_frames.push_back(frame);
        }
        {
            let runner = runner.clone();
            std::thread::Builder::new()
                .name(format!("krnl-device{}-encode", index))
                .spawn(move || {
                    if let Err(e) = runner.encode(ready_frames) {
                        let mut result = runner.result.write();
                        if result.is_ok() {
                            *result = Err(Arc::new(e));
                        }
                    }
                })?;
        }
        {
            let runner = runner.clone();
            std::thread::Builder::new()
                .name(format!("krnl-device{}-submit", index))
                .spawn(move || {
                    if let Err(e) = runner.submit() {
                        let mut result = runner.result.write();
                        if result.is_ok() {
                            *result = Err(Arc::new(e));
                        }
                    }
                })?;
        }
        {
            let runner = runner.clone();
            std::thread::Builder::new()
                .name(format!("krnl-device{}-poll", index))
                .spawn(move || {
                    if let Err(e) = runner.poll() {
                        let mut result = runner.result.write();
                        if result.is_ok() {
                            *result = Err(Arc::new(e));
                        }
                    }
                })?;
        }
        Ok(runner)
    }
    fn encode(&self, mut ready_frames: VecDeque<Frame>) -> Result<()> {
        let mut last_submit = Instant::now();
        let mut encoder = None;
        let mut op_iter = self.op_channel.receiver.try_iter().peekable();
        while !self.done.load(Ordering::Relaxed) {
            ready_frames.extend(self.encode_channel.receiver.try_iter());
            if encoder.is_none() {
                if let Some(frame) = ready_frames.pop_front() {
                    encoder.replace(Encoder::new(frame)?);
                }
            }
            if let Some(encoder_mut) = encoder.as_mut() {
                if op_iter.peek().is_none() {
                    op_iter = self.op_channel.receiver.try_iter().peekable();
                }
                encoder_mut.extend(&mut op_iter);
                if (ready_frames.len() >= 2
                    && last_submit.elapsed().as_millis() >= 1
                    && !encoder_mut.is_empty())
                    || (ready_frames.len() >= 1 && encoder_mut.is_full())
                {
                    let frame = encoder.take().unwrap().finish()?;
                    self.submit_channel.sender.send(frame).unwrap();
                    last_submit = Instant::now();
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
    fn submit(&self) -> Result<()> {
        let mut wait_semaphore: Option<Arc<Semaphore>> = None;
        while !self.done.load(Ordering::Relaxed) {
            for mut frame in self.submit_channel.receiver.try_iter() {
                frame.submit(wait_semaphore.take().as_deref())?;
                wait_semaphore.replace(frame.semaphore.clone());
                self.poll_channel.sender.send(frame).unwrap();
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
    fn poll(&self) -> Result<()> {
        while !self.done.load(Ordering::Relaxed) {
            for mut frame in self.poll_channel.receiver.try_iter() {
                frame.poll()?;
                self.encode_channel.sender.send(frame).unwrap();
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
    }
}
