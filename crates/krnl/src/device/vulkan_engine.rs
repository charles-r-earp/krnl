use super::{
    DeviceEngine, DeviceEngineBuffer, DeviceEngineKernel, DeviceInfo, DeviceLost, DeviceOptions,
    Features, KernelDesc, KernelKey,
};
use crate::{device, scalar::ScalarElem};
use anyhow::{format_err, Result};
use ash::vk::{Handle, PipelineStageFlags};
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard};
use std::{
    borrow::Cow,
    collections::HashMap,
    hash::{Hash, Hasher},
    ops::{Deref, Range, RangeBounds},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};
use vulkano::{
    buffer::{
        sys::{Buffer, BufferCreateInfo, BufferError, BufferMemory, RawBuffer},
        BufferAccess, BufferInner, BufferSlice, BufferUsage, CpuAccessibleBuffer,
        DeviceLocalBuffer,
    },
    command_buffer::{
        self,
        allocator::{
            CommandBufferAlloc, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
        pool::{CommandBufferAllocateInfo, CommandPool, CommandPoolAlloc, CommandPoolCreateInfo},
        sys::{CommandBufferBeginInfo, UnsafeCommandBuffer, UnsafeCommandBufferBuilder},
        CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
    },
    descriptor_set::{
        self,
        allocator::{
            DescriptorSetAlloc, DescriptorSetAllocator, StandardDescriptorSetAlloc,
            StandardDescriptorSetAllocator,
        },
        layout::{DescriptorSetLayout, DescriptorType},
        pool::{DescriptorPool, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo},
        sys::UnsafeDescriptorSet,
        DescriptorSet, DescriptorSetResources, DescriptorSetWithOffsets, WriteDescriptorSet,
    },
    device::{Device, DeviceCreateInfo, DeviceOwned, Queue, QueueCreateInfo},
    instance::{Instance, InstanceCreateInfo, Version},
    library::VulkanLibrary,
    memory::{
        allocator::{
            GenericMemoryAllocator, GenericMemoryAllocatorCreateInfo, MemoryAlloc, MemoryAllocator,
            StandardMemoryAllocator,
        },
        DedicatedAllocation, DeviceMemory, MemoryAllocateInfo,
    },
    pipeline::{self, ComputePipeline, Pipeline, PipelineBindPoint},
    shader::{
        DescriptorRequirements, ShaderExecution, ShaderInterface, ShaderModule, ShaderStages,
    },
    sync::{Fence, FenceError, PipelineStage, Semaphore},
    VulkanObject,
};

pub struct Engine {
    info: Arc<DeviceInfo>,
    compute_families: Vec<u32>,
    compute_op_sender: Sender<Op>,
    transfer_op_sender: Sender<Op>,
    worker_states: Vec<WorkerState>,
    exited: Arc<AtomicBool>,
    kernels: DashMap<KernelKey, KernelInner>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    device: Arc<Device>,
    instance: Arc<Instance>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.compute_op_sender = crossbeam_channel::bounded(0).0;
        self.transfer_op_sender = crossbeam_channel::bounded(0).0;
        while Arc::strong_count(&self.exited) > 1 {
            std::thread::sleep(Duration::from_micros(10));
        }
    }
}

impl DeviceEngine for Engine {
    type DeviceBuffer = DeviceBuffer;
    type Kernel = Kernel;
    fn new(options: DeviceOptions) -> anyhow::Result<std::sync::Arc<Self>> {
        let DeviceOptions {
            index,
            optimal_features,
        } = options;
        let instance = Instance::new(
            VulkanLibrary::new()?,
            InstanceCreateInfo::application_from_cargo_toml(),
        )?;
        let physical_devices = instance.enumerate_physical_devices()?;
        let devices = physical_devices.len();
        let physical_device = if let Some(physical_device) = physical_devices.skip(index).next() {
            physical_device
        } else {
            return Err(super::DeviceIndexOutOfRange { index, devices }.into());
        };
        let physical_device = instance
            .enumerate_physical_devices()?
            .skip(options.index)
            .next()
            .unwrap();
        let name = physical_device.properties().device_name.clone();
        let optimal_device_extensions = vulkano::device::DeviceExtensions {
            khr_vulkan_memory_model: true,
            ..vulkano::device::DeviceExtensions::none()
        };
        let device_extensions = physical_device
            .supported_extensions()
            .intersection(&optimal_device_extensions);
        let optimal_device_features = vulkano::device::Features {
            vulkan_memory_model: true,
            shader_int8: optimal_features.shader_int8,
            shader_int16: optimal_features.shader_int16,
            shader_int64: optimal_features.shader_int64,
            shader_float16: optimal_features.shader_float16,
            shader_float64: optimal_features.shader_float64,
            ..vulkano::device::Features::empty()
        };
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
        let mut compute_families: Vec<_> = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .filter(|(_, x)| x.queue_flags.compute)
            .map(|(i, x)| (i as u32, x.queue_flags))
            .collect();
        compute_families.sort_by_key(|(i, flags)| flags.graphics);
        let mut compute_families: Vec<u32> = compute_families.iter().map(|(i, _)| *i).collect();
        let mut transfer_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|x| {
                let flags = x.queue_flags;
                flags.transfer && !flags.compute && !flags.graphics
            })
            .map(|x| x as u32);
        if transfer_family.is_none() {
            compute_families.truncate(1);
        }
        let queue_create_infos: Vec<_> = compute_families
            .iter()
            .copied()
            .chain(transfer_family)
            .map(|queue_family_index| QueueCreateInfo {
                queue_family_index,
                queues: vec![1f32],
                ..Default::default()
            })
            .collect();
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                queue_create_infos,
                ..Default::default()
            },
        )?;
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(
            device.clone(),
            GenericMemoryAllocatorCreateInfo {
                block_sizes: &[
                    (0, (DeviceBuffer::HOST_BUFFER_SIZE * 2) as _),
                    (DeviceBuffer::MAX_SIZE as _, DeviceBuffer::MAX_SIZE as _),
                ],
                dedicated_allocation: false,
                ..Default::default()
            },
        )?);
        let mut worker_states = Vec::with_capacity(queues.len());
        let exited = Arc::new(AtomicBool::default());
        let compute_queues: Vec<_> = queues.by_ref().take(compute_families.len()).collect();
        let (compute_op_sender, compute_op_receiver) = crossbeam_channel::bounded(0);
        for queue in compute_queues {
            let op_receiver = Arc::new(Mutex::new(compute_op_receiver.clone()));
            let memory_allocator = if transfer_family.is_none() {
                Some(&memory_allocator)
            } else {
                None
            };
            for _ in 0..2 {
                let worker = Worker::new(
                    op_receiver.clone(),
                    memory_allocator,
                    true,
                    queue.clone(),
                    exited.clone(),
                )?;
                worker_states.push(worker.state.clone());
                std::thread::spawn(move || worker.run());
            }
        }
        let transfer_queue = queues.next();
        let transfer_op_sender = if let Some(queue) = transfer_queue {
            let (op_sender, op_receiver) = crossbeam_channel::bounded(0);
            for _ in 0..2 {
                let worker = Worker::new(
                    Arc::new(Mutex::new(op_receiver.clone())),
                    Some(&memory_allocator),
                    false,
                    queue.clone(),
                    exited.clone(),
                )?;
                worker_states.push(worker.state.clone());
                std::thread::spawn(move || worker.run());
            }
            op_sender
        } else {
            compute_op_sender.clone()
        };
        let queue_family_indices: Vec<u32> = compute_families
            .iter()
            .copied()
            .chain(transfer_family)
            .collect();
        let kernels = DashMap::default();
        let info = Arc::new(DeviceInfo {
            index,
            name,
            compute_queues: compute_families.len(),
            transfer_queues: transfer_family.is_some() as usize,
            features,
        });
        Ok(Arc::new(Self {
            info,
            compute_families,
            compute_op_sender,
            transfer_op_sender,
            worker_states,
            exited,
            kernels,
            memory_allocator,
            device,
            instance,
        }))
    }
    fn handle(&self) -> u64 {
        self.device.handle().as_raw()
    }
    fn info(&self) -> &Arc<DeviceInfo> {
        &self.info
    }
    fn wait(&self) -> Result<(), DeviceLost> {
        let pending: Vec<usize> = self
            .worker_states
            .iter()
            .map(|x| x.pending.load(Ordering::SeqCst))
            .collect();
        loop {
            if self.exited.load(Ordering::SeqCst) {
                return Err(DeviceLost {
                    index: self.info.index,
                    handle: self.handle(),
                });
            } else if self
                .worker_states
                .iter()
                .zip(pending.iter().copied())
                .any(|(state, pending)| state.completed.load(Ordering::SeqCst) < pending)
            {
                std::thread::sleep(Duration::from_micros(1));
            } else {
                return Ok(());
            }
        }
    }
}

struct HostBuffer {
    inner: Arc<Buffer>,
}

unsafe impl DeviceOwned for HostBuffer {
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl BufferAccess for HostBuffer {
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.inner,
            offset: 0,
        }
    }
    fn size(&self) -> vulkano::DeviceSize {
        self.inner.size()
    }
}

#[derive(Default, Clone, Debug)]
struct WorkerState {
    pending: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
}

impl WorkerState {
    fn next(&self) -> WorkerFuture {
        let pending = self.pending.fetch_add(1, Ordering::SeqCst) + 1;
        let completed = self.completed.clone();
        WorkerFuture { pending, completed }
    }
    fn finish(&self) {
        self.completed.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Default, Clone, Debug)]
struct WorkerFuture {
    pending: usize,
    completed: Arc<AtomicUsize>,
}

impl WorkerFuture {
    fn ready(&self) -> bool {
        self.completed.load(Ordering::SeqCst) >= self.pending
    }
}

struct Worker {
    op_receiver: Arc<Mutex<Receiver<Op>>>,
    state: WorkerState,
    fence: Fence,
    command_pool: CommandPool,
    descriptor_pool: Option<DescriptorPool>,
    host_buffer: Option<Arc<CpuAccessibleBuffer<[u8]>>>,
    queue: Arc<Queue>,
    guard: WorkerDropGuard,
}

impl Worker {
    fn new(
        op_receiver: Arc<Mutex<Receiver<Op>>>,
        memory_allocator: Option<&Arc<StandardMemoryAllocator>>,
        compute: bool,
        queue: Arc<Queue>,
        exited: Arc<AtomicBool>,
    ) -> Result<Self> {
        let device = queue.device();
        let host_buffer = if let Some(memory_allocator) = memory_allocator {
            let buffer = CpuAccessibleBuffer::from_iter(
                memory_allocator,
                BufferUsage {
                    transfer_src: true,
                    transfer_dst: true,
                    ..Default::default()
                },
                true,
                (0..DeviceBuffer::HOST_BUFFER_SIZE).into_iter().map(|_| 0u8),
            )?;
            Some(buffer)
        } else {
            None
        };
        let command_pool = CommandPool::new(
            queue.device().clone(),
            CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                transient: true,
                reset_command_buffer: false,
                ..Default::default()
            },
        )?;
        let command_pool_alloc = command_pool
            .allocate_command_buffers(CommandBufferAllocateInfo {
                level: CommandBufferLevel::Primary,
                command_buffer_count: 1,
                ..Default::default()
            })?
            .next()
            .unwrap();
        let descriptor_pool = if compute {
            Some(DescriptorPool::new(
                device.clone(),
                DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_sizes: [(DescriptorType::StorageBuffer, 8)].into_iter().collect(),
                    ..Default::default()
                },
            )?)
        } else {
            None
        };
        let fence = Fence::new(
            device.clone(),
            vulkano::sync::FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )?;
        let state = Default::default();
        let guard = WorkerDropGuard { exited };
        Ok(Self {
            op_receiver,
            state,
            fence,
            command_pool,
            descriptor_pool,
            host_buffer,
            queue,
            guard,
        })
    }
    unsafe fn submit(&self, command_buffer: &UnsafeCommandBuffer) -> Result<()> {
        let queue = &self.queue;
        let device = queue.device();
        let command_buffers = &[command_buffer.handle()];
        let submit_info = ash::vk::SubmitInfo::builder().command_buffers(command_buffers);
        queue.with(|_| unsafe {
            (device.fns().v1_0.queue_submit)(
                queue.handle(),
                1,
                [submit_info].as_ptr() as _,
                self.fence.handle(),
            )
            .result()
        })?;
        Ok(())
    }
    fn run(&self) -> Result<()> {
        loop {
            let device = self.queue.device();
            unsafe {
                self.command_pool.reset(false)?;
            }
            let command_pool_alloc = self
                .command_pool
                .allocate_command_buffers(CommandBufferAllocateInfo {
                    level: CommandBufferLevel::Primary,
                    command_buffer_count: 1,
                    ..Default::default()
                })?
                .next()
                .unwrap();
            let mut builder = unsafe {
                UnsafeCommandBufferBuilder::new(
                    &command_pool_alloc,
                    CommandBufferBeginInfo {
                        usage: CommandBufferUsage::OneTimeSubmit,
                        ..Default::default()
                    },
                )?
            };
            self.fence.reset()?;
            let op = self.op_receiver.lock().recv()?;
            match op {
                Op::Upload {
                    sender,
                    dst,
                    submit,
                } => {
                    let buffer = self.host_buffer.as_ref().unwrap();
                    while Arc::strong_count(buffer) > 1 {
                        std::thread::sleep(Duration::from_micros(1));
                    }
                    sender.send((buffer.clone(), self.state.next())).unwrap();
                    unsafe {
                        builder.copy_buffer(&CopyBufferInfo::buffers(buffer.clone(), dst.clone()));
                    }
                    let command_buffer = builder.build()?;
                    while !submit.load(Ordering::Relaxed) {
                        if Arc::strong_count(&submit) == 1 {
                            return Ok(());
                        }
                        std::thread::sleep(Duration::from_micros(1));
                    }
                    unsafe {
                        self.submit(&command_buffer)?;
                    }
                    self.fence.wait(None)?;
                    self.state.finish();
                }
                Op::Download {
                    src,
                    dst_sender,
                    future,
                } => {
                    let buffer = self.host_buffer.as_ref().unwrap();
                    unsafe {
                        builder.copy_buffer(&CopyBufferInfo::buffers(src, buffer.clone()));
                    }
                    let command_buffer = builder.build()?;
                    if let Some(future) = future {
                        while !future.ready() {
                            if self.guard.exited.load(Ordering::Relaxed) {
                                return Ok(());
                            }
                            std::thread::sleep(Duration::from_micros(1));
                        }
                    }
                    unsafe {
                        self.submit(&command_buffer)?;
                    }
                    self.fence.wait(None)?;
                    let _ = dst_sender.send(buffer.clone());
                    while Arc::strong_count(buffer) > 1 {
                        std::thread::sleep(Duration::from_micros(1));
                    }
                }
                Op::Compute {
                    futures,
                    future_sender,
                    compute_pipeline,
                    buffers,
                    push_consts,
                    groups,
                } => {
                    let descriptor_pool = self.descriptor_pool.as_ref().unwrap();
                    unsafe {
                        builder.bind_pipeline_compute(&compute_pipeline);
                    }
                    let pipeline_layout = compute_pipeline.layout();
                    if !buffers.is_empty() {
                        let descriptor_set_layout = pipeline_layout.set_layouts().first().unwrap();
                        // TODO Push descriptor
                        let mut descriptor_set = unsafe {
                            descriptor_pool
                                .allocate_descriptor_sets([DescriptorSetAllocateInfo {
                                    layout: descriptor_set_layout,
                                    variable_descriptor_count: 0,
                                }])?
                                .next()
                                .unwrap()
                        };
                        let buffer_iter = buffers
                            .iter()
                            .map(|x| -> Arc<dyn BufferAccess> { x.clone() });
                        unsafe {
                            descriptor_set.write(
                                descriptor_set_layout,
                                &[WriteDescriptorSet::buffer_array(0, 0, buffer_iter)],
                            );
                        }
                        unsafe {
                            builder.bind_descriptor_sets(
                                PipelineBindPoint::Compute,
                                pipeline_layout,
                                0,
                                &[descriptor_set],
                                [],
                            );
                        }
                    }
                    if !push_consts.is_empty() {
                        unsafe {
                            builder.push_constants(
                                pipeline_layout,
                                ShaderStages::compute(),
                                0,
                                push_consts.len() as u32,
                                push_consts.as_slice(),
                            );
                        }
                    }
                    unsafe {
                        builder.dispatch(groups);
                    }
                    let command_buffer = builder.build()?;
                    for future in futures.iter() {
                        while !future.ready() {
                            if self.guard.exited.load(Ordering::SeqCst) {
                                anyhow::bail!("Exited while waiting for compute!");
                            }
                            std::thread::sleep(Duration::from_micros(1));
                        }
                    }
                    unsafe {
                        self.submit(&command_buffer)?;
                    }
                    let _ = future_sender.send(self.state.next());
                    self.fence.wait(None)?;
                    self.state.finish();
                    unsafe {
                        descriptor_pool.reset()?;
                    }
                }
            }
        }
    }
}

struct WorkerDropGuard {
    exited: Arc<AtomicBool>,
}

impl Drop for WorkerDropGuard {
    fn drop(&mut self) {
        self.exited.store(true, Ordering::Relaxed);
    }
}

enum Op {
    Upload {
        sender: Sender<(Arc<CpuAccessibleBuffer<[u8]>>, WorkerFuture)>,
        dst: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
        submit: Arc<AtomicBool>,
    },
    Download {
        src: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
        dst_sender: Sender<Arc<CpuAccessibleBuffer<[u8]>>>,
        future: Option<WorkerFuture>,
    },
    Compute {
        futures: Vec<WorkerFuture>,
        compute_pipeline: Arc<ComputePipeline>,
        buffers: Vec<Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>>,
        push_consts: Vec<u8>,
        groups: [u32; 3],
        future_sender: Sender<WorkerFuture>,
    },
}

const fn aligned_floor(x: usize, align: usize) -> usize {
    if x % align != 0 {
        align * (x / align)
    } else {
        x
    }
}

const fn aligned_ceil(x: usize, align: usize) -> usize {
    if x % align != 0 {
        aligned_floor(x, align) + align
    } else {
        x
    }
}

enum WorkerFutureGuard<'a> {
    UpgradableRead(RwLockUpgradableReadGuard<'a, WorkerFuture>),
    Read(RwLockReadGuard<'a, WorkerFuture>),
}

impl Deref for WorkerFutureGuard<'_> {
    type Target = WorkerFuture;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::UpgradableRead(x) => &*x,
            Self::Read(x) => &*x,
        }
    }
}

pub(super) struct DeviceBuffer {
    inner: Option<Arc<DeviceLocalBuffer<[u8]>>>,
    engine: Arc<Engine>,
    offset: usize,
    len: usize,
    future: Arc<RwLock<WorkerFuture>>,
}

impl DeviceBuffer {
    const MAX_LEN: usize = i32::MAX as usize;
    const MAX_SIZE: usize = aligned_ceil(Self::MAX_LEN, Self::ALIGN);
    const ALIGN: usize = 256;
    const HOST_BUFFER_SIZE: usize = 128_000_000; // 32_000_000;
    fn chunk_size(size: usize) -> usize {
        Self::HOST_BUFFER_SIZE
    }
    fn host_visible(&self) -> bool {
        use vulkano::buffer::sys::BufferMemory;
        if let Some(inner) = self.inner.as_ref() {
            if let BufferMemory::Normal(memory_alloc) = inner.inner().buffer.memory() {
                return memory_alloc.mapped_ptr().is_some();
            }
        }
        false
    }
    fn write(&mut self, data: &[u8]) -> Result<()> {
        let engine = &self.engine;
        let device = &engine.device;
        if let Some(buffer) = self.inner.as_ref() {
            if let Ok(mut mapped) = buffer.inner().buffer.write(0..data.len() as _) {
                mapped[..data.len()].copy_from_slice(data);
            } else {
                let mut offset = 0;
                let device_lost = DeviceLost {
                    index: engine.info.index,
                    handle: engine.handle(),
                };
                let mut future_guard = self.future.write();
                for data in data.chunks(Self::chunk_size(data.len())) {
                    let (sender, receiver) = crossbeam_channel::bounded(1);
                    let dst = buffer
                        .slice(offset as _..(offset + data.len()) as _)
                        .unwrap();
                    let submit = Arc::new(AtomicBool::default());
                    let op = Op::Upload {
                        sender,
                        dst,
                        submit: submit.clone(),
                    };
                    engine
                        .transfer_op_sender
                        .send(op)
                        .map_err(|_| device_lost)?;
                    let (src, future) = receiver.recv().map_err(|_| device_lost)?;
                    src.write().unwrap()[..data.len()].copy_from_slice(data);
                    submit.store(true, Ordering::Relaxed);
                    std::mem::drop(src);
                    *future_guard = future;
                    offset += data.len();
                }
            }
        }
        Ok(())
    }
}

impl DeviceEngineBuffer for DeviceBuffer {
    type Engine = Engine;
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    unsafe fn uninit(engine: Arc<Engine>, len: usize) -> Result<Self> {
        let inner = if len > 0 {
            let len = aligned_ceil(len, Self::ALIGN);
            let usage = BufferUsage {
                storage_buffer: true,
                transfer_dst: true,
                transfer_src: true,
                ..Default::default()
            };
            let inner = DeviceLocalBuffer::array(
                &engine.memory_allocator,
                len as _,
                usage,
                engine.compute_families.iter().copied(),
            )?;
            Some(inner)
        } else {
            None
        };
        Ok(Self {
            inner,
            engine,
            offset: 0,
            len,
            future: Arc::default(),
        })
    }
    fn upload(engine: Arc<Self::Engine>, data: &[u8]) -> Result<Self> {
        let mut buffer = unsafe { Self::uninit(engine.clone(), data.len())? };
        buffer.write(data)?;
        Ok(buffer)
    }
    fn download(&self, data: &mut [u8]) -> Result<()> {
        if let Some(buffer) = self.inner.as_ref() {
            let engine = &self.engine;
            let device = &engine.device;
            let device_lost = DeviceLost {
                index: engine.info.index,
                handle: engine.handle(),
            };
            let prev_future = self.future.read();
            let buffer_inner = buffer.inner();
            if self.host_visible() {
                while !prev_future.ready() {
                    if engine.exited.load(Ordering::SeqCst) {
                        return Err(device_lost.into());
                    }
                    std::thread::sleep(Duration::from_micros(1));
                }
                loop {
                    if let Ok(mapped) = buffer_inner
                        .buffer
                        .read(buffer_inner.offset..buffer_inner.offset + data.len() as u64)
                    {
                        data.copy_from_slice(&mapped[..data.len()]);
                        return Ok(());
                    } else {
                        std::thread::sleep(Duration::from_micros(1));
                    }
                }
            }
            let mut prev_future = Some(prev_future.clone());
            let mut offset = self.offset;
            struct HostCopy<'a> {
                data: &'a mut [u8],
                dst_receiver: Receiver<Arc<CpuAccessibleBuffer<[u8]>>>,
            }
            let mut host_copy: Option<HostCopy> = None;
            for data in data.chunks_mut(Self::chunk_size(data.len())) {
                let src = buffer
                    .slice(offset as _..(offset + data.len()) as _)
                    .unwrap();
                offset += data.len();
                let (dst_sender, dst_receiver) = crossbeam_channel::bounded(1);
                let future = prev_future.take();
                let op = Op::Download {
                    src,
                    dst_sender,
                    future: future.clone(),
                };
                engine
                    .transfer_op_sender
                    .send(op)
                    .map_err(|_| device_lost)?;
                if let Some(future) = prev_future.take() {
                    while !future.ready() {
                        if engine.exited.load(Ordering::SeqCst) {
                            return Err(device_lost.into());
                        }
                        std::thread::sleep(Duration::from_micros(1));
                    }
                }
                let host_copy = host_copy.replace(HostCopy { data, dst_receiver });
                if let Some(host_copy) = host_copy {
                    let dst = host_copy.dst_receiver.recv().map_err(|_| device_lost)?;
                    host_copy
                        .data
                        .copy_from_slice(&dst.read().unwrap()[..host_copy.data.len()]);
                }
            }
            if let Some(host_copy) = host_copy {
                let dst = host_copy.dst_receiver.recv().map_err(|_| device_lost)?;
                host_copy
                    .data
                    .copy_from_slice(&dst.read().unwrap()[..host_copy.data.len()]);
            }
        }
        Ok(())
    }
    fn transfer(&self, engine: Arc<Self::Engine>) -> Result<Self> {
        let mut output = unsafe { Self::uninit(engine, self.len)? };
        if output.len == 0 {
            return Ok(output);
        }
        let buffer1 = self.inner.as_ref().unwrap();
        let engine1 = &self.engine;
        let device1 = &engine1.device;
        let device_lost1 = DeviceLost {
            index: engine1.info.index,
            handle: engine1.handle(),
        };
        let buffer2 = output.inner.as_ref().unwrap();
        let engine2 = &output.engine;
        let device2 = &engine2.device;
        let device_lost2 = DeviceLost {
            index: engine2.info.index,
            handle: engine2.handle(),
        };
        let prev_future = self.future.read();
        let buffer_inner1 = buffer1.inner();
        let buffer_inner2 = buffer2.inner();
        if self.host_visible() {
            while !prev_future.ready() {
                if engine1.exited.load(Ordering::SeqCst) {
                    return Err(device_lost1.into());
                }
                std::thread::sleep(Duration::from_micros(1));
            }
            loop {
                if let Ok(mapped1) = buffer_inner1
                    .buffer
                    .read(buffer_inner1.offset..buffer_inner1.offset + self.len() as u64)
                {
                    if output.host_visible() {
                        let mut mapped2 =
                            buffer_inner2.buffer.write(0..output.len() as u64).unwrap();
                        mapped2.copy_from_slice(&mapped1[..output.len()]);
                    } else {
                        output.write(&mapped1)?;
                    }
                    return Ok(output);
                } else {
                    std::thread::sleep(Duration::from_micros(1));
                }
            }
        } else if output.host_visible() {
            {
                let mut mapped2 = buffer_inner2.buffer.write(0..output.len() as u64).unwrap();
                self.download(&mut mapped2)?;
            }
            return Ok(output);
        }
        let mut prev_future = Some(prev_future.clone());
        let mut offset1 = self.offset;
        let mut offset2 = 0;
        struct HostCopy {
            chunk_size: usize,
            receiver: Receiver<(Arc<CpuAccessibleBuffer<[u8]>>, WorkerFuture)>,
            dst_receiver: Receiver<Arc<CpuAccessibleBuffer<[u8]>>>,
            submit: Arc<AtomicBool>,
        }
        let mut host_copy = None;
        while offset2 < output.len() {
            let chunk_size = output
                .len()
                .checked_sub(offset2)
                .unwrap()
                .min(Self::HOST_BUFFER_SIZE);
            let src = buffer1
                .slice(offset1 as _..(offset1 + chunk_size) as _)
                .unwrap();
            let dst = buffer2
                .slice(offset2 as _..(offset2 + chunk_size) as _)
                .unwrap();
            offset1 += chunk_size;
            offset2 += chunk_size;
            let (dst_sender, dst_receiver) = crossbeam_channel::bounded(1);
            let future = prev_future.take();
            let op = Op::Download {
                src,
                dst_sender,
                future: future.clone(),
            };
            engine1
                .transfer_op_sender
                .send(op)
                .map_err(|_| device_lost2)?;
            if let Some(future) = future {
                while !future.ready() {
                    if engine1.exited.load(Ordering::SeqCst) {
                        return Err(device_lost2.into());
                    }
                    std::thread::sleep(Duration::from_micros(1));
                }
            }
            let (sender, receiver) = crossbeam_channel::bounded(0);
            let submit = Arc::new(AtomicBool::default());
            let op = Op::Upload {
                sender,
                dst,
                submit: submit.clone(),
            };
            engine2
                .transfer_op_sender
                .send(op)
                .map_err(|_| device_lost2)?;
            let host_copy = host_copy.replace(HostCopy {
                chunk_size,
                receiver,
                dst_receiver,
                submit,
            });
            if let Some(host_copy) = host_copy {
                let dst = host_copy.dst_receiver.recv().map_err(|_| device_lost1)?;
                let (src, _future) = host_copy.receiver.recv().map_err(|_| device_lost2)?;
                src.write().unwrap()[..host_copy.chunk_size]
                    .copy_from_slice(&dst.read().unwrap()[..host_copy.chunk_size]);
                host_copy.submit.store(true, Ordering::Relaxed);
            }
        }
        if let Some(host_copy) = host_copy {
            let dst = host_copy.dst_receiver.recv().map_err(|_| device_lost1)?;
            let (src, future) = host_copy.receiver.recv().map_err(|_| device_lost2)?;
            src.write().unwrap()[..host_copy.chunk_size]
                .copy_from_slice(&dst.read().unwrap()[..host_copy.chunk_size]);
            host_copy.submit.store(true, Ordering::Relaxed);
            *output.future.write() = future;
        }
        Ok(output)
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn len(&self) -> usize {
        self.len
    }
    fn slice(self: &Arc<Self>, bounds: impl RangeBounds<usize>) -> Option<Arc<Self>> {
        todo!()
    }
}

#[derive(Clone)]
struct KernelInner {
    desc: Arc<KernelDesc>,
    compute_pipeline: Arc<ComputePipeline>,
}

impl KernelInner {
    fn new(engine: &Arc<Engine>, desc: Arc<KernelDesc>) -> Result<Self> {
        use vulkano::{
            descriptor_set::layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
            pipeline::layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
            shader::{spirv::ExecutionModel, EntryPointInfo},
        };
        let device = &engine.device;
        let stages = ShaderStages {
            compute: true,
            ..ShaderStages::none()
        };
        let descriptor_requirements = desc
            .slice_descs
            .iter()
            .enumerate()
            .map(|(i, desc)| {
                let set = 0u32;
                let binding = i as u32;
                let storage_write = if desc.mutable { Some(binding) } else { None };
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
        let push_consts_range = desc.push_consts_range();
        let push_constant_range = if push_consts_range > 0 {
            Some(PushConstantRange {
                stages,
                offset: 0,
                size: push_consts_range,
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
                &desc.spirv,
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
        let bindings = (0..desc.slice_descs.len())
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
            device.clone(),
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
}

pub(super) struct Kernel {
    engine: Arc<Engine>,
    desc: Arc<KernelDesc>,
    compute_pipeline: Arc<ComputePipeline>,
}

impl DeviceEngineKernel for Kernel {
    type Engine = Engine;
    type DeviceBuffer = DeviceBuffer;
    fn cached(
        engine: Arc<Self::Engine>,
        key: KernelKey,
        desc_fn: impl FnOnce() -> Result<Arc<KernelDesc>>,
    ) -> Result<Arc<Self>> {
        let KernelInner {
            desc,
            compute_pipeline,
        } = engine
            .kernels
            .entry(key)
            .or_try_insert_with(|| KernelInner::new(&engine, desc_fn()?))?
            .clone();
        Ok(Arc::new(Kernel {
            engine,
            desc,
            compute_pipeline,
        }))
    }
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    unsafe fn dispatch(
        &self,
        groups: [u32; 3],
        buffers: &[Arc<Self::DeviceBuffer>],
        mut push_consts: Vec<u8>,
    ) -> Result<()> {
        let engine = &self.engine;
        let device = &engine.device;
        while push_consts.len() % 4 != 0 {
            push_consts.push(0);
        }
        for (buffer, slice_desc) in buffers.iter().zip(self.desc.slice_descs.iter()) {
            push_consts.extend((buffer.offset as u32).to_ne_bytes());
            let len = buffer.len / slice_desc.scalar_type.size();
            debug_assert_ne!(len, 0);
            push_consts.extend((len as u32).to_ne_bytes());
        }
        let mut futures: Vec<_> = buffers
            .iter()
            .map(|x| x.future.clone())
            .zip(self.desc.slice_descs.iter().map(|x| x.mutable))
            .collect();
        futures.sort_by_key(|(x, _)| Arc::as_ptr(x) as usize);
        let future_guards: Vec<_> = futures
            .iter()
            .map(|(x, mutable)| {
                if *mutable {
                    WorkerFutureGuard::UpgradableRead(x.upgradable_read())
                } else {
                    WorkerFutureGuard::Read(x.read())
                }
            })
            .collect();
        let futures = future_guards.iter().map(|x| x.deref().clone()).collect();
        let buffers = buffers
            .iter()
            .map(|x| x.inner.as_ref().unwrap().into_buffer_slice())
            .collect();
        let (future_sender, future_receiver) = crossbeam_channel::bounded(0);
        let device_lost = DeviceLost {
            index: engine.info.index,
            handle: engine.handle(),
        };
        let op = Op::Compute {
            futures,
            compute_pipeline: self.compute_pipeline.clone(),
            buffers,
            push_consts,
            groups,
            future_sender,
        };
        engine.compute_op_sender.send(op).map_err(|_| device_lost)?;
        let future = future_receiver.recv().map_err(|_| device_lost)?;
        for guard in future_guards {
            match guard {
                WorkerFutureGuard::UpgradableRead(x) => {
                    *RwLockUpgradableReadGuard::upgrade(x) = future.clone();
                }
                WorkerFutureGuard::Read(_) => (),
            }
        }
        Ok(())
    }
    fn desc(&self) -> &Arc<KernelDesc> {
        &self.desc
    }
}
