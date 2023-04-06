use super::{
    DeviceEngine, DeviceEngineBuffer, DeviceEngineKernel, DeviceInfo, DeviceLost, DeviceOptions,
    Features, KernelDesc, KernelKey,
};

use anyhow::Result;
use ash::vk::Handle;
use atomicbox::AtomicOptionBox;
use dashmap::DashMap;
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard};
use std::{
    ops::{Deref, Range},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Weak,
    },
    thread::Thread,
    time::{Duration, Instant},
};
use vulkano::{
    buffer::{
        sys::Buffer, BufferAccess, BufferInner, BufferSlice, BufferUsage, CpuAccessibleBuffer,
        DeviceLocalBuffer,
    },
    command_buffer::{
        pool::{CommandBufferAllocateInfo, CommandPool, CommandPoolCreateInfo},
        sys::{CommandBufferBeginInfo, UnsafeCommandBuffer, UnsafeCommandBufferBuilder},
        CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
    },
    descriptor_set::{
        layout::{DescriptorSetLayout, DescriptorType},
        pool::{DescriptorPool, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo},
        WriteDescriptorSet,
    },
    device::{Device, DeviceCreateInfo, DeviceOwned, Queue, QueueCreateInfo},
    instance::{Instance, InstanceCreateInfo, Version},
    library::{LoadingError, VulkanLibrary},
    memory::allocator::{GenericMemoryAllocatorCreateInfo, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::{
        DescriptorRequirements, ShaderExecution, ShaderInterface, ShaderModule, ShaderStages,
    },
    sync::Fence,
    VulkanObject,
};

/*#[cfg(any(target_os = "ios", target_os = "macos"))]
struct MoltenLoader;

#[cfg(any(target_os = "ios", target_os = "macos"))]
unsafe impl vulkano::library::Loader for MoltenLoader {
    unsafe fn get_instance_proc_addr(
        &self,
        instance: ash::vk::Instance,
        name: *const std::os::raw::c_char,
    ) -> ash::vk::PFN_vkVoidFunction {
        unsafe { ash_molten::load().get_instance_proc_addr(instance, name) }
    }
}*/

fn vulkan_library() -> Result<Arc<VulkanLibrary>, LoadingError> {
    /*#[cfg(target_os = "ios")]
    {
        VulkanLibrary::with_loader(MoltenLoader)
    }
    #[cfg(target_os = "macos")]
    {
        match VulkanLibrary::new() {
            Err(LoadingError::LibraryLoadFailure(_)) => VulkanLibrary::with_loader(MoltenLoader),
            result => result,
        }
    }
    #[cfg(not(any(target_os = "ios", target_os = "macos")))]*/
    {
        VulkanLibrary::new()
    }
}

pub struct Engine {
    info: Arc<DeviceInfo>,
    compute_families: Vec<u32>,
    compute: Mutex<()>,
    transfer: Mutex<()>,
    workers: Vec<[Worker; 2]>,
    exited: Arc<AtomicBool>,
    kernels: DashMap<KernelKey, KernelInner>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    device: Arc<Device>,
    _instance: Arc<Instance>,
}

impl Engine {
    fn wait_for_future(&self, future: &WorkerFuture) -> Result<(), DeviceLost> {
        while !future.ready() {
            if self.exited.load(Ordering::SeqCst) {
                return Err(DeviceLost {
                    index: self.info.index,
                    handle: self.handle(),
                });
            }
            std::thread::yield_now();
        }
        Ok(())
    }
    fn transfer_workers(&self) -> (MutexGuard<()>, [&Worker; 2]) {
        let transfer = self.transfer.lock();
        let [worker1, worker2] = self.workers.last().unwrap();
        let workers = if worker1.ready() {
            [worker1, worker2]
        } else {
            [worker2, worker1]
        };
        (transfer, workers)
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
        let library = vulkan_library()?;
        let instance = Instance::new(library, InstanceCreateInfo::application_from_cargo_toml())?;
        let physical_devices = instance.enumerate_physical_devices()?;
        let devices = physical_devices.len();
        let physical_device = if let Some(physical_device) = physical_devices.skip(index).next() {
            physical_device
        } else {
            return Err(super::DeviceIndexOutOfRange { index, devices }.into());
        };
        let name = physical_device.properties().device_name.clone();
        let optimal_device_extensions = vulkano::device::DeviceExtensions {
            khr_vulkan_memory_model: true,
            ..vulkano::device::DeviceExtensions::empty()
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
        compute_families.sort_by_key(|(_, flags)| flags.graphics);
        let mut compute_families: Vec<u32> = compute_families.iter().map(|(i, _)| *i).collect();
        let transfer_family = physical_device
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
                    (0, 64_000_000),
                    (DeviceBuffer::MAX_SIZE as _, DeviceBuffer::MAX_SIZE as _),
                ],
                dedicated_allocation: false,
                ..Default::default()
            },
        )?);
        let exited = Arc::new(AtomicBool::default());
        let compute_queues: Vec<_> = queues.by_ref().take(compute_families.len()).collect();
        let mut workers = Vec::with_capacity(queues.len());
        for queue in compute_queues {
            let memory_allocator = if transfer_family.is_none() {
                Some(&memory_allocator)
            } else {
                None
            };
            let worker1 = Worker::new(memory_allocator, true, queue.clone(), exited.clone())?;
            let worker2 = Worker::new(memory_allocator, true, queue.clone(), exited.clone())?;
            workers.push([worker1, worker2]);
        }
        let transfer_queue = queues.next();
        if let Some(queue) = transfer_queue {
            let worker1 =
                Worker::new(Some(&memory_allocator), true, queue.clone(), exited.clone())?;
            let worker2 = Worker::new(Some(&memory_allocator), true, queue, exited.clone())?;
            workers.push([worker1, worker2]);
        }
        let kernels = DashMap::default();
        let info = Arc::new(DeviceInfo {
            index,
            name,
            compute_queues: compute_families.len(),
            transfer_queues: transfer_family.is_some() as usize,
            features,
        });
        let compute = Mutex::default();
        let transfer = Mutex::default();
        Ok(Arc::new(Self {
            info,
            compute_families,
            compute,
            transfer,
            workers,
            exited,
            kernels,
            memory_allocator,
            device,
            _instance: instance,
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
            .workers
            .iter()
            .flat_map(|[w1, w2]| [&w1.state, &w2.state])
            .map(|state| state.pending.load(Ordering::SeqCst))
            .collect();
        loop {
            if self.exited.load(Ordering::SeqCst) {
                return Err(DeviceLost {
                    index: self.info.index,
                    handle: self.handle(),
                });
            }
            if self
                .workers
                .iter()
                .flat_map(|[w1, w2]| [&w1.state, &w2.state])
                .zip(pending.iter().copied())
                .all(|(state, pending)| state.completed.load(Ordering::SeqCst) >= pending)
            {
                return Ok(());
            }
            std::thread::yield_now();
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
    fn ready(&self) -> bool {
        self.pending.load(Ordering::SeqCst) == self.completed.load(Ordering::SeqCst)
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
    op_slot: Arc<AtomicOptionBox<Op>>,
    ready: Arc<AtomicBool>,
    state: WorkerState,
    host_buffer: Option<Arc<CpuAccessibleBuffer<[u8]>>>,
    thread: Thread,
}

impl Worker {
    fn new(
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
                reset_command_buffer: true,
                ..Default::default()
            },
        )?;
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
        let ready = Arc::new(AtomicBool::default());
        let state = WorkerState::default();
        let guard = WorkerDropGuard { exited };
        let op_slot = Arc::new(AtomicOptionBox::<Op>::none());
        let thread = {
            let ready = ready.clone();
            let op_slot = op_slot.clone();
            let queue = queue.clone();
            let state = state.clone();
            let host_buffer = host_buffer.clone();
            std::thread::spawn(move || {
                let _guard = guard;
                let command_pool_alloc = command_pool
                    .allocate_command_buffers(CommandBufferAllocateInfo {
                        level: CommandBufferLevel::Primary,
                        command_buffer_count: 1,
                        ..Default::default()
                    })
                    .unwrap()
                    .next()
                    .unwrap();
                let mut last_msg = Instant::now();
                loop {
                    let device = queue.device();
                    unsafe {
                        (device.fns().v1_0.reset_command_buffer)(
                            command_pool_alloc.handle(),
                            Default::default(),
                        )
                        .result()
                        .unwrap();
                    }
                    let mut builder = unsafe {
                        UnsafeCommandBufferBuilder::new(
                            &command_pool_alloc,
                            CommandBufferBeginInfo {
                                usage: CommandBufferUsage::OneTimeSubmit,
                                ..Default::default()
                            },
                        )
                        .unwrap()
                    };
                    fence.reset().unwrap();
                    ready.store(true, Ordering::SeqCst);
                    let op = loop {
                        if Arc::strong_count(&op_slot) == 1 {
                            return;
                        }
                        if let Some(op) = op_slot.take(Ordering::SeqCst) {
                            last_msg = Instant::now();
                            break *op;
                        } else if last_msg.elapsed().as_millis() > 400 {
                            std::thread::park();
                        }
                    };
                    match op {
                        Op::Upload { dst, submit } => {
                            let buffer = host_buffer.as_ref().unwrap();
                            unsafe {
                                builder.copy_buffer(&CopyBufferInfo::buffers(
                                    buffer.clone(),
                                    dst.clone(),
                                ));
                            }
                            let command_buffer = builder.build().unwrap();
                            while !submit.load(Ordering::Relaxed) {
                                if Arc::strong_count(&submit) == 1 && !submit.load(Ordering::SeqCst)
                                {
                                    return;
                                }
                            }
                            unsafe {
                                Worker::submit(&queue, &command_buffer, &fence).unwrap();
                            }
                            fence.wait(None).unwrap();
                            state.finish();
                        }
                        Op::Download { src, submit } => {
                            let buffer = host_buffer.as_ref().unwrap();
                            unsafe {
                                builder.copy_buffer(&CopyBufferInfo::buffers(src, buffer.clone()));
                            }
                            let command_buffer = builder.build().unwrap();
                            while !submit.load(Ordering::Relaxed) {
                                if Arc::strong_count(&submit) == 1 && !submit.load(Ordering::SeqCst)
                                {
                                    return;
                                }
                            }
                            unsafe {
                                Worker::submit(&queue, &command_buffer, &fence).unwrap();
                            }
                            fence.wait(None).unwrap();
                            state.finish();
                        }
                        Op::Compute {
                            compute_pipeline,
                            buffers,
                            push_consts,
                            groups,
                            submit,
                        } => {
                            let descriptor_pool = descriptor_pool.as_ref().unwrap();
                            unsafe {
                                builder.bind_pipeline_compute(&compute_pipeline);
                            }
                            let pipeline_layout = compute_pipeline.layout();
                            if !buffers.is_empty() {
                                let descriptor_set_layout =
                                    pipeline_layout.set_layouts().first().unwrap();
                                // TODO Push descriptor
                                let mut descriptor_set = unsafe {
                                    descriptor_pool
                                        .allocate_descriptor_sets([DescriptorSetAllocateInfo {
                                            layout: descriptor_set_layout,
                                            variable_descriptor_count: 0,
                                        }])
                                        .unwrap()
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
                            let command_buffer = builder.build().unwrap();
                            while !submit.load(Ordering::Relaxed) {
                                if Arc::strong_count(&submit) == 1 && !submit.load(Ordering::SeqCst)
                                {
                                    return;
                                }
                                std::thread::sleep(Duration::from_nanos(100));
                            }
                            unsafe {
                                Worker::submit(&queue, &command_buffer, &fence).unwrap();
                            }
                            fence.wait(None).unwrap();
                            state.finish();
                            unsafe {
                                descriptor_pool.reset().unwrap();
                            }
                        }
                    }
                }
            })
            .thread()
            .clone()
        };
        Ok(Self {
            ready,
            op_slot,
            state,
            host_buffer,
            thread,
        })
    }
    unsafe fn submit(
        queue: &Arc<Queue>,
        command_buffer: &UnsafeCommandBuffer,
        fence: &Fence,
    ) -> Result<()> {
        let device = queue.device();
        let command_buffers = &[command_buffer.handle()];
        let submit_info = ash::vk::SubmitInfo::builder().command_buffers(command_buffers);
        queue.with(|_| unsafe {
            (device.fns().v1_0.queue_submit)(
                queue.handle(),
                1,
                [submit_info].as_ptr() as _,
                fence.handle(),
            )
            .result()
        })?;
        Ok(())
    }
    fn send(&self, op: Op) -> Result<WorkerFuture, ()> {
        self.wait()?;
        let future = self.state.next();
        self.ready.store(false, Ordering::SeqCst);
        self.op_slot.store(Some(op.into()), Ordering::SeqCst);
        self.thread.unpark();
        Ok(future)
    }
    fn ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }
    fn wait(&self) -> Result<(), ()> {
        while !self.state.ready() {
            if Arc::strong_count(&self.op_slot) == 1 {
                return Err(());
            }
        }
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let op_slot = Arc::downgrade(&std::mem::take(&mut self.op_slot));
        self.thread.unpark();
        while Weak::strong_count(&op_slot) > 0 {
            std::thread::sleep(Duration::from_micros(1));
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
        dst: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
        submit: Arc<AtomicBool>,
    },
    Download {
        src: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
        submit: Arc<AtomicBool>,
    },
    Compute {
        compute_pipeline: Arc<ComputePipeline>,
        buffers: Vec<Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>>,
        push_consts: Vec<u8>,
        groups: [u32; 3],
        submit: Arc<AtomicBool>,
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

#[derive(Clone)]
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
    const HOST_BUFFER_SIZE: usize = 32_000_000;
    fn host_visible(&self) -> bool {
        use vulkano::buffer::sys::BufferMemory;
        if let Some(inner) = self.inner.as_ref() {
            if let BufferMemory::Normal(memory_alloc) = inner.inner().buffer.memory() {
                return memory_alloc.mapped_ptr().is_some();
            }
        }
        false
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
    fn upload(&self, data: &[u8]) -> Result<()> {
        let engine = &self.engine;
        let mut future_guard = self.future.write();
        if let Some(buffer) = self.inner.as_ref() {
            if let Ok(mut mapped) = buffer.inner().buffer.write(0..data.len() as _) {
                engine.wait_for_future(&future_guard)?;
                mapped[..data.len()].copy_from_slice(data);
            } else {
                let (_transfer, workers) = engine.transfer_workers();
                let mut offset = self.offset;
                let device_lost = DeviceLost {
                    index: engine.info.index,
                    handle: engine.handle(),
                };
                let mut prev_future = Some(std::mem::take(&mut *future_guard));
                for (data, worker) in data
                    .chunks(Self::HOST_BUFFER_SIZE)
                    .zip(workers.into_iter().cycle())
                {
                    let src = worker.host_buffer.as_ref().unwrap();
                    let dst = buffer
                        .slice(offset as _..(offset + data.len()) as _)
                        .unwrap();
                    let submit = Arc::new(AtomicBool::default());
                    let op = Op::Upload {
                        dst,
                        submit: submit.clone(),
                    };
                    let future = worker.send(op).map_err(|_| device_lost)?;
                    if let Some(prev_future) = prev_future.take() {
                        engine.wait_for_future(&prev_future)?;
                    }
                    src.write().unwrap()[..data.len()].copy_from_slice(data);
                    submit.store(true, Ordering::Relaxed);
                    *future_guard = future;
                    offset += data.len();
                }
            }
        }
        Ok(())
    }
    fn download(&self, data: &mut [u8]) -> Result<()> {
        if let Some(buffer) = self.inner.as_ref() {
            let engine = &self.engine;
            let device_lost = DeviceLost {
                index: engine.info.index,
                handle: engine.handle(),
            };
            let prev_future = self.future.read();
            let buffer_inner = buffer.inner();
            if self.host_visible() {
                engine.wait_for_future(&prev_future)?;
                let mapped = buffer_inner
                    .buffer
                    .read(buffer_inner.offset..buffer_inner.offset + data.len() as u64)
                    .unwrap();
                data.copy_from_slice(&mapped[..data.len()]);
                return Ok(());
            }
            let (_transfer, workers) = engine.transfer_workers();
            let mut prev_future = Some(prev_future.clone());
            let mut offset = self.offset;
            struct HostCopy<'a> {
                data: &'a mut [u8],
                worker: &'a Worker,
            }
            impl HostCopy<'_> {
                fn finish(self) -> Result<(), ()> {
                    let worker = self.worker;
                    worker.wait()?;
                    let data = self.data;
                    let dst = worker.host_buffer.as_ref().unwrap();
                    data.copy_from_slice(&dst.read().unwrap()[..data.len()]);
                    Ok(())
                }
            }

            let mut host_copy: Option<HostCopy> = None;
            for (data, worker) in data
                .chunks_mut(Self::HOST_BUFFER_SIZE)
                .zip(workers.into_iter().cycle())
            {
                let src = buffer
                    .slice(offset as _..(offset + data.len()) as _)
                    .unwrap();
                offset += data.len();
                let submit = Arc::new(AtomicBool::default());
                let op = Op::Download {
                    src,
                    submit: submit.clone(),
                };
                worker.send(op).map_err(|_| device_lost)?;
                if let Some(future) = prev_future.take() {
                    engine.wait_for_future(&future)?;
                }
                submit.store(true, Ordering::SeqCst);
                let host_copy = host_copy.replace(HostCopy { data, worker });
                if let Some(host_copy) = host_copy {
                    host_copy.finish().map_err(|_| device_lost)?;
                }
            }
            if let Some(host_copy) = host_copy {
                host_copy.finish().map_err(|_| device_lost)?;
            }
        }
        Ok(())
    }
    fn transfer(&self, dst: &Self) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        let output = dst;
        let buffer1 = self.inner.as_ref().unwrap();
        let engine1 = &self.engine;
        let device_lost1 = DeviceLost {
            index: engine1.info.index,
            handle: engine1.handle(),
        };
        let buffer2 = output.inner.as_ref().unwrap();
        let engine2 = &output.engine;
        let device_lost2 = DeviceLost {
            index: engine2.info.index,
            handle: engine2.handle(),
        };
        let prev_future = self.future.read();
        let buffer_inner1 = buffer1.inner();
        let buffer_inner2 = buffer2.inner();
        if self.host_visible() {
            engine1.wait_for_future(&prev_future)?;
            loop {
                let mapped1 = buffer_inner1
                    .buffer
                    .read(buffer_inner1.offset..buffer_inner1.offset + self.len() as u64)
                    .unwrap();
                if dst.host_visible() {
                    let mut mapped2 = buffer_inner2.buffer.write(0..output.len() as u64).unwrap();
                    mapped2.copy_from_slice(&mapped1[..output.len()]);
                } else {
                    output.upload(&mapped1)?;
                }
                return Ok(());
            }
        } else if output.host_visible() {
            let mut mapped2 = buffer_inner2.buffer.write(0..output.len() as u64).unwrap();
            self.download(&mut mapped2)?;
            return Ok(());
        }
        let mut prev_future = Some(prev_future.clone());
        let mut offset1 = self.offset;
        let mut offset2 = 0;

        struct HostCopy<'a> {
            size: usize,
            download_worker: &'a Worker,
            upload_worker: &'a Worker,
            submit: Arc<AtomicBool>,
            future: WorkerFuture,
        }

        impl HostCopy<'_> {
            fn finish(self) -> Result<WorkerFuture, ()> {
                self.download_worker.wait()?;
                let src = self
                    .download_worker
                    .host_buffer
                    .as_ref()
                    .unwrap()
                    .read()
                    .unwrap();
                self.upload_worker
                    .host_buffer
                    .as_ref()
                    .unwrap()
                    .write()
                    .unwrap()[..self.size]
                    .copy_from_slice(&src[..self.size]);
                self.submit.store(true, Ordering::SeqCst);
                Ok(self.future)
            }
        }

        let (_transfer1, workers1) = engine1.transfer_workers();
        let mut workers1 = workers1.into_iter().cycle();
        let (_transfer2, workers2) = engine2.transfer_workers();
        let mut workers2 = workers2.into_iter().cycle();
        let mut host_copy = None;
        while offset2 < output.len() {
            let download_worker = workers1.next().unwrap();
            let upload_worker = workers2.next().unwrap();
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
            let submit = Arc::new(AtomicBool::default());
            let op = Op::Download {
                src,
                submit: submit.clone(),
            };
            download_worker.send(op).map_err(|_| device_lost2)?;
            if let Some(future) = prev_future.take() {
                engine1.wait_for_future(&future)?;
            }
            submit.store(true, Ordering::SeqCst);
            let submit = Arc::new(AtomicBool::default());
            let op = Op::Upload {
                dst,
                submit: submit.clone(),
            };
            let future = upload_worker.send(op).map_err(|_| device_lost2)?;
            let host_copy = host_copy.replace(HostCopy {
                size: chunk_size,
                download_worker,
                upload_worker,
                submit,
                future,
            });
            if let Some(host_copy) = host_copy {
                host_copy.finish().map_err(|_| device_lost1)?;
            }
        }
        if let Some(host_copy) = host_copy {
            let future = host_copy.finish().map_err(|_| device_lost1)?;
            let mut future_guard = output.future.write();
            engine2.wait_for_future(&future_guard)?;
            *future_guard = future;
        }
        Ok(())
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn len(&self) -> usize {
        self.len
    }
    fn slice(self: &Arc<Self>, range: Range<usize>) -> Option<Arc<Self>> {
        let Range { start, end } = range;
        if start > self.len {
            return None;
        }
        if end > self.len {
            return None;
        }
        let offset = self.offset.checked_add(start)?;
        let len = end.checked_sub(start)?;
        Some(Arc::new(Self {
            offset,
            len,
            ..Self::clone(self)
        }))
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
            ..ShaderStages::empty()
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
            .map(|binding| {
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
        let buffers = buffers
            .iter()
            .map(|x| x.inner.as_ref().unwrap().into_buffer_slice())
            .collect();
        let submit = Arc::new(AtomicBool::default());
        let device_lost = DeviceLost {
            index: engine.info.index,
            handle: engine.handle(),
        };
        let op = Op::Compute {
            compute_pipeline: self.compute_pipeline.clone(),
            buffers,
            push_consts,
            groups,
            submit: submit.clone(),
        };
        let compute = engine.compute.lock();
        let workers = if engine.workers.len() == 1 {
            engine.workers.as_slice()
        } else {
            engine.workers.get(..engine.workers.len() - 1).unwrap()
        };
        let worker = 'outer: loop {
            for max_workers in [1, 2] {
                for [worker1, worker2] in workers {
                    let ready1 = worker1.ready();
                    let ready2 = worker2.ready();
                    if ready1 && (ready1 || max_workers == 2) {
                        break 'outer worker1;
                    }
                    if ready2 && (ready1 || max_workers == 2) {
                        break 'outer worker2;
                    }
                }
            }
        };
        let future = worker.send(op).map_err(|_| device_lost)?;
        std::mem::drop(compute);
        for guard in future_guards.iter() {
            while !guard.ready() {
                if engine.exited.load(Ordering::SeqCst) {
                    return Err(device_lost.into());
                }
            }
        }
        submit.store(true, Ordering::SeqCst);
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
