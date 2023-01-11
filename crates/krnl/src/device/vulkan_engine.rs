use super::{
    DeviceEngine, DeviceEngineBuffer, DeviceInfo, DeviceLost, DeviceOptions, Features,
    PerformanceMetrics, TransferMetrics,
};
use anyhow::Result;
use ash::vk::Handle;
use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use std::{
    ops::{Range, RangeBounds},
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
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
    instance::{Instance, InstanceCreateInfo},
    library::VulkanLibrary,
    memory::{
        allocator::{
            GenericMemoryAllocator, GenericMemoryAllocatorCreateInfo, MemoryAlloc, MemoryAllocator,
            StandardMemoryAllocator,
        },
        DedicatedAllocation, DeviceMemory, MemoryAllocateInfo,
    },
    pipeline::{self, ComputePipeline, Pipeline, PipelineBindPoint},
    shader::ShaderModule,
    sync::{Fence, FenceError, FenceSignalFuture, GpuFuture, NowFuture},
    VulkanObject,
};

pub struct Engine {
    info: Arc<DeviceInfo>,
    compute_families: Vec<u32>,
    transfer_op_sender: Sender<Op>,
    exited: Arc<AtomicUsize>,
    worker_states: Vec<WorkerState>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    device: Arc<Device>,
    instance: Arc<Instance>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.transfer_op_sender = crossbeam_channel::bounded(0).0;
        while self.exited.load(Ordering::SeqCst) < self.worker_states.len() {
            std::thread::sleep(Duration::from_micros(10));
        }
    }
}

impl DeviceEngine for Engine {
    type DeviceBuffer = DeviceBuffer;
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
        let mut compute_families: Vec<_> = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .filter(|(_, x)| x.queue_flags.compute)
            .map(|(i, x)| (i as u32, x.queue_flags))
            .collect();
        compute_families.sort_by_key(|(i, flags)| flags.graphics);
        let compute_families: Vec<u32> = compute_families.iter().map(|(i, _)| *i).collect();
        let mut transfer_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|x| {
                let flags = x.queue_flags;
                flags.transfer && !flags.compute && !flags.graphics
            })
            .map(|x| x as u32);
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
                queue_create_infos,
                ..Default::default()
            },
        )?;
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(
            device.clone(),
            GenericMemoryAllocatorCreateInfo {
                block_sizes: &[(0, 64_000_000), (i32::MAX as _, i32::MAX as _)],
                dedicated_allocation: false,
                ..Default::default()
            },
        )?);
        let mut worker_states = Vec::with_capacity(queues.len() * 2);
        let exited = Arc::new(AtomicUsize::default());
        let compute_queues: Vec<_> = queues.by_ref().take(compute_families.len()).collect();
        let transfer_queue = queues.next();
        let transfer_op_sender = if let Some(queue) = transfer_queue {
            let (op_sender, op_receiver) = crossbeam_channel::bounded(0);
            for _ in 0..2 {
                let worker = Worker::new(
                    op_receiver.clone(),
                    Some(&memory_allocator),
                    queue.clone(),
                    exited.clone(),
                )?;
                worker_states.push(worker.state.clone());
                std::thread::spawn(move || worker.run());
            }
            op_sender
        } else {
            let queue = compute_queues.first().unwrap();
            let (op_sender, op_receiver) = crossbeam_channel::bounded(0);
            for _ in 0..2 {
                let worker = Worker::new(
                    op_receiver.clone(),
                    Some(&memory_allocator),
                    queue.clone(),
                    exited.clone(),
                )?;
                worker_states.push(worker.state.clone());
                std::thread::spawn(move || worker.run());
            }
            op_sender
        };
        let queue_family_indices: Vec<u32> = compute_families
            .iter()
            .copied()
            .chain(transfer_family)
            .collect();
        let info = Arc::new(DeviceInfo {
            index,
            name,
            compute_queues: compute_families.len(),
            transfer_queues: transfer_family.is_some() as usize,
            features: Features::empty(),
        });
        Ok(Arc::new(Self {
            info,
            compute_families,
            transfer_op_sender,
            exited,
            worker_states,
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
            if self.exited.load(Ordering::SeqCst) > 0 {
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

#[derive(Clone, Default)]
struct WorkerState {
    pending: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
}

impl WorkerState {
    fn set_pending(&self) {
        self.pending.fetch_add(1, Ordering::SeqCst);
    }
    fn set_completed(&self) {
        self.completed
            .store(self.pending.load(Ordering::SeqCst), Ordering::SeqCst);
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

struct Worker {
    op_receiver: Receiver<Op>,
    state: WorkerState,
    fence: Fence,
    command_pool: CommandPool,
    command_pool_alloc: CommandPoolAlloc,
    host_buffer: Option<Arc<CpuAccessibleBuffer<[u8]>>>,
    queue: Arc<Queue>,
    guard: WorkerDropGuard,
}

impl Worker {
    fn new(
        op_receiver: Receiver<Op>,
        memory_allocator: Option<&Arc<StandardMemoryAllocator>>,
        queue: Arc<Queue>,
        exited: Arc<AtomicUsize>,
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
                transient: false,
                reset_command_buffer: true,
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
        let state = WorkerState::default();
        let fence = Fence::new(
            queue.device().clone(),
            vulkano::sync::FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )?;
        let guard = WorkerDropGuard { exited };
        Ok(Self {
            op_receiver,
            state,
            fence,
            command_pool,
            command_pool_alloc,
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
            self.state.set_completed();
            let device = self.queue.device();
            let command_pool_alloc = &self.command_pool_alloc;
            let mut builder = unsafe {
                UnsafeCommandBufferBuilder::new(
                    &self.command_pool_alloc,
                    CommandBufferBeginInfo {
                        usage: CommandBufferUsage::OneTimeSubmit,
                        ..Default::default()
                    },
                )?
            };
            self.fence.reset()?;
            let op = self.op_receiver.recv()?;
            self.state.set_pending();
            match op {
                Op::Upload {
                    src_sender,
                    dst,
                    submit_receiver,
                } => {
                    let buffer = self.host_buffer.as_ref().unwrap();
                    src_sender.send(buffer.clone()).unwrap();
                    unsafe {
                        builder.copy_buffer(&CopyBufferInfo::buffers(buffer.clone(), dst.clone()));
                    }
                    let command_buffer = builder.build()?;
                    let _ = submit_receiver.recv();
                    unsafe {
                        self.submit(&command_buffer)?;
                    }
                    self.fence.wait(None)?;
                }
                Op::Download {
                    src,
                    dst_sender,
                    finished_receiver,
                } => {
                    let buffer = self.host_buffer.as_ref().unwrap();
                    unsafe {
                        builder.copy_buffer(&CopyBufferInfo::buffers(src, buffer.clone()));
                    }
                    let command_buffer = builder.build()?;
                    unsafe {
                        self.submit(&command_buffer)?;
                    }
                    self.fence.wait(None)?;
                    let _ = dst_sender.send(buffer.clone());
                    let _ = finished_receiver.recv();
                }
            }
        }
    }
}

struct WorkerDropGuard {
    exited: Arc<AtomicUsize>,
}

impl Drop for WorkerDropGuard {
    fn drop(&mut self) {
        self.exited.fetch_add(1, Ordering::SeqCst);
    }
}

enum Op {
    Upload {
        src_sender: Sender<Arc<CpuAccessibleBuffer<[u8]>>>,
        dst: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
        submit_receiver: Receiver<()>,
    },
    Download {
        src: Arc<BufferSlice<[u8], DeviceLocalBuffer<[u8]>>>,
        dst_sender: Sender<Arc<CpuAccessibleBuffer<[u8]>>>,
        finished_receiver: Receiver<()>,
    },
}

fn aligned_floor(x: usize, align: usize) -> usize {
    if x % align != 0 {
        align * (x / align)
    } else {
        x
    }
}

fn aligned_ceil(x: usize, align: usize) -> usize {
    if x % align != 0 {
        aligned_floor(x, align) + align
    } else {
        x
    }
}

pub(super) struct DeviceBuffer {
    inner: Option<Arc<DeviceLocalBuffer<[u8]>>>,
    engine: Arc<Engine>,
    offset: usize,
    len: usize,
}

impl DeviceBuffer {
    const ALIGN: usize = 256;
    const HOST_BUFFER_SIZE: usize = 32_000_000;
}

impl DeviceEngineBuffer for DeviceBuffer {
    type Engine = Engine;
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    unsafe fn uninit(engine: Arc<Engine>, len: usize) -> Result<Arc<Self>> {
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
        Ok(Arc::new(Self {
            inner,
            engine,
            offset: 0,
            len,
        }))
    }
    fn upload(engine: Arc<Self::Engine>, data: &[u8]) -> Result<Arc<Self>> {
        let mut buffer = unsafe { Self::uninit(engine.clone(), data.len())? };
        if let Some(buffer) = buffer.inner.as_ref() {
            if let Ok(mut mapped) = buffer.inner().buffer.write(0..data.len() as _) {
                mapped.copy_from_slice(data);
            } else {
                let mut offset = 0;
                let device_lost = DeviceLost {
                    index: engine.info.index,
                    handle: engine.handle(),
                };
                for data in data.chunks(Self::HOST_BUFFER_SIZE) {
                    let (src_sender, src_receiver) = crossbeam_channel::bounded(0);
                    let dst = buffer
                        .slice(offset as _..(offset + data.len()) as _)
                        .unwrap();
                    let (submit_sender, submit_receiver) = crossbeam_channel::bounded(0);
                    let op = Op::Upload {
                        src_sender,
                        dst,
                        submit_receiver,
                    };
                    let send = Instant::now();
                    engine
                        .transfer_op_sender
                        .send(op)
                        .map_err(|_| device_lost)?;
                    let src = src_receiver.recv().map_err(|_| device_lost)?;
                    src.write().unwrap()[..data.len()].copy_from_slice(data);
                    submit_sender.send(()).map_err(|_| device_lost)?;
                    offset += data.len();
                }
            }
        }
        Ok(buffer)
    }
    fn download(&self, data: &mut [u8]) -> Result<(), DeviceLost> {
        if let Some(buffer) = self.inner.as_ref() {
            {
                let buffer_inner = buffer.inner();
                loop {
                    match buffer_inner
                        .buffer
                        .read(buffer_inner.offset..buffer_inner.offset + data.len() as u64)
                    {
                        Ok(mapped) => {
                            data.copy_from_slice(&mapped);
                            return Ok(());
                        }
                        Err(BufferError::InUseByDevice) => {
                            std::thread::sleep(Duration::from_micros(1));
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
            }
            let engine = &self.engine;
            let device_lost = DeviceLost {
                index: engine.info.index,
                handle: engine.handle(),
            };
            let mut offset = self.offset;
            struct HostCopy<'a> {
                data: &'a mut [u8],
                dst_receiver: Receiver<Arc<CpuAccessibleBuffer<[u8]>>>,
                finished_sender: Sender<()>,
            }
            let mut host_copy: Option<HostCopy> = None;
            for data in data.chunks_mut(Self::HOST_BUFFER_SIZE) {
                let src = buffer
                    .slice(offset as _..(offset + data.len()) as _)
                    .unwrap();
                offset += data.len();
                let (dst_sender, dst_receiver) = crossbeam_channel::bounded(0);
                let (finished_sender, finished_receiver) = crossbeam_channel::bounded(0);
                let op = Op::Download {
                    src,
                    dst_sender,
                    finished_receiver,
                };
                engine
                    .transfer_op_sender
                    .send(op)
                    .map_err(|_| device_lost)?;
                let host_copy = host_copy.replace(HostCopy {
                    data,
                    dst_receiver,
                    finished_sender,
                });
                if let Some(host_copy) = host_copy {
                    let dst = host_copy.dst_receiver.recv().map_err(|_| device_lost)?;
                    host_copy
                        .data
                        .copy_from_slice(&dst.read().unwrap()[..host_copy.data.len()]);
                    let _ = host_copy.finished_sender.send(());
                }
            }
            if let Some(host_copy) = host_copy {
                let dst = host_copy.dst_receiver.recv().map_err(|_| device_lost)?;
                host_copy
                    .data
                    .copy_from_slice(&dst.read().unwrap()[..host_copy.data.len()]);
                let _ = host_copy.finished_sender.send(());
            }
        }
        Ok(())
    }
    fn len(&self) -> usize {
        self.len
    }
    fn slice(self: &Arc<Self>, bounds: impl RangeBounds<usize>) -> Option<Arc<Self>> {
        todo!()
    }
}
