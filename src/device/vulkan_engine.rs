use super::{
    error::{DeviceIndexOutOfRange, DeviceUnavailable, OutOfDeviceMemory},
    DeviceEngine, DeviceEngineBuffer, DeviceEngineKernel, DeviceId, DeviceInfo, DeviceLost,
    DeviceOptions, Features, KernelDesc, KernelKey,
};

use anyhow::{Error, Result};
use ash::vk::Handle;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::{
    mem::MaybeUninit,
    ops::Range,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        pool::{CommandBufferAllocateInfo, CommandPool, CommandPoolAlloc, CommandPoolCreateInfo},
        sys::{CommandBufferBeginInfo, UnsafeCommandBuffer, UnsafeCommandBufferBuilder},
        CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
    },
    descriptor_set::{
        layout::{DescriptorSetLayout, DescriptorType},
        pool::{DescriptorPool, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo},
        WriteDescriptorSet,
    },
    device::{
        Device, DeviceCreateInfo, DeviceOwned, Queue, QueueCreateInfo, QueueFlags, QueueGuard,
    },
    instance::{Instance, InstanceCreateInfo, Version},
    library::{LoadingError, VulkanLibrary},
    memory::allocator::{
        AllocationCreateInfo, GenericMemoryAllocatorCreateInfo, MemoryUsage,
        StandardMemoryAllocator,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::{
        DescriptorBindingRequirements, DescriptorRequirements, ShaderExecution, ShaderInterface,
        ShaderModule, ShaderStages,
    },
    sync::semaphore::Semaphore,
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
    semaphore: Semaphore,
    epoch: AtomicU64,
    frame_sender: Sender<Frame>,
    frame_receiver: Receiver<Frame>,
    host_buffer_sender: Sender<HostBuffer>,
    host_buffer_receiver: Receiver<HostBuffer>,
    pending_buffers: Mutex<Option<PendingBuffers>>,
    pending_buffers_sender: Sender<PendingBuffers>,
    pending_buffers_receiver: Receiver<PendingBuffers>,
    kernels: DashMap<KernelKey, KernelInner>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    _instance: Arc<Instance>,
}

impl Engine {
    fn poll(&self) -> Result<(), DeviceLost> {
        if let Some(mut pending_buffers_guard) = self.pending_buffers.try_lock() {
            let epoch = unsafe {
                semaphore_value(self.queue.device(), &self.semaphore)
                    .map_err(|_| DeviceLost(self.id()))?
            };
            for mut pending_buffers in pending_buffers_guard
                .take()
                .into_iter()
                .chain(self.pending_buffers_receiver.try_iter())
            {
                if pending_buffers.epoch <= epoch {
                    pending_buffers.buffers.clear();
                } else {
                    pending_buffers_guard.replace(pending_buffers);
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.queue.with(|mut x| x.wait_idle());
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
        let library = vulkan_library().map_err(|e| Error::new(DeviceUnavailable).context(e))?;
        let instance = Instance::new(library, InstanceCreateInfo::application_from_cargo_toml())?;
        let mut physical_devices = instance.enumerate_physical_devices()?;
        let devices = physical_devices.len();
        let physical_device = if let Some(physical_device) = physical_devices.nth(index) {
            physical_device
        } else {
            return Err(DeviceIndexOutOfRange { index, devices }.into());
        };
        let name = physical_device.properties().device_name.clone();
        let optimal_device_extensions = vulkano::device::DeviceExtensions {
            khr_vulkan_memory_model: true,
            khr_push_descriptor: true,
            ..vulkano::device::DeviceExtensions::empty()
        };
        let device_extensions = physical_device
            .supported_extensions()
            .intersection(&optimal_device_extensions);
        let optimal_device_features = vulkano::device::Features {
            vulkan_memory_model: true,
            descriptor_buffer_push_descriptors: true,
            timeline_semaphore: true,
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
        let compute_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|x| {
                x.queue_flags.contains(QueueFlags::COMPUTE)
                    && !x.queue_flags.contains(QueueFlags::GRAPHICS)
            })
            .or_else(|| {
                physical_device
                    .queue_family_properties()
                    .iter()
                    .position(|x| x.queue_flags.contains(QueueFlags::COMPUTE))
            })
            .map(|x| x as u32)
            .unwrap();
        let queue_create_infos = vec![QueueCreateInfo {
            queue_family_index: compute_family,
            queues: vec![1f32],
            ..Default::default()
        }];
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                queue_create_infos,
                ..Default::default()
            },
        )?;
        let queue = queues.next().unwrap();
        let semaphore = new_semaphore(&device)?;
        let (frame_sender, frame_receiver) = crossbeam_channel::bounded::<Frame>(3);
        for _ in 0..3 {
            let frame = Frame::new(queue.clone())?;
            frame_sender.send(frame).unwrap();
        }
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
        let (host_buffer_sender, host_buffer_receiver) = crossbeam_channel::bounded(2);
        for _ in 0..2 {
            let buffer_info = BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            };
            let allocation_info = AllocationCreateInfo {
                usage: MemoryUsage::Download,
                ..Default::default()
            };
            let inner = Buffer::new_slice(
                &memory_allocator,
                buffer_info,
                allocation_info,
                DeviceBuffer::HOST_BUFFER_SIZE as u64,
            )?;
            host_buffer_sender
                .send(HostBuffer {
                    inner,
                    queue: queue.clone(),
                    epoch: 0,
                })
                .unwrap();
        }
        let pending_buffers = Mutex::new(None);
        let (pending_buffers_sender, pending_buffers_receiver) = crossbeam_channel::unbounded();
        let kernels = DashMap::default();
        let info = Arc::new(DeviceInfo {
            index,
            name,
            compute_queues: 1,
            transfer_queues: 0,
            features,
        });
        let epoch = AtomicU64::default();
        Ok(Arc::new(Self {
            info,
            semaphore,
            epoch,
            frame_sender,
            frame_receiver,
            host_buffer_sender,
            host_buffer_receiver,
            pending_buffers,
            pending_buffers_sender,
            pending_buffers_receiver,
            kernels,
            memory_allocator,
            queue,
            _instance: instance,
        }))
    }
    fn id(&self) -> DeviceId {
        let index = self.info.index;
        let handle = self.queue.device().handle().as_raw().try_into().unwrap();
        DeviceId { index, handle }
    }
    fn info(&self) -> &Arc<DeviceInfo> {
        &self.info
    }
    fn wait(&self) -> Result<(), DeviceLost> {
        let value = self.epoch.load(Ordering::SeqCst);
        unsafe { wait_semaphore(self.queue.device(), &self.semaphore, value) }
            .map_err(|_| DeviceLost(self.id()))?;
        self.poll()
    }
}

fn new_semaphore(device: &Arc<Device>) -> Result<Semaphore> {
    let mut semaphore = MaybeUninit::uninit();
    let mut semaphore_type_create_info = ash::vk::SemaphoreTypeCreateInfo::builder()
        .semaphore_type(ash::vk::SemaphoreType::TIMELINE);
    let semaphore_create_info =
        ash::vk::SemaphoreCreateInfo::builder().push_next(&mut semaphore_type_create_info);
    unsafe {
        (device.fns().v1_0.create_semaphore)(
            device.handle(),
            &*semaphore_create_info,
            std::ptr::null(),
            semaphore.as_mut_ptr(),
        )
        .result()
        .unwrap();
        Ok(Semaphore::from_handle(
            device.clone(),
            semaphore.assume_init(),
            Default::default(),
        ))
    }
}

unsafe fn queue_submit(
    queue: &Queue,
    _guard: QueueGuard,
    command_buffer: &UnsafeCommandBuffer,
    semaphore: &Semaphore,
    epoch: u64,
) -> Result<(), ash::vk::Result> {
    let command_buffers = &[command_buffer.handle()];
    let wait_semaphore_values = &[epoch];
    let signal_semaphore_values = &[epoch + 1];
    let mut semaphore_submit_info = ash::vk::TimelineSemaphoreSubmitInfo::builder()
        .wait_semaphore_values(wait_semaphore_values)
        .signal_semaphore_values(signal_semaphore_values);
    let wait_semaphores = &[semaphore.handle()];
    let signal_semaphores = &[semaphore.handle()];
    let wait_dst_stage_mask = &[ash::vk::PipelineStageFlags::ALL_COMMANDS];
    let submit_info = ash::vk::SubmitInfo::builder()
        .command_buffers(command_buffers)
        .wait_semaphores(wait_semaphores)
        .wait_dst_stage_mask(wait_dst_stage_mask)
        .signal_semaphores(signal_semaphores)
        .push_next(&mut semaphore_submit_info);
    let device = queue.device();
    unsafe {
        (device.fns().v1_0.queue_submit)(
            queue.handle(),
            1,
            [submit_info].as_ptr() as _,
            ash::vk::Fence::null(),
        )
        .result()?;
    }
    Ok(())
}

unsafe fn semaphore_value(device: &Device, semaphore: &Semaphore) -> Result<u64, ash::vk::Result> {
    let mut value = 0;
    unsafe {
        (device.fns().v1_2.get_semaphore_counter_value)(
            device.handle(),
            semaphore.handle(),
            &mut value,
        )
        .result_with_success(value)
    }
}

unsafe fn wait_semaphore(
    device: &Device,
    semaphore: &Semaphore,
    value: u64,
) -> Result<(), ash::vk::Result> {
    let semaphores = &[semaphore.handle()];
    let values = &[value];
    let semaphore_wait_info = ash::vk::SemaphoreWaitInfo::builder()
        .semaphores(semaphores)
        .values(values);
    loop {
        let result = unsafe {
            (device.fns().v1_2.wait_semaphores)(device.handle(), &*semaphore_wait_info, 0)
        };
        match result {
            ash::vk::Result::SUCCESS => return Ok(()),
            ash::vk::Result::TIMEOUT => continue,
            _ => return result.result(),
        }
    }
}

struct HostBuffer {
    inner: Subbuffer<[u8]>,
    queue: Arc<Queue>,
    epoch: u64,
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let result = self.queue.with(|mut x| x.wait_idle());
        if !std::thread::panicking() {
            result.unwrap();
        }
    }
}

struct PendingBuffers {
    buffers: Vec<Subbuffer<[u8]>>,
    queue: Arc<Queue>,
    epoch: u64,
}

impl Drop for PendingBuffers {
    fn drop(&mut self) {
        if !self.buffers.is_empty() {
            let result = self.queue.with(|mut x| x.wait_idle());
            if !std::thread::panicking() {
                result.unwrap();
            }
        }
    }
}

struct Frame {
    queue: Arc<Queue>,
    _command_pool: CommandPool,
    command_pool_alloc: CommandPoolAlloc,
    descriptor_pool: Option<DescriptorPool>,
    epoch: u64,
}

impl Frame {
    fn new(queue: Arc<Queue>) -> Result<Self> {
        let device = queue.device();
        let command_pool = CommandPool::new(
            device.clone(),
            CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                transient: true,
                reset_command_buffer: true,
                ..Default::default()
            },
        )?;
        let command_pool_alloc = command_pool
            .allocate_command_buffers(CommandBufferAllocateInfo {
                level: CommandBufferLevel::Primary,
                command_buffer_count: 1,
                ..Default::default()
            })
            .unwrap()
            .next()
            .unwrap();
        let descriptor_pool = if !device.enabled_features().descriptor_buffer_push_descriptors {
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
        let epoch = 0;
        Ok(Self {
            queue,
            _command_pool: command_pool,
            command_pool_alloc,
            descriptor_pool,
            epoch,
        })
    }
    unsafe fn command_buffer_builder(&mut self) -> Result<UnsafeCommandBufferBuilder> {
        let device = self.queue.device();
        unsafe {
            (device.fns().v1_0.reset_command_buffer)(
                self.command_pool_alloc.handle(),
                Default::default(),
            )
            .result()?;
        }
        unsafe {
            UnsafeCommandBufferBuilder::new(
                &self.command_pool_alloc,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::OneTimeSubmit,
                    ..Default::default()
                },
            )
            .map_err(Into::into)
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        let result = self.queue.with(|mut x| x.wait_idle());
        if !std::thread::panicking() {
            result.unwrap();
        }
    }
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

#[derive(Clone)]
pub(super) struct DeviceBuffer {
    inner: Option<Subbuffer<[u8]>>,
    engine: Arc<Engine>,
    offset: usize,
    len: usize,
    epoch: Arc<AtomicU64>,
}

impl DeviceBuffer {
    const MAX_LEN: usize = i32::MAX as usize;
    const MAX_SIZE: usize = aligned_ceil(Self::MAX_LEN, Self::ALIGN);
    const ALIGN: usize = 256;
    const HOST_BUFFER_SIZE: usize = 32_000_000;
    fn host_visible(&self) -> bool {
        if let Some(inner) = self.inner.as_ref() {
            inner.mapped_ptr().is_some()
        } else {
            false
        }
    }
}

impl DeviceEngineBuffer for DeviceBuffer {
    type Engine = Engine;
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    unsafe fn uninit(engine: Arc<Engine>, len: usize) -> Result<Self> {
        use vulkano::{memory::allocator::AllocationCreationError, VulkanError};
        let inner = if len > 0 {
            let len = aligned_ceil(len, Self::ALIGN);
            let usage =
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
            let buffer_info = BufferCreateInfo {
                usage,
                size: len.try_into().unwrap(),
                ..Default::default()
            };
            let allocation_info = AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            };
            use vulkano::{
                buffer::sys::RawBuffer,
                memory::{
                    allocator::{AllocationType, MemoryAllocator},
                    DeviceAlignment,
                },
            };
            let device = engine.queue.device();
            let raw_buffer = RawBuffer::new(device.clone(), buffer_info)?;
            let align = DeviceAlignment::new(DeviceBuffer::ALIGN.try_into().unwrap()).unwrap();
            let mut requirements = raw_buffer.memory_requirements().clone();
            requirements.layout = requirements.layout.align_to(align).unwrap();
            requirements.prefers_dedicated_allocation = false;
            engine.poll()?;
            let memory_alloc = engine
                .memory_allocator
                .allocate(requirements, AllocationType::Unknown, allocation_info, None)
                .map_err(|e| {
                    if let AllocationCreationError::VulkanError(VulkanError::OutOfDeviceMemory) = e
                    {
                        Error::new(OutOfDeviceMemory(engine.id())).context(e)
                    } else {
                        e.into()
                    }
                })?;
            debug_assert!(!memory_alloc.is_root());
            let buffer = raw_buffer
                .bind_memory(memory_alloc)
                .map_err(|(e, _, _)| e)?;
            Some(Subbuffer::new(Arc::new(buffer)))
            /*match Buffer::new_slice(
                &engine.memory_allocator,
                buffer_info,
                allocation_info,
                len.try_into().unwrap(),
            ) {
                Ok(inner) => {
                    use vulkano::buffer::BufferMemory;
                    let buffer = inner.buffer();
                    let alloc = match buffer.memory() {
                        BufferMemory::Normal(x) => x,
                        _ => todo!(),
                    };
                    if alloc.is_root() {
                        panic!("{:?}", (alloc.size(), alloc.device_memory().memory_type_index()));
                    }
                    Some(inner)
                },
                Err(
                    e @ BufferError::AllocError(AllocationCreationError::VulkanError(
                        VulkanError::OutOfDeviceMemory,
                    )),
                ) => return Err(Error::new(OutOfDeviceMemory(engine.id())).context(e)),
                Err(e) => {
                    return Err(e.into());
                }
            }*/
        } else {
            None
        };
        Ok(Self {
            inner,
            engine,
            offset: 0,
            len,
            epoch: Arc::new(AtomicU64::new(0)),
        })
    }
    fn upload(&self, data: &[u8]) -> Result<()> {
        debug_assert_eq!(data.len(), self.len);
        if self.len == 0 {
            return Ok(());
        }
        let buffer = if let Some(buffer) = self.inner.as_ref() {
            buffer
                .clone()
                .slice(self.offset as u64..(self.offset + self.len) as u64)
        } else {
            return Ok(());
        };
        let engine = &self.engine;
        let queue = &engine.queue;
        let device = queue.device();
        let buffer_epoch = self.epoch.load(Ordering::SeqCst);
        if self.host_visible() {
            unsafe {
                wait_semaphore(device, &engine.semaphore, buffer_epoch)?;
            }
            buffer.write().unwrap().copy_from_slice(data);
            return Ok(());
        }
        let mut offset = 0;
        for chunk in data.chunks(Self::HOST_BUFFER_SIZE) {
            let mut host_buffer = engine.host_buffer_receiver.recv().unwrap();
            let size = chunk.len() as u64;
            let buffer_slice = buffer.clone().slice(offset..offset + size);
            let host_buffer_slice = host_buffer.inner.clone().slice(0..size);
            unsafe {
                wait_semaphore(device, &engine.semaphore, host_buffer.epoch)?;
            }
            host_buffer_slice.write().unwrap().copy_from_slice(&chunk);
            let mut frame = engine
                .frame_receiver
                .recv()
                .map_err(|_| DeviceLost(engine.id()))?;
            unsafe {
                wait_semaphore(device, &engine.semaphore, frame.epoch)?;
            }
            let command_buffer = unsafe {
                let mut builder = frame.command_buffer_builder()?;
                builder.copy_buffer(&CopyBufferInfo::buffers(
                    host_buffer_slice.clone(),
                    buffer_slice.clone(),
                ));
                builder.build()?
            };
            queue.with(|guard| -> Result<()> {
                let epoch = engine.epoch.load(Ordering::SeqCst);
                unsafe {
                    queue_submit(&queue, guard, &command_buffer, &engine.semaphore, epoch)?;
                }
                let epoch = epoch + 1;
                engine.epoch.store(epoch, Ordering::SeqCst);
                frame.epoch = epoch;
                engine
                    .pending_buffers_sender
                    .send(PendingBuffers {
                        buffers: vec![buffer_slice],
                        queue: queue.clone(),
                        epoch,
                    })
                    .unwrap();
                host_buffer.epoch = epoch;
                engine.host_buffer_sender.send(host_buffer).unwrap();
                engine.frame_sender.send(frame).unwrap();
                self.epoch.store(epoch, Ordering::SeqCst);
                Ok(())
            })?;
            offset += size;
        }
        self.engine.poll()?;
        Ok(())
    }
    fn download(&self, data: &mut [u8]) -> Result<()> {
        debug_assert_eq!(data.len(), self.len);
        if self.len == 0 {
            return Ok(());
        }
        let buffer = if let Some(buffer) = self.inner.as_ref() {
            buffer
                .clone()
                .slice(self.offset as u64..(self.offset + self.len) as u64)
        } else {
            return Ok(());
        };
        let engine = &self.engine;
        let queue = &engine.queue;
        let device = queue.device();
        let buffer_epoch = self.epoch.load(Ordering::SeqCst);
        if self.host_visible() {
            unsafe {
                wait_semaphore(device, &engine.semaphore, buffer_epoch)?;
            }
            data.copy_from_slice(&buffer.read().unwrap());
            return Ok(());
        }
        struct HostCopy<'a> {
            chunk: &'a mut [u8],
            host_buffer: HostBuffer,
            host_slice: Subbuffer<[u8]>,
        }
        let mut host_copy: Option<HostCopy> = None;
        let mut offset = 0;
        for chunk in data.chunks_mut(Self::HOST_BUFFER_SIZE).chain([[].as_mut()]) {
            let prev_host_copy = host_copy.take();
            if !chunk.is_empty() {
                let mut host_buffer = engine.host_buffer_receiver.recv().unwrap();
                let mut frame = engine
                    .frame_receiver
                    .recv()
                    .map_err(|_| DeviceLost(engine.id()))?;
                let size = chunk.len() as u64;
                let buffer_slice = buffer.clone().slice(offset..offset + size);
                let host_slice = host_buffer.inner.clone().slice(0..size);
                unsafe {
                    wait_semaphore(device, &engine.semaphore, frame.epoch)?;
                }
                let command_buffer = unsafe {
                    let mut builder = frame.command_buffer_builder()?;
                    builder.copy_buffer(&CopyBufferInfo::buffers(
                        buffer_slice.clone(),
                        host_slice.clone(),
                    ));
                    builder.build()?
                };
                queue.with(|guard| -> Result<()> {
                    let epoch = engine.epoch.load(Ordering::SeqCst);
                    unsafe {
                        queue_submit(&queue, guard, &command_buffer, &engine.semaphore, epoch)?;
                    }
                    let epoch = epoch + 1;
                    engine.epoch.store(epoch, Ordering::SeqCst);
                    frame.epoch = epoch;
                    engine
                        .pending_buffers_sender
                        .send(PendingBuffers {
                            buffers: vec![buffer_slice],
                            queue: queue.clone(),
                            epoch,
                        })
                        .unwrap();
                    engine.frame_sender.send(frame).unwrap();
                    host_buffer.epoch = epoch;
                    Ok(())
                })?;
                host_copy.replace(HostCopy {
                    chunk,
                    host_buffer,
                    host_slice,
                });
                offset += size;
            }
            if let Some(prev_host_copy) = prev_host_copy {
                let HostCopy {
                    chunk,
                    host_buffer,
                    host_slice,
                } = prev_host_copy;
                unsafe {
                    wait_semaphore(device, &engine.semaphore, host_buffer.epoch)?;
                }
                chunk.copy_from_slice(&host_slice.read().unwrap());
                engine.host_buffer_sender.send(host_buffer).unwrap();
            }
        }
        self.engine.poll()?;
        Ok(())
    }
    fn transfer(&self, dst: &Self) -> Result<()> {
        debug_assert_eq!(dst.len, self.len);
        if self.len == 0 {
            return Ok(());
        }
        let (buffer1, buffer2) =
            if let Some((buffer1, buffer2)) = self.inner.as_ref().zip(dst.inner.as_ref()) {
                let buffer1 = buffer1
                    .clone()
                    .slice(self.offset as u64..(self.offset + self.len) as u64);
                let buffer2 = buffer2
                    .clone()
                    .slice(dst.offset as u64..(dst.offset + dst.len) as u64);
                (buffer1, buffer2)
            } else {
                return Ok(());
            };
        let engine1 = &self.engine;
        let queue1 = &engine1.queue;
        let device1 = queue1.device();
        let buffer1_epoch = self.epoch.load(Ordering::SeqCst);
        let engine2 = &dst.engine;
        let queue2 = &engine2.queue;
        let device2 = queue2.device();
        let buffer2_epoch = dst.epoch.load(Ordering::SeqCst);
        if self.host_visible() && dst.host_visible() {
            unsafe {
                wait_semaphore(device1, &engine1.semaphore, buffer1_epoch)?;
                wait_semaphore(device2, &engine2.semaphore, buffer2_epoch)?;
            }
            buffer2
                .write()
                .unwrap()
                .copy_from_slice(&buffer1.read().unwrap());
            return Ok(());
        } else if self.host_visible() {
            unsafe {
                wait_semaphore(device1, &engine1.semaphore, buffer1_epoch)?;
            }
            return dst.upload(&buffer1.read().unwrap());
        } else if dst.host_visible() {
            unsafe {
                wait_semaphore(device2, &engine2.semaphore, buffer2_epoch)?;
            }
            return self.download(&mut buffer2.write().unwrap());
        }
        struct HostCopy {
            host_buffer1: HostBuffer,
            host_slice1: Subbuffer<[u8]>,
            buffer_slice2: Subbuffer<[u8]>,
        }
        let mut host_copy: Option<HostCopy> = None;
        let mut offset = 0;
        loop {
            let size = buffer1
                .size()
                .checked_sub(offset)
                .unwrap_or_default()
                .min(Self::HOST_BUFFER_SIZE as u64);
            let prev_host_copy = host_copy.take();
            if size > 0 {
                let mut host_buffer1 = engine1.host_buffer_receiver.recv().unwrap();
                let mut frame1 = engine1
                    .frame_receiver
                    .recv()
                    .map_err(|_| DeviceLost(engine1.id()))?;
                let buffer_slice1 = buffer1.clone().slice(offset..offset + size);
                let host_slice1 = host_buffer1.inner.clone().slice(0..size);
                unsafe {
                    wait_semaphore(device1, &engine1.semaphore, frame1.epoch)?;
                }
                let command_buffer = unsafe {
                    let mut builder = frame1.command_buffer_builder()?;
                    builder.copy_buffer(&CopyBufferInfo::buffers(
                        buffer_slice1.clone(),
                        host_slice1.clone(),
                    ));
                    builder.build()?
                };
                queue1.with(|guard| -> Result<()> {
                    let epoch = engine1.epoch.load(Ordering::SeqCst);
                    unsafe {
                        queue_submit(&queue1, guard, &command_buffer, &engine1.semaphore, epoch)?;
                    }
                    let epoch = epoch + 1;
                    engine1.epoch.store(epoch, Ordering::SeqCst);
                    frame1.epoch = epoch;
                    engine1
                        .pending_buffers_sender
                        .send(PendingBuffers {
                            buffers: vec![buffer_slice1],
                            queue: queue1.clone(),
                            epoch,
                        })
                        .unwrap();
                    engine1.frame_sender.send(frame1).unwrap();
                    host_buffer1.epoch = epoch;
                    Ok(())
                })?;
                let buffer_slice2 = buffer2.clone().slice(offset..offset + size);
                host_copy.replace(HostCopy {
                    host_buffer1,
                    host_slice1,
                    buffer_slice2,
                });
                offset += size;
            }
            if let Some(prev_host_copy) = prev_host_copy {
                let HostCopy {
                    host_buffer1,
                    host_slice1,
                    buffer_slice2,
                } = prev_host_copy;
                let size = buffer_slice2.size();
                let mut host_buffer2 = engine2.host_buffer_receiver.recv().unwrap();
                let host_slice2 = host_buffer2.inner.clone().slice(0..size);
                unsafe {
                    wait_semaphore(device1, &engine1.semaphore, host_buffer1.epoch)?;
                    wait_semaphore(device2, &engine2.semaphore, host_buffer2.epoch)?;
                }
                host_slice2
                    .write()
                    .unwrap()
                    .copy_from_slice(&host_slice1.read().unwrap());
                engine1.host_buffer_sender.send(host_buffer1).unwrap();
                let mut frame2 = engine2
                    .frame_receiver
                    .recv()
                    .map_err(|_| DeviceLost(engine2.id()))?;
                unsafe {
                    wait_semaphore(device2, &engine2.semaphore, frame2.epoch)?;
                }
                let command_buffer = unsafe {
                    let mut builder = frame2.command_buffer_builder()?;
                    builder.copy_buffer(&CopyBufferInfo::buffers(
                        host_slice2.clone(),
                        buffer_slice2.clone(),
                    ));
                    builder.build()?
                };
                queue2.with(|guard| -> Result<()> {
                    let epoch = engine2.epoch.load(Ordering::SeqCst);
                    unsafe {
                        queue_submit(&queue2, guard, &command_buffer, &engine2.semaphore, epoch)?;
                    }
                    let epoch = epoch + 1;
                    engine2.epoch.store(epoch, Ordering::SeqCst);
                    frame2.epoch = epoch;
                    engine2
                        .pending_buffers_sender
                        .send(PendingBuffers {
                            buffers: vec![buffer_slice2],
                            queue: queue2.clone(),
                            epoch,
                        })
                        .unwrap();
                    engine2.frame_sender.send(frame2).unwrap();
                    host_buffer2.epoch = epoch;
                    dst.epoch.store(epoch, Ordering::SeqCst);
                    Ok(())
                })?;
                engine2.host_buffer_sender.send(host_buffer2).unwrap();
            } else if size == 0 {
                break;
            }
        }
        engine1.poll()?;
        engine2.poll()?;
        Ok(())
        /*
        if self.len == 0 {
            return Ok(());
        }
        let output = dst;
        let buffer1 = self.inner.as_ref().unwrap();
        let engine1 = &self.engine;
        let buffer2 = output.inner.as_ref().unwrap();
        let engine2 = &output.engine;
        let prev_future = self.future.read();
        if self.host_visible() {
            engine1.wait_for_future(&prev_future)?;
            let buffer1 = buffer1.clone().slice(
                DeviceSize::try_from(self.offset).unwrap()
                    ..DeviceSize::try_from(self.offset + self.len()).unwrap(),
            );
            let mapped1 = buffer1.read()?;
            if dst.host_visible() {
                let buffer2 = buffer2.clone().slice(
                    DeviceSize::try_from(dst.offset).unwrap()
                        ..DeviceSize::try_from(dst.offset + dst.len()).unwrap(),
                );
                let mut mapped2 = buffer2.write()?;
                mapped2.copy_from_slice(&mapped1[..output.len()]);
            } else {
                output.upload(&mapped1)?;
            }
            return Ok(());
        } else if output.host_visible() {
            let buffer2 = buffer2.clone().slice(
                DeviceSize::try_from(dst.offset).unwrap()
                    ..DeviceSize::try_from(dst.offset + dst.len()).unwrap(),
            );
            let mut mapped2 = buffer2.write()?;
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
            let src = buffer1.clone().slice(
                DeviceSize::try_from(offset1).unwrap()
                    ..DeviceSize::try_from(offset1 + chunk_size).unwrap(),
            );
            let dst = buffer2.clone().slice(
                DeviceSize::try_from(offset2).unwrap()
                    ..DeviceSize::try_from(offset2 + chunk_size).unwrap(),
            );
            offset1 += chunk_size;
            offset2 += chunk_size;
            let submit = Arc::new(AtomicBool::default());
            let op = Op::Download {
                src,
                submit: submit.clone(),
            };
            download_worker
                .send(op)
                .map_err(|_| DeviceLost(engine1.id()))?;
            if let Some(future) = prev_future.take() {
                engine1.wait_for_future(&future)?;
            }
            submit.store(true, Ordering::SeqCst);
            let submit = Arc::new(AtomicBool::default());
            let op = Op::Upload {
                dst,
                submit: submit.clone(),
            };
            let future = upload_worker
                .send(op)
                .map_err(|_| DeviceLost(engine2.id()))?;
            let host_copy = host_copy.replace(HostCopy {
                size: chunk_size,
                download_worker,
                upload_worker,
                submit,
                future,
            });
            if let Some(host_copy) = host_copy {
                host_copy.finish().map_err(|_| DeviceLost(engine1.id()))?;
            }
        }
        if let Some(host_copy) = host_copy {
            let future = host_copy.finish().map_err(|_| DeviceLost(engine1.id()))?;
            let mut future_guard = output.future.write();
            engine2.wait_for_future(&future_guard)?;
            *future_guard = future;
        }
        Ok(())*/
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
        let device = engine.queue.device();
        let descriptor_binding_requirements = desc
            .slice_descs
            .iter()
            .enumerate()
            .map(|(i, desc)| {
                let set = 0;
                let binding = i.try_into().unwrap();
                let memory_write = if desc.mutable {
                    ShaderStages::COMPUTE
                } else {
                    ShaderStages::empty()
                };
                let descriptors = DescriptorRequirements {
                    memory_read: ShaderStages::COMPUTE,
                    memory_write,
                    ..DescriptorRequirements::default()
                };
                let descriptor_binding_requirements = DescriptorBindingRequirements {
                    descriptor_types: vec![DescriptorType::StorageBuffer],
                    descriptor_count: Some(1),
                    stages: ShaderStages::COMPUTE,
                    descriptors: [(Some(0), descriptors)].into_iter().collect(),
                    ..Default::default()
                };
                ((set, binding), descriptor_binding_requirements)
            })
            .collect();
        let push_consts_range = desc.push_consts_range();
        let push_constant_range = if push_consts_range > 0 {
            Some(PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: push_consts_range,
            })
        } else {
            None
        };
        let entry_point_info = EntryPointInfo {
            execution: ShaderExecution::Compute,
            descriptor_binding_requirements,
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
                    stages: ShaderStages::COMPUTE,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                };
                (binding.try_into().unwrap(), descriptor_set_layout_binding)
            })
            .collect();
        let descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings,
            push_descriptor: device.enabled_features().descriptor_buffer_push_descriptors,
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
        let compute_pipeline = ComputePipeline::with_pipeline_layout(
            device.clone(),
            shader_module.entry_point(entry_point).unwrap(),
            &(),
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
        push_consts: Vec<u8>,
    ) -> Result<()> {
        let engine = &self.engine;
        let queue = &engine.queue;
        let device = queue.device();
        let mut frame = engine.frame_receiver.recv().unwrap();
        unsafe {
            wait_semaphore(device, &engine.semaphore, frame.epoch)?;
        }
        let command_buffer = {
            let mut builder = unsafe { frame.command_buffer_builder()? };
            let compute_pipeline = &self.compute_pipeline;
            unsafe {
                builder.bind_pipeline_compute(&compute_pipeline);
            }
            let pipeline_layout = compute_pipeline.layout();
            if !buffers.is_empty() {
                let descriptor_set_layout = pipeline_layout.set_layouts().first().unwrap();
                let write_descriptor_set = WriteDescriptorSet::buffer_array(
                    0,
                    0,
                    buffers.iter().map(|x| x.inner.as_ref().unwrap().clone()),
                );
                if descriptor_set_layout.push_descriptor() {
                    unsafe {
                        builder.push_descriptor_set(
                            PipelineBindPoint::Compute,
                            pipeline_layout,
                            0,
                            &[write_descriptor_set],
                        );
                    }
                } else {
                    let descriptor_pool = frame.descriptor_pool.as_ref().unwrap();
                    let mut descriptor_set = unsafe {
                        descriptor_pool.reset()?;
                        descriptor_pool
                            .allocate_descriptor_sets([DescriptorSetAllocateInfo {
                                layout: descriptor_set_layout,
                                variable_descriptor_count: 0,
                            }])
                            .unwrap()
                            .next()
                            .unwrap()
                    };
                    unsafe {
                        descriptor_set.write(descriptor_set_layout, [&write_descriptor_set]);
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
            }
            if !push_consts.is_empty() {
                unsafe {
                    builder.push_constants(
                        pipeline_layout,
                        ShaderStages::COMPUTE,
                        0,
                        push_consts.len() as u32,
                        push_consts.as_slice(),
                    );
                }
            }
            unsafe {
                builder.dispatch(groups);
            }
            builder.build()?
        };
        queue.with(|guard| -> Result<()> {
            let epoch = engine.epoch.load(Ordering::SeqCst);
            unsafe {
                queue_submit(&queue, guard, &command_buffer, &engine.semaphore, epoch)?;
            }
            let epoch = epoch + 1;
            engine.epoch.store(epoch, Ordering::SeqCst);
            frame.epoch = epoch;
            engine
                .pending_buffers_sender
                .send(PendingBuffers {
                    buffers: buffers
                        .iter()
                        .map(|x| x.inner.as_ref().unwrap().clone())
                        .collect(),
                    queue: queue.clone(),
                    epoch,
                })
                .unwrap();
            engine.frame_sender.send(frame).unwrap();
            for buffer in buffers.iter() {
                buffer.epoch.store(epoch, Ordering::SeqCst);
            }
            Ok(())
        })?;
        engine.poll()?;
        Ok(())
        /*
        let engine = &self.engine;
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
            .map(|x| x.inner.as_ref().unwrap().clone())
            .collect();
        let submit = Arc::new(AtomicBool::default());
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
                    if ready1 && ready2 {
                        break 'outer worker1;
                    } else if max_workers == 2 {
                        if ready1 {
                            break 'outer worker1;
                        } else if ready2 {
                            break 'outer worker2;
                        }
                    }
                }
            }
        };
        let future = worker.send(op).map_err(|_| DeviceLost(engine.id()))?;
        std::mem::drop(compute);
        for guard in future_guards.iter() {
            while !guard.ready() {
                if engine.exited.load(Ordering::SeqCst) {
                    return Err(DeviceLost(engine.id()).into());
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
        Ok(())*/
    }
    fn desc(&self) -> &Arc<KernelDesc> {
        &self.desc
    }
}
