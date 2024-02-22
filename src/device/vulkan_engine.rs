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
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{CommandBufferAlloc, CommandBufferAllocator, CommandBufferBuilderAlloc},
        pool::{
            CommandBufferAllocateInfo, CommandPool, CommandPoolAlloc, CommandPoolCreateFlags,
            CommandPoolCreateInfo,
        },
        sys::{CommandBufferBeginInfo, UnsafeCommandBuffer, UnsafeCommandBufferBuilder},
        CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
    },
    descriptor_set::{
        layout::{DescriptorSetLayout, DescriptorSetLayoutCreateFlags, DescriptorType},
        pool::{DescriptorPool, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceOwned, Queue, QueueCreateInfo, QueueFlags, QueueGuard,
    },
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateInfo, InstanceExtensions,
    },
    library::VulkanLibrary,
    memory::{
        allocator::{
            AllocationCreateInfo, GenericMemoryAllocatorCreateInfo, MemoryAllocatorError,
            MemoryTypeFilter, StandardMemoryAllocator,
        },
        MemoryPropertyFlags, ResourceMemory,
    },
    pipeline::{
        compute::ComputePipelineCreateInfo, ComputePipeline, Pipeline,
        PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
    sync::semaphore::Semaphore,
    VulkanError, VulkanObject,
};

pub struct Engine {
    info: Arc<DeviceInfo>,
    semaphore: Arc<Semaphore>,
    epoch: AtomicU64,
    pending: Arc<AtomicU64>,
    frame_outer: Mutex<FrameOuter>,
    host_buffer_sender: Sender<HostBuffer>,
    host_buffer_receiver: Receiver<HostBuffer>,
    kernels: DashMap<KernelKey, KernelInner>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    engine_exited: Arc<AtomicBool>,
    worker_exited: Arc<AtomicBool>,
    _instance: Arc<Instance>,
}

impl Engine {
    unsafe fn transfer(
        &self,
        src: Subbuffer<[u8]>,
        dst: Subbuffer<[u8]>,
        host_buffer: &mut HostBuffer,
        dst_device_buffer: Option<&DeviceBuffer>,
    ) -> Result<()> {
        let mut frame_outer = self.frame_outer.lock();
        unsafe { frame_outer.transfer(&self.epoch, src, dst, host_buffer, dst_device_buffer) }
    }
    unsafe fn compute(
        &self,
        kernel_desc: &Arc<KernelDesc>,
        pipeline: &Arc<ComputePipeline>,
        groups: u32,
        buffers: &[Arc<DeviceBuffer>],
        push_consts: &[u8],
        debug_printf_panic: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        let mut frame_outer = self.frame_outer.lock();
        let new_descriptors: u32 = buffers.len().try_into().unwrap();
        if frame_outer.kernels >= Frame::MAX_KERNELS
            || frame_outer.descriptors + new_descriptors > Frame::MAX_DESCRIPTORS
        {
            loop {
                if frame_outer.empty.load(Ordering::SeqCst) {
                    break;
                }
                if self.worker_exited.load(Ordering::SeqCst) {
                    return Err(DeviceLost(self.id()).into());
                }
                std::hint::spin_loop();
            }
        }
        unsafe {
            frame_outer.compute(
                kernel_desc,
                &self.epoch,
                pipeline,
                groups,
                buffers,
                push_consts,
                debug_printf_panic,
            )
        }
    }
    fn wait_pending(&self, epoch: u64) -> Result<(), DeviceLost> {
        while self.pending.load(Ordering::SeqCst) < epoch {
            if self.worker_exited.load(Ordering::SeqCst) {
                return Err(DeviceLost(self.id()));
            }
            std::hint::spin_loop();
        }
        Ok(())
    }
    fn wait_epoch(&self, epoch: u64) -> Result<(), DeviceLost> {
        loop {
            let result = unsafe { wait_semaphore(self.queue.device(), &self.semaphore, epoch) };
            match result {
                ash::vk::Result::SUCCESS => return Ok(()),
                ash::vk::Result::TIMEOUT => (),
                _ => return Err(DeviceLost(self.id())),
            }
            if self.worker_exited.load(Ordering::SeqCst) {
                return Err(DeviceLost(self.id()));
            }
            std::hint::spin_loop();
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.engine_exited.store(true, Ordering::SeqCst);
        while !self.worker_exited.load(Ordering::SeqCst) {}
        let result = self.queue.with(|mut x| x.wait_idle());
        if !std::thread::panicking() {
            result.unwrap();
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
        let library = VulkanLibrary::new().map_err(|e| Error::new(DeviceUnavailable).context(e))?;
        let debug_printf = Arc::new(AtomicBool::default());
        let debug_printf2 = debug_printf.clone();
        let debug_create_info = DebugUtilsMessengerCreateInfo {
            message_severity: DebugUtilsMessageSeverity::INFO,
            message_type: DebugUtilsMessageType::VALIDATION,
            ..DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(move |_severity, _message_type, data| {
                    if debug_printf2.load(Ordering::SeqCst) {
                        return;
                    }
                    if data.message_id_name
                        == Some("UNASSIGNED-khronos-validation-createinstance-status-message")
                        && data.message.contains("Khronos Validation Layer Active:")
                        && data.message.contains(
                            "Current Enables: VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT.",
                        )
                    {
                        debug_printf2.store(true, Ordering::SeqCst);
                    }
                })
            })
        };
        let instance_create_info = InstanceCreateInfo {
            enabled_extensions: InstanceExtensions {
                ext_debug_utils: true,
                khr_portability_enumeration: true,
                ..Default::default()
            },
            // enumerate_portability: true,
            debug_utils_messengers: vec![debug_create_info],
            ..InstanceCreateInfo::application_from_cargo_toml()
        };
        let instance = Instance::new(library, instance_create_info)?;
        let debug_printf = debug_printf.load(Ordering::SeqCst);
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
            ext_subgroup_size_control: true,
            ..vulkano::device::DeviceExtensions::empty()
        };
        let device_extensions = physical_device
            .supported_extensions()
            .intersection(&optimal_device_extensions);
        let optimal_device_features = vulkano::device::Features {
            vulkan_memory_model: true,
            timeline_semaphore: true,
            subgroup_size_control: true,
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
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                queue_create_infos,
                ..Default::default()
            },
        )?;
        let queue = queues.next().unwrap();
        let memory_propoerties = physical_device.memory_properties();
        let memory_types = &memory_propoerties.memory_types;
        let memory_heaps = &memory_propoerties.memory_heaps;
        let block_sizes: Vec<u64> = memory_types
            .iter()
            .map(|x| {
                let size = memory_heaps[x.heap_index as usize].size;
                let device_local = x.property_flags.contains(MemoryPropertyFlags::DEVICE_LOCAL);
                let block_size: u64 = if device_local {
                    DeviceBuffer::MAX_SIZE as _
                } else {
                    64_000_000
                };
                block_size.min(size)
            })
            .collect();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(
            device.clone(),
            GenericMemoryAllocatorCreateInfo {
                block_sizes: &block_sizes,
                dedicated_allocation: false,
                ..Default::default()
            },
        ));
        let (host_buffer_sender, host_buffer_receiver) = crossbeam_channel::bounded(2);
        for _ in 0..2 {
            let buffer_info = BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            };
            let allocation_info = AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter {
                    required_flags: MemoryPropertyFlags::HOST_VISIBLE,
                    preferred_flags: MemoryPropertyFlags::HOST_COHERENT
                        | MemoryPropertyFlags::HOST_CACHED,
                    not_preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                },
                ..Default::default()
            };
            let inner = Buffer::new_slice(
                memory_allocator.clone(),
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
        let kernels = DashMap::default();
        let properties = device.physical_device().properties();
        let info = Arc::new(DeviceInfo {
            index,
            name,
            device_id: properties.device_id,
            vendor_id: properties.vendor_id,
            max_groups: properties.max_compute_work_group_count[0],
            max_threads: properties.max_compute_work_group_size[0],
            subgroup_threads: properties.subgroup_size,
            features,
            debug_printf,
        });
        let mut worker = Worker::new(queue.clone(), index)?;
        let semaphore = worker.semaphore.clone();
        let epoch = AtomicU64::default();
        let pending = worker.pending.clone();
        let frame_outer = Mutex::new(FrameOuter::new(
            worker.ready_frame.clone(),
            worker.empty.clone(),
        ));
        let engine_exited = worker.engine_exited.clone();
        let worker_exited = worker.worker_exited.clone();
        std::thread::spawn(move || worker.run());
        Ok(Arc::new(Self {
            info,
            semaphore,
            epoch,
            pending,
            frame_outer,
            host_buffer_sender,
            host_buffer_receiver,
            kernels,
            memory_allocator,
            engine_exited,
            worker_exited,
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
        let epoch = self.epoch.load(Ordering::SeqCst);
        self.wait_epoch(epoch)
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
    _guard: &mut QueueGuard,
    command_buffer: &UnsafeCommandBuffer<ArcCommandPoolBufferAllocator>,
    semaphore: &Semaphore,
    epoch: u64,
) -> Result<(), ash::vk::Result> {
    let command_buffers = &[command_buffer.handle()];
    let signal_semaphore_values = &[epoch];
    let mut semaphore_submit_info = ash::vk::TimelineSemaphoreSubmitInfo::builder()
        .signal_semaphore_values(signal_semaphore_values);
    let signal_semaphores = &[semaphore.handle()];
    let submit_info = ash::vk::SubmitInfo::builder()
        .command_buffers(command_buffers)
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

unsafe fn wait_semaphore(device: &Device, semaphore: &Semaphore, value: u64) -> ash::vk::Result {
    let semaphores = &[semaphore.handle()];
    let values = &[value];
    let semaphore_wait_info = ash::vk::SemaphoreWaitInfo::builder()
        .semaphores(semaphores)
        .values(values);
    unsafe { (device.fns().v1_2.wait_semaphores)(device.handle(), &*semaphore_wait_info, 0) }
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

struct FrameOuter {
    frame: Arc<Mutex<Frame>>,
    empty: Arc<AtomicBool>,
    kernels: u32,
    descriptors: u32,
}

impl FrameOuter {
    fn new(frame: Arc<Mutex<Frame>>, empty: Arc<AtomicBool>) -> Self {
        Self {
            frame,
            empty,
            kernels: 0,
            descriptors: 0,
        }
    }
    unsafe fn transfer(
        &mut self,
        epoch: &AtomicU64,
        src: Subbuffer<[u8]>,
        dst: Subbuffer<[u8]>,
        host_buffer: &mut HostBuffer,
        dst_device_buffer: Option<&DeviceBuffer>,
    ) -> Result<()> {
        let mut frame = self.frame.lock();
        if frame.command_buffer_builder.is_none() {
            self.kernels = 0;
            self.descriptors = 0;
            unsafe {
                frame.begin()?;
            }
            epoch.store(frame.epoch, Ordering::SeqCst);
            self.empty.store(false, Ordering::SeqCst);
        }
        unsafe { frame.transfer(src, dst, host_buffer, dst_device_buffer) }
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn compute(
        &mut self,
        kernel_desc: &Arc<KernelDesc>,
        epoch: &AtomicU64,
        pipeline: &Arc<ComputePipeline>,
        groups: u32,
        buffers: &[Arc<DeviceBuffer>],
        push_consts: &[u8],
        debug_printf_panic: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        let new_descriptors: u32 = buffers.len().try_into().unwrap();
        let mut frame = self.frame.lock();
        if frame.command_buffer_builder.is_none() {
            self.kernels = 0;
            self.descriptors = 0;
            unsafe {
                frame.begin()?;
            }
            epoch.store(frame.epoch, Ordering::SeqCst);
            self.empty.store(false, Ordering::SeqCst);
        }
        unsafe {
            frame.compute(
                kernel_desc,
                pipeline,
                groups,
                buffers,
                push_consts,
                debug_printf_panic,
            )?;
        }
        self.kernels += 1;
        self.descriptors += new_descriptors;
        Ok(())
    }
}

struct Frame {
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<CommandPoolBufferAllocator>,
    command_buffer_builder: Option<UnsafeCommandBufferBuilder<ArcCommandPoolBufferAllocator>>,
    descriptor_pool: DescriptorPool,
    buffers: Vec<Subbuffer<[u8]>>,
    epoch: u64,
    debug_kernel_desc_panic: Option<(Arc<KernelDesc>, Arc<AtomicBool>)>,
}

impl Frame {
    const MAX_KERNELS: u32 = 4;
    const MAX_DESCRIPTORS: u32 = 32;
    fn new(queue: Arc<Queue>) -> Result<Self> {
        let device = queue.device();
        let command_buffer_allocator = CommandPoolBufferAllocator::new(&queue)?;
        let command_buffer_builder = None;
        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolCreateInfo {
                max_sets: Self::MAX_DESCRIPTORS,
                pool_sizes: [(DescriptorType::StorageBuffer, Self::MAX_DESCRIPTORS)]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
        )?;
        let buffers = Vec::new();
        let epoch = 0;
        Ok(Self {
            queue,
            command_buffer_allocator,
            command_buffer_builder,
            descriptor_pool,
            buffers,
            epoch,
            debug_kernel_desc_panic: None,
        })
    }
    unsafe fn begin(&mut self) -> Result<()> {
        unsafe {
            self.command_buffer_allocator.reset()?;
            self.descriptor_pool.reset()?;
        }
        self.command_buffer_builder.replace(unsafe {
            UnsafeCommandBufferBuilder::new(
                &ArcCommandPoolBufferAllocator(self.command_buffer_allocator.clone()),
                self.queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::OneTimeSubmit,
                    ..Default::default()
                },
            )?
        });
        Ok(())
    }
    unsafe fn transfer(
        &mut self,
        src: Subbuffer<[u8]>,
        dst: Subbuffer<[u8]>,
        host_buffer: &mut HostBuffer,
        dst_device_buffer: Option<&DeviceBuffer>,
    ) -> Result<()> {
        let builder = self.command_buffer_builder.as_mut().unwrap();
        unsafe {
            builder.copy_buffer(&CopyBufferInfo::buffers(src.clone(), dst.clone()))?;
        }
        self.buffers.extend_from_slice(&[src, dst]);
        host_buffer.epoch = self.epoch;
        if let Some(dst_device_buffer) = dst_device_buffer {
            dst_device_buffer.epoch.store(self.epoch, Ordering::SeqCst);
        }
        Ok(())
    }
    unsafe fn compute(
        &mut self,
        kernel_desc: &Arc<KernelDesc>,
        pipeline: &Arc<ComputePipeline>,
        groups: u32,
        buffers: &[Arc<DeviceBuffer>],
        push_consts: &[u8],
        debug_printf_panic: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        let device = self.queue.device();
        let builder = self.command_buffer_builder.as_mut().unwrap();
        unsafe {
            builder.bind_pipeline_compute(pipeline)?;
        }
        let pipeline_layout = pipeline.layout();
        if !buffers.is_empty() {
            let descriptor_set_layout = pipeline_layout.set_layouts().first().unwrap();
            let descriptor_set = unsafe {
                self.descriptor_pool
                    .allocate_descriptor_sets([DescriptorSetAllocateInfo::new(
                        descriptor_set_layout.clone(),
                    )])
                    .unwrap()
                    .next()
                    .unwrap()
            };
            let buffer_infos: Vec<_> = buffers
                .iter()
                .map(|x| {
                    let inner = x.inner.as_ref().unwrap();
                    ash::vk::DescriptorBufferInfo::builder()
                        .buffer(inner.buffer().handle())
                        .offset(inner.offset())
                        .range(inner.size())
                        .build()
                })
                .collect();
            let write_descriptor_set = ash::vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set.handle())
                .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos)
                .build();
            unsafe {
                let writes = [write_descriptor_set];
                let copies = [];
                (device.fns().v1_0.update_descriptor_sets)(
                    device.handle(),
                    writes.len() as u32,
                    writes.as_ptr(),
                    copies.len() as u32,
                    copies.as_ptr(),
                )
            }
            let sets = [descriptor_set.handle()];
            let first_set = 0;
            let dynamic_offsets = [];
            unsafe {
                (device.fns().v1_0.cmd_bind_descriptor_sets)(
                    builder.handle(),
                    ash::vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout.handle(),
                    first_set,
                    sets.len() as u32,
                    sets.as_ptr(),
                    dynamic_offsets.len() as u32,
                    dynamic_offsets.as_ptr(),
                );
            }
        }
        if !push_consts.is_empty() {
            let offset = 0;
            unsafe {
                (device.fns().v1_0.cmd_push_constants)(
                    builder.handle(),
                    pipeline_layout.handle(),
                    ash::vk::ShaderStageFlags::COMPUTE,
                    offset,
                    push_consts.len().try_into().unwrap(),
                    push_consts.as_ptr() as *const _,
                );
            }
        }
        unsafe {
            builder.dispatch([groups, 1, 1])?;
        }
        self.buffers
            .extend(buffers.iter().map(|x| x.inner.as_ref().unwrap().clone()));
        for (buffer, slice_desc) in buffers.iter().zip(kernel_desc.slice_descs.iter()) {
            if slice_desc.mutable {
                buffer.epoch.store(self.epoch, Ordering::SeqCst);
            }
        }
        if let Some(debug_printf_panic) = debug_printf_panic {
            self.debug_kernel_desc_panic
                .replace((kernel_desc.clone(), debug_printf_panic));
        }
        Ok(())
    }
    unsafe fn finish(&mut self) {
        self.buffers.clear();
        self.debug_kernel_desc_panic.take();
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

struct Worker {
    queue: Arc<Queue>,
    index: usize,
    semaphore: Arc<Semaphore>,
    empty: Arc<AtomicBool>,
    pending: Arc<AtomicU64>,
    ready_frame: Arc<Mutex<Frame>>,
    pending_frame: Frame,
    engine_exited: Arc<AtomicBool>,
    worker_exited: Arc<AtomicBool>,
}

impl Worker {
    fn new(queue: Arc<Queue>, index: usize) -> Result<Self> {
        let semaphore = Arc::new(new_semaphore(queue.device())?);
        let empty = Arc::new(AtomicBool::new(true));
        let pending = Arc::new(AtomicU64::default());
        let mut ready_frame = Frame::new(queue.clone())?;
        ready_frame.epoch = 1;
        let ready_frame = Arc::new(Mutex::new(ready_frame));
        let pending_frame = Frame::new(queue.clone())?;
        let engine_exited = Arc::new(AtomicBool::default());
        let worker_exited = Arc::new(AtomicBool::default());
        Ok(Self {
            queue,
            index,
            semaphore,
            empty,
            pending,
            ready_frame,
            pending_frame,
            engine_exited,
            worker_exited,
        })
    }
    fn run(&mut self) {
        let id = DeviceId {
            index: self.index,
            handle: self.queue.device().handle().as_raw().try_into().unwrap(),
        };
        loop {
            while self.empty.load(Ordering::SeqCst) {
                if self.engine_exited.load(Ordering::SeqCst) {
                    return;
                }
                std::hint::spin_loop();
            }
            {
                let mut ready_frame = self.ready_frame.lock();
                self.pending_frame.epoch = ready_frame.epoch + 1;
                self.empty.store(true, Ordering::SeqCst);
                std::mem::swap(&mut *ready_frame, &mut self.pending_frame);
            }
            self.pending
                .store(self.pending_frame.epoch, Ordering::SeqCst);
            let command_buffer = self
                .pending_frame
                .command_buffer_builder
                .take()
                .unwrap()
                .build()
                .unwrap();
            let _messenger = if let Some((kernel_desc, panicked)) =
                self.pending_frame.debug_kernel_desc_panic.take()
            {
                Some(
                    DebugUtilsMessenger::new(
                        self.queue.device().instance().clone(),
                        DebugUtilsMessengerCreateInfo {
                            message_severity: DebugUtilsMessageSeverity::INFO,
                            message_type: DebugUtilsMessageType::VALIDATION,
                            ..DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                                DebugUtilsMessengerCallback::new(
                                    move |_severity, _message_type, data| {
                                        if let Some(layer_prefix) = data.message_id_name.as_ref() {
                                            if layer_prefix.contains("DEBUG-PRINTF") {
                                                eprintln!(
                                                    "[{id:?} {}] {}",
                                                    kernel_desc.name, data.message,
                                                );
                                                if data.message.contains("[Rust panicked at ") {
                                                    panicked.store(true, Ordering::SeqCst);
                                                }
                                            }
                                        }
                                    },
                                )
                            })
                        },
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            self.queue.with(|mut guard| unsafe {
                queue_submit(
                    &self.queue,
                    &mut guard,
                    &command_buffer,
                    &self.semaphore,
                    self.pending_frame.epoch,
                )
                .unwrap();
            });
            loop {
                let result = unsafe {
                    wait_semaphore(
                        self.queue.device(),
                        &self.semaphore,
                        self.pending_frame.epoch,
                    )
                };
                match result {
                    ash::vk::Result::SUCCESS => break,
                    ash::vk::Result::TIMEOUT => std::hint::spin_loop(),
                    _ => result.result().unwrap(),
                }
            }
            unsafe {
                self.pending_frame.finish();
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.queue.with(|mut guard| guard.wait_idle());
        self.worker_exited.store(true, Ordering::SeqCst);
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
            inner.mapped_slice().is_ok()
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
        use vulkano::Validated;
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
                memory_type_filter: MemoryTypeFilter {
                    required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                    preferred_flags: MemoryPropertyFlags::empty(),
                    not_preferred_flags: MemoryPropertyFlags::HOST_VISIBLE,
                },
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
            let mut requirements = *raw_buffer.memory_requirements();
            requirements.layout = requirements.layout.align_to(align).unwrap();
            requirements.prefers_dedicated_allocation = false;
            let memory_alloc = engine
                .memory_allocator
                .allocate(requirements, AllocationType::Unknown, allocation_info, None)
                .map_err(|e| {
                    if let MemoryAllocatorError::AllocateDeviceMemory(Validated::Error(
                        VulkanError::OutOfDeviceMemory,
                    )) = e
                    {
                        Error::new(OutOfDeviceMemory(engine.id())).context(e)
                    } else {
                        e.into()
                    }
                })?;
            let resource_memory = unsafe {
                ResourceMemory::from_allocation(engine.memory_allocator.clone(), memory_alloc)
            };
            let buffer = raw_buffer
                .bind_memory(resource_memory)
                .map_err(|(e, _, _)| e)?;
            Some(Subbuffer::new(Arc::new(buffer)))
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
        let buffer_epoch = self.epoch.load(Ordering::SeqCst);
        if self.host_visible() {
            engine.wait_epoch(buffer_epoch)?;
            buffer.write().unwrap().copy_from_slice(data);
            return Ok(());
        }
        let mut offset = 0;
        for chunk in data.chunks(Self::HOST_BUFFER_SIZE) {
            let mut host_buffer = engine.host_buffer_receiver.recv().unwrap();
            let size = chunk.len() as u64;
            let buffer_slice = buffer.clone().slice(offset..offset + size);
            let host_slice = host_buffer.inner.clone().slice(0..size);
            engine.wait_epoch(host_buffer.epoch)?;
            host_slice.write().unwrap().copy_from_slice(chunk);
            engine.wait_pending(buffer_epoch)?;
            unsafe {
                engine.transfer(host_slice, buffer_slice, &mut host_buffer, Some(self))?;
            }
            engine.host_buffer_sender.send(host_buffer).unwrap();
            offset += size;
        }
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
        let buffer_epoch = self.epoch.load(Ordering::SeqCst);
        if self.host_visible() {
            engine.wait_epoch(buffer_epoch)?;
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
                engine.wait_epoch(host_buffer.epoch)?;
                let size = chunk.len() as u64;
                let buffer_slice = buffer.clone().slice(offset..offset + size);
                let host_slice = host_buffer.inner.clone().slice(0..size);
                engine.wait_pending(buffer_epoch)?;
                unsafe {
                    engine.transfer(buffer_slice, host_slice.clone(), &mut host_buffer, None)?;
                }
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
                engine.wait_epoch(host_buffer.epoch)?;
                chunk.copy_from_slice(&host_slice.read().unwrap());
                engine.host_buffer_sender.send(host_buffer).unwrap();
            }
        }
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
        let buffer1_epoch = self.epoch.load(Ordering::SeqCst);
        let engine2 = &dst.engine;
        let buffer2_epoch = dst.epoch.load(Ordering::SeqCst);
        if self.host_visible() && dst.host_visible() {
            engine1.wait_epoch(buffer1_epoch)?;
            engine2.wait_epoch(buffer2_epoch)?;
            buffer2
                .write()
                .unwrap()
                .copy_from_slice(&buffer1.read().unwrap());
            return Ok(());
        } else if self.host_visible() {
            engine1.wait_epoch(buffer1_epoch)?;
            return dst.upload(&buffer1.read().unwrap());
        } else if dst.host_visible() {
            engine2.wait_epoch(buffer2_epoch)?;
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
                let buffer_slice1 = buffer1.clone().slice(offset..offset + size);
                let host_slice1 = host_buffer1.inner.clone().slice(0..size);
                engine1.wait_epoch(host_buffer1.epoch)?;
                engine1.wait_pending(buffer1_epoch)?;
                unsafe {
                    engine1.transfer(
                        buffer_slice1,
                        host_slice1.clone(),
                        &mut host_buffer1,
                        None,
                    )?;
                }
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
                engine1.wait_epoch(host_buffer1.epoch)?;
                engine2.wait_epoch(host_buffer2.epoch)?;
                host_slice2
                    .write()
                    .unwrap()
                    .copy_from_slice(&host_slice1.read().unwrap());
                engine1.host_buffer_sender.send(host_buffer1).unwrap();
                engine2.wait_pending(buffer2_epoch)?;
                unsafe {
                    engine2.transfer(host_slice2, buffer_slice2, &mut host_buffer2, Some(dst))?;
                }
                engine2.host_buffer_sender.send(host_buffer2).unwrap();
            } else if size == 0 {
                break;
            }
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

// adapted from https://docs.rs/vulkano/0.33.0/src/vulkano/shader/mod.rs.html#229-312
unsafe fn shader_module_from_words(
    device: Arc<Device>,
    words: &[u32],
) -> Result<Arc<ShaderModule>, VulkanError> {
    let handle = {
        let infos = ash::vk::ShaderModuleCreateInfo {
            flags: ash::vk::ShaderModuleCreateFlags::empty(),
            code_size: std::mem::size_of_val(words),
            p_code: words.as_ptr(),
            ..Default::default()
        };

        let fns = device.fns();
        let mut output = MaybeUninit::uninit();
        unsafe {
            (fns.v1_0.create_shader_module)(
                device.handle(),
                &infos,
                std::ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        }
    };
    Ok(unsafe { ShaderModule::from_handle(device, handle, ShaderModuleCreateInfo::new(words)) })
}

impl KernelInner {
    fn new(engine: &Arc<Engine>, desc: Arc<KernelDesc>) -> Result<Self> {
        /*use vulkano::shader::spirv::Spirv;
        let entry_point = vulkano::shader::reflect::entry_points(&Spirv::new(&desc.spirv).unwrap())
            .next()
            .unwrap();
        dbg!(entry_point);*/
        use vulkano::{
            descriptor_set::layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
            pipeline::layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
            shader::spirv::ExecutionModel,
        };
        let device = engine.queue.device();
        /*let descriptor_binding_requirements = desc
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
        .collect();*/
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
        let shader_module = unsafe { shader_module_from_words(device.clone(), &desc.spirv)? };
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
        let enabled_features = device.enabled_features();
        let descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings,
            flags: if enabled_features.descriptor_buffer_push_descriptors {
                DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR
            } else {
                DescriptorSetLayoutCreateFlags::empty()
            },
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
        let required_subgroup_size = if enabled_features.subgroup_size_control {
            engine.info.subgroup_threads
        } else {
            None
        };
        let stage_create_info = PipelineShaderStageCreateInfo {
            required_subgroup_size,
            ..PipelineShaderStageCreateInfo::new(
                shader_module
                    .single_entry_point_with_execution(ExecutionModel::GLCompute)
                    .unwrap(),
            )
        };
        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            cache,
            ComputePipelineCreateInfo::stage_layout(stage_create_info, pipeline_layout),
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
        groups: u32,
        buffers: &[Arc<Self::DeviceBuffer>],
        push_consts: Vec<u8>,
        debug_printf_panic: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        let engine = &self.engine;
        if let Some(epoch) = buffers.iter().map(|x| x.epoch.load(Ordering::SeqCst)).max() {
            engine.wait_pending(epoch)?;
        }
        unsafe {
            engine.compute(
                &self.desc,
                &self.compute_pipeline,
                groups,
                buffers,
                &push_consts,
                debug_printf_panic,
            )
        }
    }
    fn desc(&self) -> &Arc<KernelDesc> {
        &self.desc
    }
}

struct CommandPoolBufferAllocator {
    pool: CommandPool,
    alloc: CommandPoolAlloc,
}

impl CommandPoolBufferAllocator {
    fn new(queue: &Queue) -> Result<Arc<Self>> {
        let device = queue.device();
        let pool = CommandPool::new(
            device.clone(),
            CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                flags: CommandPoolCreateFlags::TRANSIENT
                    | CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            },
        )?;
        let alloc = pool
            .allocate_command_buffers(CommandBufferAllocateInfo {
                level: CommandBufferLevel::Primary,
                command_buffer_count: 1,
                ..Default::default()
            })
            .unwrap()
            .next()
            .unwrap();
        Ok(Arc::new(Self { pool, alloc }))
    }
    unsafe fn reset(self: &mut Arc<Self>) -> Result<(), ash::vk::Result> {
        let this = Arc::get_mut(self).unwrap();
        unsafe {
            (this.device().fns().v1_0.reset_command_buffer)(this.alloc.handle(), Default::default())
                .result()
        }
    }
}

unsafe impl DeviceOwned for CommandPoolBufferAllocator {
    fn device(&self) -> &Arc<Device> {
        self.pool.device()
    }
}

struct ArcCommandPoolBufferAllocator(Arc<CommandPoolBufferAllocator>);

unsafe impl DeviceOwned for ArcCommandPoolBufferAllocator {
    fn device(&self) -> &Arc<Device> {
        self.0.device()
    }
}

unsafe impl CommandBufferAllocator for ArcCommandPoolBufferAllocator {
    type Iter = std::iter::Once<Self::Builder>;
    type Builder = Self::Alloc;
    type Alloc = CommandPoolBufferAllocatorAlloc;

    fn allocate(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
        command_buffer_count: u32,
    ) -> Result<Self::Iter, vulkano::VulkanError> {
        let allocator = self.0.clone();
        debug_assert_eq!(allocator.pool.queue_family_index(), queue_family_index);
        debug_assert_eq!(level, CommandBufferLevel::Primary);
        debug_assert_eq!(command_buffer_count, 1);
        Ok(std::iter::once(CommandPoolBufferAllocatorAlloc {
            allocator,
        }))
    }
}

unsafe impl Send for CommandPoolBufferAllocator {}
unsafe impl Sync for CommandPoolBufferAllocator {}

struct CommandPoolBufferAllocatorAlloc {
    allocator: Arc<CommandPoolBufferAllocator>,
}

unsafe impl DeviceOwned for CommandPoolBufferAllocatorAlloc {
    fn device(&self) -> &Arc<Device> {
        self.allocator.device()
    }
}

unsafe impl CommandBufferBuilderAlloc for CommandPoolBufferAllocatorAlloc {
    type Alloc = Self;
    fn inner(&self) -> &CommandPoolAlloc {
        &self.allocator.alloc
    }
    fn into_alloc(self) -> Self::Alloc {
        self
    }
    fn queue_family_index(&self) -> u32 {
        self.allocator.pool.queue_family_index()
    }
}

unsafe impl CommandBufferAlloc for CommandPoolBufferAllocatorAlloc {
    fn inner(&self) -> &CommandPoolAlloc {
        &self.allocator.alloc
    }
    fn queue_family_index(&self) -> u32 {
        self.allocator.pool.queue_family_index()
    }
}
