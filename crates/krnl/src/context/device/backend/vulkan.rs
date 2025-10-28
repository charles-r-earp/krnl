use super::{BufferRange, DeviceOwned, DeviceSpecifier, Properties};
use crate::{
    Result,
    context::device::Features,
    kernel::{KernelCreateInfo, KernelDesc, KernelKey},
};
use core::u64;
use fxhash::FxHashMap;
use parking_lot::{ArcMutexGuard, Mutex, RawMutex, RwLock};
use std::{
    collections::VecDeque,
    ffi::CString,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use vk_mem::Alloc as _;

pub struct Backend {
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

impl super::Backend for Backend {
    type Device = Device;
    type Event = Event;
    type Buffer = Buffer;
    type Slice = Slice;
    type Kernel = Kernel;
    fn create() -> Result<Arc<Self>> {
        let entry = unsafe { ash::Entry::load().unwrap() };
        let engine_name: CString = "krnl".parse().unwrap();
        let instance_version = unsafe { entry.try_enumerate_instance_version().unwrap() };
        let api_version = if let Some(instance_version) = instance_version {
            if matches!(
                instance_version,
                ash::vk::API_VERSION_1_0 | ash::vk::API_VERSION_1_1 | ash::vk::API_VERSION_1_2
            ) {
                todo!();
            }
            ash::vk::API_VERSION_1_3
        } else {
            todo!()
        };
        let app_info = ash::vk::ApplicationInfo::default()
            .engine_name(&engine_name)
            .api_version(api_version);
        let instance_create_info =
            ash::vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_create_info, None).unwrap() };
        Ok(Arc::new(Self {
            instance,
            _entry: entry,
        }))
    }
}

struct RawDevice {
    device: ash::Device,
    //physical_device: ash::vk::PhysicalDevice,
    backend: Arc<Backend>,
    queue_family_indices: Vec<u32>,
    features: Features,
    properties: Properties,
}

impl RawDevice {
    unsafe fn new(
        backend: Arc<Backend>,
        physical_device: ash::vk::PhysicalDevice,
        queue_family_indices: Vec<u32>,
    ) -> Result<Arc<Self>> {
        let queue_priorities = [1f32];
        let queue_create_infos: Vec<_> = queue_family_indices
            .iter()
            .copied()
            .map(|queue_family_index| {
                ash::vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&queue_priorities)
            })
            .collect();
        let mut supported_physical_device_features2 = ash::vk::PhysicalDeviceFeatures2::default();
        let mut supported_physical_device_vulkan11_features =
            ash::vk::PhysicalDeviceVulkan11Features::default();
        let mut supported_physical_device_vulkan12_features =
            ash::vk::PhysicalDeviceVulkan12Features::default();
        supported_physical_device_features2 = supported_physical_device_features2
            .push_next(&mut supported_physical_device_vulkan11_features)
            .push_next(&mut supported_physical_device_vulkan12_features);
        unsafe {
            backend.instance.get_physical_device_features2(
                physical_device,
                &mut supported_physical_device_features2,
            );
        };
        let supported_physical_device_features = supported_physical_device_features2.features;
        let physical_device_features = ash::vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(true)
            .shader_int16(supported_physical_device_features.shader_int16 != 0)
            .shader_int64(supported_physical_device_features.shader_int64 != 0)
            .shader_float64(supported_physical_device_features.shader_float64 != 0);
        let mut vulkan11_features = ash::vk::PhysicalDeviceVulkan11Features::default()
            .variable_pointers_storage_buffer(true)
            .storage_buffer16_bit_access(
                supported_physical_device_vulkan11_features.storage_buffer16_bit_access != 0,
            )
            .storage_push_constant16(
                supported_physical_device_vulkan11_features.storage_push_constant16 != 0,
            );
        let mut vulkan12_features = ash::vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .vulkan_memory_model(true)
            .shader_int8(supported_physical_device_vulkan12_features.shader_int8 != 0)
            .storage_buffer8_bit_access(
                supported_physical_device_vulkan12_features.storage_buffer8_bit_access != 0,
            )
            .storage_push_constant8(
                supported_physical_device_vulkan12_features.storage_push_constant8 != 0,
            );
        let mut vulkan13_features = ash::vk::PhysicalDeviceVulkan13Features::default()
            .maintenance4(true)
            .subgroup_size_control(true);
        let mut physical_device_features2 = ash::vk::PhysicalDeviceFeatures2::default()
            .features(physical_device_features)
            .push_next(&mut vulkan11_features)
            .push_next(&mut vulkan12_features)
            .push_next(&mut vulkan13_features);
        let device_create_info = ash::vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .push_next(&mut physical_device_features2);
        let mut physical_device_properties = ash::vk::PhysicalDeviceProperties2::default();
        let mut vulkan13_properties = ash::vk::PhysicalDeviceVulkan13Properties::default();
        // TODO: This doesn't work
        //physical_device_properties.push_next(&mut vulkan13_properties);
        physical_device_properties.p_next = <*mut _>::cast(&mut vulkan13_properties);
        unsafe {
            backend
                .instance
                .get_physical_device_properties2(physical_device, &mut physical_device_properties);
        }
        let mut features = Features::default();
        if supported_physical_device_vulkan12_features.shader_int8 != 0
            && supported_physical_device_vulkan12_features.storage_buffer8_bit_access != 0
            && supported_physical_device_vulkan12_features.storage_push_constant8 != 0
        {
            features.insert(Features::INT8);
        }
        if supported_physical_device_features.shader_int16 != 0
            && supported_physical_device_vulkan11_features.storage_buffer16_bit_access != 0
            && supported_physical_device_vulkan11_features.storage_push_constant16 != 0
        {
            features.insert(Features::INT16);
        }
        if supported_physical_device_features.shader_int64 != 0 {
            features.insert(Features::INT64);
        }
        if supported_physical_device_features.shader_float64 != 0 {
            features.insert(Features::FLOAT64);
        }
        let properties = Properties {
            max_buffer_size: physical_device_properties
                .properties
                .limits
                .max_storage_buffer_range,
            min_subgroup_threads: vulkan13_properties.min_subgroup_size,
            max_subgroup_threads: vulkan13_properties.max_subgroup_size,
        };
        let device = unsafe {
            backend
                .instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };
        Ok(Arc::new(RawDevice {
            device,
            //physical_device,
            backend,
            queue_family_indices,
            features,
            properties,
        }))
    }
}

impl Drop for RawDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

unsafe impl Send for RawDevice {}
unsafe impl Sync for RawDevice {}

pub struct Device {
    raw: Arc<RawDevice>,
    compute_queues: Vec<Arc<Queue>>,
    transfer_queue: Option<Arc<Queue>>,
    buffer_allocator: Arc<BufferAllocator>,
    //command_buffer_allocators: FxHashMap<u32, Arc<CommandBufferAllocator>>,
    queue_selector: Option<Mutex<QueueSelector>>,
    kernels: Mutex<FxHashMap<KernelKey, Result<Arc<RawKernel>, CompileError>>>,
}

impl super::Device for Device {
    type Backend = Backend;
    type Event = Event;
    fn create(backend: Arc<Self::Backend>, specifier: DeviceSpecifier) -> Result<Arc<Self>> {
        let DeviceSpecifier::Index(index) = specifier;
        let instance = &backend.instance;
        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let physical_device = physical_devices[index];
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_info = QueueFamilyInfo::new(&queue_family_properties).unwrap();
        let device = unsafe {
            RawDevice::new(
                backend,
                physical_device,
                queue_family_info.queue_family_indices().collect(),
            )?
        };
        let descriptor_pool_config = Some(DescriptorPoolConfig {
            max_sets: 1,
            descriptor_count: 16,
        });
        let command_buffer_allocators: FxHashMap<u32, Arc<CommandBufferAllocator>> =
            queue_family_info
                .queue_family_indices()
                .map(|queue_family_index| unsafe {
                    (
                        queue_family_index,
                        CommandBufferAllocator::new(
                            device.clone(),
                            queue_family_index,
                            descriptor_pool_config,
                        ),
                    )
                })
                .collect();
        let mut queues = Vec::new();
        let compute_queues = 1;
        for _ in 0..compute_queues {
            let queue_family_index = queue_family_info.compute;
            let queue = unsafe {
                Queue::new(
                    device.clone(),
                    command_buffer_allocators[&queue_family_index].clone(),
                )?
            };
            queues.push(queue);
        }
        if let Some(queue_family_index) = queue_family_info.graphics {
            let queue = unsafe {
                Queue::new(
                    device.clone(),
                    command_buffer_allocators[&queue_family_index].clone(),
                )?
            };
            queues.push(queue);
        }
        let transfer_queue = queue_family_info
            .transfer
            .map(|queue_family_index| unsafe {
                Queue::new(
                    device.clone(),
                    command_buffer_allocators[&queue_family_index].clone(),
                )
            })
            .transpose()?;
        let buffer_allocator = unsafe { BufferAllocator::new(device.clone(), physical_device)? };
        let queue_selector = if queues.len() > 1 {
            Some(Mutex::new(QueueSelector::new(queues.len())))
        } else {
            None
        };
        let kernels = Mutex::default();
        Ok(Arc::new(Self {
            raw: device,
            compute_queues: queues,
            transfer_queue,
            buffer_allocator,
            //command_buffer_allocators,
            queue_selector,
            kernels,
        }))
    }
    fn features(&self) -> Features {
        self.raw.features
    }
    fn event(self: &Arc<Self>) -> Arc<Self::Event> {
        let queue_events = self.queues().cloned().map(|x| x.event()).collect();
        Arc::new(Event {
            device: self.clone(),
            queue_events,
        })
    }
    fn properties(&self) -> &Properties {
        &self.raw.properties
    }
}

#[derive(Debug)]
struct QueueFamilyInfo {
    compute: u32,
    graphics: Option<u32>,
    transfer: Option<u32>,
}

impl QueueFamilyInfo {
    fn new(properties: &[ash::vk::QueueFamilyProperties]) -> Option<Self> {
        let compute_only = properties
            .iter()
            .position(|p| {
                p.queue_flags.contains(ash::vk::QueueFlags::COMPUTE)
                    && !p.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
            })
            .map(|x| x as u32);
        let graphics = properties
            .iter()
            .position(|p| {
                p.queue_flags.contains(ash::vk::QueueFlags::COMPUTE)
                    && p.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
            })
            .map(|x| x as u32);
        let transfer = properties
            .iter()
            .position(|p| {
                p.queue_flags.contains(ash::vk::QueueFlags::TRANSFER)
                    && !p.queue_flags.contains(ash::vk::QueueFlags::COMPUTE)
                    && !p.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
            })
            .map(|x| x as u32);
        let (compute, graphics) = if let Some(compute) = compute_only {
            (compute, graphics)
        } else {
            (graphics?, None)
        };
        Some(Self {
            compute,
            graphics,
            transfer,
        })
    }
    fn queue_family_indices(&self) -> impl Iterator<Item = u32> {
        std::iter::once(self.compute)
            .chain(self.graphics)
            .chain(self.transfer)
    }
}

impl Device {
    fn transfer_queue(&self) -> &Arc<Queue> {
        self.transfer_queue
            .as_ref()
            .unwrap_or(self.compute_queues.first().unwrap())
    }
    fn queues(&self) -> impl Iterator<Item = &Arc<Queue>> + '_ {
        self.compute_queues
            .iter()
            .chain(self.transfer_queue.as_ref())
    }
    fn poll(&self) -> Result<()> {
        for queue in self.queues() {
            queue.poll()?;
        }
        Ok(())
    }
}

pub struct Event {
    device: Arc<Device>,
    queue_events: Vec<QueueEvent>,
}

impl DeviceOwned for Event {
    type Device = Device;
    fn device(&self) -> &Arc<Self::Device> {
        &self.device
    }
}

impl super::Event for Event {
    fn wait(&self) -> Result<()> {
        for event in self.queue_events.iter() {
            event.wait()?;
        }
        self.device.poll()?;
        Ok(())
    }
}

struct QueueEntry {
    index: usize,
    encode: Arc<Mutex<()>>,
}

struct QueueSelector {
    entries: VecDeque<QueueEntry>,
}

impl QueueSelector {
    fn new(queues: usize) -> Self {
        let entries = (0..queues)
            .map(|index| QueueEntry {
                index,
                encode: Arc::default(),
            })
            .collect();
        Self { entries }
    }
    fn select(&mut self, queues: &[Arc<Queue>]) -> Result<(usize, ArcMutexGuard<RawMutex, ()>)> {
        let mut index = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            if queues[entry.index].raw.is_idle()? && !entry.encode.is_locked() {
                index = i;
                break;
            }
        }
        let entry = self.entries.remove(index).unwrap();
        let queue_index = entry.index;
        let encode = entry.encode.lock_arc();
        self.entries.push_back(entry);
        Ok((queue_index, encode))
    }
}

#[derive(Clone, Copy, Debug)]
struct WaitTimeout;

struct RawQueue {
    device: Arc<RawDevice>,
    queue: Mutex<ash::vk::Queue>,
    family_index: u32,
    semaphore: ash::vk::Semaphore,
    semaphore_value: AtomicU64,
}

impl RawQueue {
    unsafe fn new(device: Arc<RawDevice>, family_index: u32) -> Arc<Self> {
        let queue = Mutex::new(unsafe { device.device.get_device_queue(family_index, 0) });
        let mut semaphore_type_create_info = ash::vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(ash::vk::SemaphoreType::TIMELINE);
        let semaphore_create_info =
            ash::vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type_create_info);
        let semaphore = unsafe {
            device
                .device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap()
        };
        let semaphore_value = AtomicU64::default();
        Arc::new(Self {
            device,
            queue,
            family_index,
            semaphore,
            semaphore_value,
        })
    }
    fn is_idle(&self) -> Result<bool> {
        let current_value = unsafe {
            self.device
                .device
                .get_semaphore_counter_value(self.semaphore)
                .unwrap()
        };
        let idle = current_value == self.semaphore_value.load(Ordering::SeqCst);
        Ok(idle)
    }
    fn wait(&self, wait_value: u64) -> Result<()> {
        self.wait_timeout(wait_value, u64::MAX)?.unwrap();
        Ok(())
    }
    fn try_wait(&self, wait_value: u64) -> Result<Result<(), WaitTimeout>> {
        self.wait_timeout(wait_value, 0)
    }
    fn wait_timeout(&self, wait_value: u64, timeout: u64) -> Result<Result<(), WaitTimeout>> {
        let semaphores = [self.semaphore];
        let values = [wait_value];
        let info = ash::vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        let result = unsafe { self.device.device.wait_semaphores(&info, timeout) };
        if let Err(ash::vk::Result::TIMEOUT) = result {
            return Ok(Err(WaitTimeout));
        }
        result.unwrap();
        Ok(Ok(()))
    }
    unsafe fn submit(&self, command_buffer: ash::vk::CommandBuffer) -> Result<u64> {
        let queue = self.queue.lock();
        let command_buffers = [command_buffer];
        let semaphores = [self.semaphore];
        let semaphore_value = self
            .semaphore_value
            .load(Ordering::SeqCst)
            .checked_add(1)
            .unwrap();
        let semaphore_values = [semaphore_value];
        let mut timeline_submit_info = ash::vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&semaphore_values);
        let submit_info = ash::vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline_submit_info);
        unsafe {
            self.device
                .device
                .queue_submit(*queue, &[submit_info], ash::vk::Fence::null())
                .unwrap();
        }
        self.semaphore_value
            .store(semaphore_value, Ordering::SeqCst);
        Ok(semaphore_value)
    }
}

impl Drop for RawQueue {
    fn drop(&mut self) {
        let value = self.semaphore_value.load(Ordering::SeqCst);
        self.wait(value).unwrap();
        let device = &self.device.device;
        unsafe {
            device.destroy_semaphore(self.semaphore, None);
        }
    }
}

struct Queue {
    raw: Arc<RawQueue>,
    command_buffer_allocator: Arc<CommandBufferAllocator>,
    command_buffers: Mutex<VecDeque<CommandBuffer>>,
}

impl Queue {
    unsafe fn new(
        device: Arc<RawDevice>,
        command_buffer_allocator: Arc<CommandBufferAllocator>,
    ) -> Result<Arc<Self>> {
        let raw = unsafe { RawQueue::new(device, command_buffer_allocator.queue_family_index) };
        let command_buffers = Mutex::new(VecDeque::default());
        Ok(Arc::new(Self {
            raw,
            command_buffer_allocator,
            command_buffers,
        }))
    }
    fn poll(&self) -> Result<()> {
        let mut command_buffers = self.command_buffers.lock();
        loop {
            if let Some(command_buffer) = command_buffers.front() {
                if command_buffer.try_wait()?.is_ok() {
                    let mut command_buffer = command_buffers.pop_front().unwrap();
                    unsafe {
                        command_buffer.reset()?;
                        self.command_buffer_allocator.recycle(command_buffer.raw)?;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(())
    }
    fn event(self: Arc<Self>) -> QueueEvent {
        let semaphore_value = self.raw.semaphore_value.load(Ordering::SeqCst);
        QueueEvent {
            queue: self,
            semaphore_value,
        }
    }
    fn transfer(
        self: &Arc<Self>,
        src: Arc<RawBuffer>,
        src_offset: usize,
        dst: Arc<RawBuffer>,
        dst_offset: usize,
        len: usize,
    ) -> Result<SubmitCommandBuffer> {
        let mut command_buffer = self.command_buffer_allocator.allocate()?;
        unsafe {
            command_buffer.begin()?;
            command_buffer.transfer(&src, src_offset, &dst, dst_offset, len);
        }
        Ok(SubmitCommandBuffer {
            queue: self.clone(),
            command_buffer,
            buffers: vec![src, dst],
            kernel: None,
        })
    }
    fn kernel(
        self: &Arc<Self>,
        kernel: Arc<RawKernel>,
        groups: u32,
        buffers: &[BufferBinding],
        push_constants: &[u8],
    ) -> Result<SubmitCommandBuffer> {
        let mut command_buffer = self.command_buffer_allocator.allocate()?;
        unsafe {
            command_buffer.begin()?;
            command_buffer.kernel(&kernel, groups, buffers, push_constants);
        }
        let buffers = buffers
            .iter()
            .map(|x| x.slice.buffer.raw.as_ref().unwrap().clone())
            .collect();
        Ok(SubmitCommandBuffer {
            queue: self.clone(),
            command_buffer,
            buffers,
            kernel: Some(kernel),
        })
    }
}

#[derive(Clone)]
struct QueueEvent {
    queue: Arc<Queue>,
    semaphore_value: u64,
}

impl QueueEvent {
    fn wait(&self) -> Result<()> {
        self.queue.raw.wait(self.semaphore_value)
    }
    fn try_wait(&self) -> Result<Result<(), WaitTimeout>> {
        self.queue.raw.try_wait(self.semaphore_value)
    }
}

#[derive(Clone, Copy)]
struct DescriptorPoolConfig {
    max_sets: u32,
    descriptor_count: u32,
}

struct CommandBufferAllocator {
    device: Arc<RawDevice>,
    command_buffers: Mutex<VecDeque<RawCommandBuffer>>,
    queue_family_index: u32,
    descriptor_pool_config: Option<DescriptorPoolConfig>,
}

impl CommandBufferAllocator {
    unsafe fn new(
        device: Arc<RawDevice>,
        queue_family_index: u32,
        descriptor_pool_config: Option<DescriptorPoolConfig>,
    ) -> Arc<Self> {
        Arc::new(Self {
            device,
            command_buffers: Mutex::default(),
            queue_family_index,
            descriptor_pool_config,
        })
    }
    fn allocate(&self) -> Result<RawCommandBuffer> {
        {
            let mut command_buffers = self.command_buffers.lock();
            if let Some(command_buffer) = command_buffers.pop_front() {
                return Ok(command_buffer);
            }
        }
        unsafe {
            RawCommandBuffer::new(
                self.device.clone(),
                self.queue_family_index,
                self.descriptor_pool_config,
            )
        }
    }
    unsafe fn recycle(&self, mut command_buffer: RawCommandBuffer) -> Result<()> {
        unsafe { command_buffer.reset()? };
        let mut command_buffers = self.command_buffers.lock();
        command_buffers.push_back(command_buffer);
        Ok(())
    }
}

struct RawCommandBuffer {
    device: Arc<RawDevice>,
    command_pool: ash::vk::CommandPool,
    queue_family_index: u32,
    command_buffer: Option<ash::vk::CommandBuffer>,
    descriptor_pool: Option<ash::vk::DescriptorPool>,
}

impl Drop for RawCommandBuffer {
    fn drop(&mut self) {
        let device = &self.device.device;
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            if let Some(descriptor_pool) = self.descriptor_pool {
                device.destroy_descriptor_pool(descriptor_pool, None);
            }
        }
    }
}

impl RawCommandBuffer {
    unsafe fn new(
        device: Arc<RawDevice>,
        queue_family_index: u32,
        descriptor_pool_config: Option<DescriptorPoolConfig>,
    ) -> Result<Self> {
        let pool_create_info = ash::vk::CommandPoolCreateInfo::default()
            .flags(ash::vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family_index);
        let command_pool = unsafe {
            device
                .device
                .create_command_pool(&pool_create_info, None)
                .unwrap()
        };
        let descriptor_pool = if let Some(DescriptorPoolConfig {
            max_sets,
            descriptor_count,
        }) = descriptor_pool_config
        {
            let pool_size = ash::vk::DescriptorPoolSize {
                ty: ash::vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count,
            };
            let pool_sizes = [pool_size];
            let pool_create_info = ash::vk::DescriptorPoolCreateInfo::default()
                .max_sets(max_sets)
                .pool_sizes(&pool_sizes);
            let pool = unsafe {
                device
                    .device
                    .create_descriptor_pool(&pool_create_info, None)
                    .unwrap()
            };
            Some(pool)
        } else {
            None
        };
        Ok(RawCommandBuffer {
            device,
            command_pool,
            queue_family_index,
            command_buffer: None,
            descriptor_pool,
        })
    }
    unsafe fn reset(&mut self) -> Result<()> {
        let device = &self.device.device;
        unsafe {
            device
                .reset_command_pool(self.command_pool, ash::vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        if let Some(descriptor_pool) = self.descriptor_pool {
            unsafe {
                device
                    .reset_descriptor_pool(
                        descriptor_pool,
                        ash::vk::DescriptorPoolResetFlags::empty(),
                    )
                    .unwrap();
            }
        }
        Ok(())
    }
    unsafe fn begin(&mut self) -> Result<()> {
        let command_buffer_allocate_info = ash::vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(self.command_pool);
        let device = &self.device.device;
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()
        };
        let command_buffer = command_buffers[0];
        let begin_info = ash::vk::CommandBufferBeginInfo::default()
            .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }
        self.command_buffer.replace(command_buffer);
        Ok(())
    }
    unsafe fn transfer(
        &mut self,
        src: &RawBuffer,
        src_offset: usize,
        dst: &RawBuffer,
        dst_offset: usize,
        len: usize,
    ) {
        let command_buffer = self.command_buffer.unwrap();
        let device = &self.device.device;
        unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                src.buffer,
                dst.buffer,
                &[ash::vk::BufferCopy {
                    src_offset: src_offset as ash::vk::DeviceSize,
                    dst_offset: dst_offset as ash::vk::DeviceSize,
                    size: len as ash::vk::DeviceSize,
                }],
            );
            device.end_command_buffer(command_buffer).unwrap();
        }
    }
    unsafe fn kernel(
        &mut self,
        kernel: &RawKernel,
        groups: u32,
        buffers: &[BufferBinding],
        push_constants: &[u8],
    ) {
        let command_buffer = self.command_buffer.unwrap();
        let device = &self.device.device;
        let buffer_infos: Vec<_> = buffers
            .iter()
            .map(|buffer| ash::vk::DescriptorBufferInfo {
                buffer: buffer.slice.buffer.raw.as_ref().unwrap().buffer,
                offset: buffer.slice.range.start as u64,
                range: buffer.slice.range.len() as u64,
            })
            .collect();
        let mut descriptor_writes: Vec<_> = (0..buffers.len())
            .zip(buffer_infos.chunks(1))
            .map(|(i, buffer_info)| {
                ash::vk::WriteDescriptorSet::default()
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .dst_binding(i as u32)
                    .buffer_info(buffer_info)
            })
            .collect();
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                ash::vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
        }
        if let Some(descriptor_pool) = self.descriptor_pool {
            let set_layouts = [kernel.descriptor_set_layout];
            let set_allocate_info = ash::vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            let descriptor_sets =
                unsafe { device.allocate_descriptor_sets(&set_allocate_info).unwrap() };
            let descriptor_set = descriptor_sets[0];
            for write in descriptor_writes.iter_mut() {
                write.dst_set = descriptor_set;
            }
            unsafe {
                device.update_descriptor_sets(&descriptor_writes, &[]);
            }
            unsafe {
                let first_set = 0;
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    ash::vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline_layout,
                    first_set,
                    &descriptor_sets,
                    &[],
                );
            }
        } else {
            todo!();
        }
        if !push_constants.is_empty() {
            unsafe {
                let offset = 0;
                device.cmd_push_constants(
                    command_buffer,
                    kernel.pipeline_layout,
                    ash::vk::ShaderStageFlags::COMPUTE,
                    offset,
                    push_constants,
                );
            }
        }
        unsafe {
            device.cmd_dispatch(command_buffer, groups, 1, 1);
            device.end_command_buffer(command_buffer).unwrap();
        }
    }
}

struct SubmitCommandBuffer {
    queue: Arc<Queue>,
    command_buffer: RawCommandBuffer,
    buffers: Vec<Arc<RawBuffer>>,
    kernel: Option<Arc<RawKernel>>,
}

impl SubmitCommandBuffer {
    fn submit(self) -> Result<QueueEvent> {
        let queue = self.queue;
        let semaphore_value = unsafe {
            queue
                .raw
                .submit(self.command_buffer.command_buffer.unwrap())?
        };
        let command_buffer = CommandBuffer {
            queue: queue.raw.clone(),
            raw: self.command_buffer,
            buffers: self.buffers,
            kernel: self.kernel,
            semaphore_value,
        };
        queue.command_buffers.lock().push_back(command_buffer);
        Ok(QueueEvent {
            queue,
            semaphore_value,
        })
    }
}

struct CommandBuffer {
    queue: Arc<RawQueue>,
    raw: RawCommandBuffer,
    buffers: Vec<Arc<RawBuffer>>,
    kernel: Option<Arc<RawKernel>>,
    semaphore_value: u64,
}

impl CommandBuffer {
    fn wait(&self) -> Result<()> {
        self.queue.wait(self.semaphore_value)
    }
    fn try_wait(&self) -> Result<Result<(), WaitTimeout>> {
        self.queue.try_wait(self.semaphore_value)
    }
    unsafe fn reset(&mut self) -> Result<()> {
        unsafe {
            self.raw.reset()?;
        }
        self.buffers.clear();
        self.kernel.take();
        Ok(())
    }
}

struct BufferAllocator {
    allocator: Mutex<vk_mem::Allocator>,
    device: Arc<RawDevice>,
}

impl BufferAllocator {
    unsafe fn new(
        device: Arc<RawDevice>,
        physical_device: ash::vk::PhysicalDevice,
    ) -> Result<Arc<Self>> {
        let mut allocator_create_info = vk_mem::AllocatorCreateInfo::new(
            &device.backend.instance,
            &device.device,
            physical_device,
        );
        allocator_create_info.flags = vk_mem::AllocatorCreateFlags::EXTERNALLY_SYNCHRONIZED;
        let allocator = unsafe { vk_mem::Allocator::new(allocator_create_info).unwrap() };
        let allocator = Mutex::new(allocator);
        Ok(Arc::new(Self { allocator, device }))
    }
}

struct RawBuffer {
    allocator: Arc<BufferAllocator>,
    buffer: ash::vk::Buffer,
    allocation: vk_mem::Allocation,
    mapped_slice: Option<*mut [u8]>,
}

impl RawBuffer {
    unsafe fn uninit_host(
        allocator: Arc<BufferAllocator>,
        len: usize,
        usage: ash::vk::BufferUsageFlags,
        queue_family_index: u32,
    ) -> Result<Arc<Self>> {
        let size = len as ash::vk::DeviceSize;
        let queue_family_indices = [queue_family_index];
        let buffer_create_info = ash::vk::BufferCreateInfo::default()
            .size(size)
            .queue_family_indices(&queue_family_indices)
            .usage(usage);
        let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
        allocation_create_info.flags = vk_mem::AllocationCreateFlags::MAPPED;
        allocation_create_info.required_flags = ash::vk::MemoryPropertyFlags::HOST_VISIBLE;
        let raw_allocator = allocator.allocator.lock();
        let result =
            unsafe { raw_allocator.create_buffer(&buffer_create_info, &allocation_create_info) };
        let (buffer, allocation) = result.unwrap();
        let info = raw_allocator.get_allocation_info(&allocation);
        std::mem::drop(raw_allocator);
        let mapped_slice = Some(std::ptr::slice_from_raw_parts_mut(
            info.mapped_data.cast(),
            len,
        ));
        Ok(Arc::new(Self {
            allocator,
            buffer,
            allocation,
            mapped_slice,
        }))
    }
    unsafe fn uninit_device(allocator: Arc<BufferAllocator>, len: usize) -> Result<Arc<Self>> {
        let size = len as ash::vk::DeviceSize;
        let queue_family_indices = &allocator.device.queue_family_indices;
        let mut buffer_create_info = ash::vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                ash::vk::BufferUsageFlags::TRANSFER_SRC
                    | ash::vk::BufferUsageFlags::TRANSFER_DST
                    | ash::vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .queue_family_indices(&allocator.device.queue_family_indices);
        if queue_family_indices.len() > 1 {
            buffer_create_info = buffer_create_info.sharing_mode(ash::vk::SharingMode::CONCURRENT);
        }
        let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
        allocation_create_info.flags = vk_mem::AllocationCreateFlags::MAPPED;
        allocation_create_info.required_flags = ash::vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let raw_allocator = allocator.allocator.lock();
        let result =
            unsafe { raw_allocator.create_buffer(&buffer_create_info, &allocation_create_info) };
        let (buffer, allocation) = result.unwrap();
        let info = raw_allocator.get_allocation_info(&allocation);
        let mapped_slice = if info.mapped_data.is_null() {
            None
        } else {
            let mapped_slice = std::ptr::slice_from_raw_parts_mut(info.mapped_data.cast(), len);
            Some(mapped_slice)
        };
        std::mem::drop(raw_allocator);
        Ok(Arc::new(Self {
            allocator,
            buffer,
            allocation,
            mapped_slice,
        }))
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .allocator
                .lock()
                .free_memory(&mut self.allocation);
            self.allocator
                .device
                .device
                .destroy_buffer(self.buffer, None);
        }
    }
}

unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

struct HostBuffer {
    raw: Arc<RawBuffer>,
    len: usize,
    event: Option<QueueEvent>,
}

impl HostBuffer {
    unsafe fn uninit(
        allocator: Arc<BufferAllocator>,
        len: usize,
        usage: ash::vk::BufferUsageFlags,
        queue_family_index: u32,
    ) -> Result<Self> {
        let raw = unsafe { RawBuffer::uninit_host(allocator, len, usage, queue_family_index)? };
        Ok(Self {
            raw,
            len,
            event: None,
        })
    }
}

const HOST_BUFFER_LEN: usize = 32_000_000;

struct HostBufferPair {
    first: HostBuffer,
    second: Option<HostBuffer>,
}

impl HostBufferPair {
    unsafe fn uninit(
        allocator: Arc<BufferAllocator>,
        len: usize,
        usage: ash::vk::BufferUsageFlags,
        queue_family_index: u32,
    ) -> Result<Self> {
        let first_len = len.min(HOST_BUFFER_LEN);
        let first =
            unsafe { HostBuffer::uninit(allocator.clone(), first_len, usage, queue_family_index)? };
        let second_len = len
            .checked_sub(HOST_BUFFER_LEN)
            .map(|x| x.min(HOST_BUFFER_LEN));
        let second = second_len
            .map(|x| unsafe { HostBuffer::uninit(allocator, x, usage, queue_family_index) })
            .transpose()?;
        Ok(Self { first, second })
    }
    fn swap(&mut self) {
        if let Some(second) = self.second.as_mut() {
            std::mem::swap(&mut self.first, second);
        }
    }
}

pub struct Buffer {
    device: Arc<Device>,
    raw: Option<Arc<RawBuffer>>,
    len: usize,
    events: RwLock<Vec<QueueEvent>>,
}

impl super::DeviceOwned for Buffer {
    type Device = Device;
    fn device(&self) -> &Arc<Self::Device> {
        &self.device
    }
}

impl super::Buffer for Buffer {
    type Slice = Slice;
    unsafe fn uninit(device: Arc<Self::Device>, len: usize) -> Result<Arc<Self>> {
        if len == 0 {
            return Ok(Arc::new(Self {
                device,
                raw: None,
                len,
                events: RwLock::default(),
            }));
        }
        device.poll()?;
        let raw = Some(unsafe { RawBuffer::uninit_device(device.buffer_allocator.clone(), len)? });
        Ok(Arc::new(Self {
            device,
            raw,
            len,
            events: RwLock::default(),
        }))
    }
    fn len(&self) -> usize {
        self.len
    }
    fn into_slice(self: Arc<Self>) -> Arc<Self::Slice> {
        let range = BufferRange {
            start: 0,
            end: self.len,
        };
        Arc::new(Slice {
            buffer: self,
            range,
        })
    }
}

impl Buffer {
    fn poll_events(&self) -> Result<()> {
        if let Some(mut events) = self.events.try_write() {
            let mut result = Ok(());
            events.retain(|event| {
                if result.is_err() {
                    return true;
                }
                match event.try_wait() {
                    Ok(Ok(())) => false,
                    Ok(Err(_)) => true,
                    Err(err) => {
                        result = Err(err);
                        true
                    }
                }
            });
            return result;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Slice {
    buffer: Arc<Buffer>,
    range: BufferRange,
}

impl super::DeviceOwned for Slice {
    type Device = Device;
    fn device(&self) -> &Arc<Self::Device> {
        self.buffer.device()
    }
}

impl super::Slice for Slice {
    fn len(&self) -> usize {
        self.range.len()
    }
    fn range(&self) -> BufferRange {
        self.range
    }
    fn slice(self: Arc<Self>, range: BufferRange) -> Arc<Self> {
        todo!()
    }
    unsafe fn upload(&self, bytes: &[u8]) -> Result<()> {
        debug_assert_eq!(self.len(), bytes.len());
        if bytes.is_empty() {
            return Ok(());
        }
        let device = &self.buffer.device;
        let mut buffer_events = self.buffer.events.write();
        if device.transfer_queue.is_none() {
            if let Some(mapped_slice) = self.buffer.raw.as_ref().unwrap().mapped_slice {
                for event in std::mem::take(&mut *buffer_events) {
                    event.wait()?;
                }
                unsafe {
                    (&mut *mapped_slice).copy_from_slice(bytes);
                }
                device.poll()?;
                return Ok(());
            }
        }
        device.poll()?;
        let queue = device.transfer_queue();
        let mut host_buffers = unsafe {
            HostBufferPair::uninit(
                device.buffer_allocator.clone(),
                bytes.len(),
                ash::vk::BufferUsageFlags::TRANSFER_SRC,
                queue.raw.family_index,
            )?
        };
        let mut offset = 0;
        for chunk in bytes.chunks(HOST_BUFFER_LEN) {
            let host_buffer = &mut host_buffers.first;
            let command_buffer = queue.transfer(
                host_buffer.raw.clone(),
                0,
                self.buffer.raw.clone().unwrap(),
                self.range.start + offset,
                chunk.len(),
            )?;
            for event in std::mem::take(&mut *buffer_events) {
                event.wait()?;
            }
            if let Some(event) = host_buffer.event.take() {
                event.wait()?;
            }
            unsafe {
                (&mut *host_buffer.raw.mapped_slice.unwrap())[..chunk.len()].copy_from_slice(chunk);
            }
            let event = command_buffer.submit()?;
            offset += chunk.len();
            if offset < bytes.len() {
                host_buffer.event.replace(event);
                host_buffers.swap();
            } else {
                buffer_events.push(event);
            }
        }
        queue.poll()?;
        Ok(())
    }
    unsafe fn download(&self, bytes: &mut [u8]) -> Result<()> {
        debug_assert_eq!(self.len(), bytes.len());
        if bytes.is_empty() {
            return Ok(());
        }
        let device = &self.buffer.device;
        let mut buffer_events = self.buffer.events.read().clone();
        if device.transfer_queue.is_none() {
            if let Some(mapped_slice) = self.buffer.raw.as_ref().unwrap().mapped_slice {
                for event in buffer_events {
                    event.wait()?;
                }
                unsafe {
                    bytes.copy_from_slice(&*mapped_slice);
                }
                self.buffer.poll_events()?;
                device.poll()?;
                return Ok(());
            }
        }
        device.poll()?;
        let queue = device.transfer_queue();
        let mut host_buffers = unsafe {
            HostBufferPair::uninit(
                device.buffer_allocator.clone(),
                bytes.len(),
                ash::vk::BufferUsageFlags::TRANSFER_DST,
                queue.raw.family_index,
            )?
        };
        let mut offset = 0;
        let host_buffer = &mut host_buffers.first;
        let command_buffer = queue.transfer(
            self.buffer.raw.clone().unwrap(),
            self.range.start + offset,
            host_buffer.raw.clone(),
            0,
            host_buffer.len,
        )?;
        host_buffer.event.replace(command_buffer.submit()?);
        let mut chunk_iter = bytes.chunks_mut(HOST_BUFFER_LEN).peekable();
        while let Some(chunk) = chunk_iter.next() {
            offset += chunk.len();
            let command_buffer = if let Some((chunk, host_buffer)) =
                chunk_iter.peek().zip(host_buffers.second.as_mut())
            {
                let command_buffer = queue.transfer(
                    self.buffer.raw.clone().unwrap(),
                    self.range.start + offset,
                    host_buffer.raw.clone(),
                    0,
                    chunk.len(),
                )?;
                Some(command_buffer)
            } else {
                None
            };
            for event in std::mem::take(&mut buffer_events) {
                event.wait()?;
            }
            let host_buffer = &mut host_buffers.first;
            if let Some(event) = host_buffer.event.take() {
                event.wait()?;
            }
            let chunk_len = chunk.len();
            unsafe {
                chunk.copy_from_slice(&(&*host_buffer.raw.mapped_slice.unwrap())[..chunk_len]);
            }
            if let Some(command_buffer) = command_buffer {
                let host_buffer = host_buffers.second.as_mut().unwrap();
                host_buffer.event.replace(command_buffer.submit()?);
                host_buffers.swap();
            }
        }
        self.buffer.poll_events()?;
        queue.poll()?;
        Ok(())
    }
}

struct RawKernel {
    device: Arc<RawDevice>,
    pipeline: ash::vk::Pipeline,
    pipeline_layout: ash::vk::PipelineLayout,
    descriptor_set_layout: ash::vk::DescriptorSetLayout,
    desc: KernelDesc,
}

impl RawKernel {
    fn new(device: Arc<RawDevice>, info: KernelCreateInfo) -> Result<Arc<Self>, CompileError> {
        let KernelCreateInfo {
            spirv,
            spec_constants,
            desc,
        } = info;
        if !device.features.contains(desc.features) {
            todo!(
                "Device features ({:?}) does not contain ({:?})!",
                device.features,
                desc.features
            );
        }
        let module_create_info = ash::vk::ShaderModuleCreateInfo::default().code(spirv);
        let vk_device = &device.device;
        let module = unsafe {
            vk_device
                .create_shader_module(&module_create_info, None)
                .unwrap()
        };
        let descriptor_set_layout_bindings: Vec<_> = (0..desc.buffers)
            .map(|binding| {
                ash::vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let descriptor_set_layout_create_info = ash::vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&descriptor_set_layout_bindings);
        let descriptor_set_layout = unsafe {
            vk_device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .unwrap()
        };
        let set_layouts = [descriptor_set_layout];
        let push_constant_range = ash::vk::PushConstantRange::default()
            .size(desc.push_constant_bytes)
            .stage_flags(ash::vk::ShaderStageFlags::COMPUTE);
        let push_constant_ranges = [push_constant_range];
        let mut pipeline_layout_create_info =
            ash::vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if desc.push_constant_bytes > 0 {
            pipeline_layout_create_info =
                pipeline_layout_create_info.push_constant_ranges(&push_constant_ranges);
        }
        let pipeline_layout = unsafe {
            vk_device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };
        let name = CString::new("main").unwrap();
        let spec_data: Vec<u32> = spec_constants.values().copied().flatten().collect();
        let mut spec_offset = 0u32;
        let spec_entries: Vec<_> = spec_constants
            .iter()
            .map(|(id, value)| {
                let size = size_of_val(value.as_slice());
                let entry = ash::vk::SpecializationMapEntry {
                    constant_id: *id,
                    offset: spec_offset,
                    size,
                };
                spec_offset += size as u32;
                entry
            })
            .collect();
        let specialization_info = ash::vk::SpecializationInfo::default()
            .data(bytemuck::cast_slice(&spec_data))
            .map_entries(&spec_entries);
        let mut required_subgroup_size = desc.subgroup_threads.map(|subgroup_threads| {
            ash::vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(subgroup_threads)
        });
        let mut pipeline_shader_stage_create_info =
            ash::vk::PipelineShaderStageCreateInfo::default()
                .module(module)
                .name(&name)
                .stage(ash::vk::ShaderStageFlags::COMPUTE)
                .specialization_info(&specialization_info);
        if let Some(required_subgroup_size) = required_subgroup_size.as_mut() {
            pipeline_shader_stage_create_info =
                pipeline_shader_stage_create_info.push_next(required_subgroup_size);
        }
        let pipeline_create_info = ash::vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout)
            .stage(pipeline_shader_stage_create_info);
        let pipelines = unsafe {
            vk_device
                .create_compute_pipelines(
                    ash::vk::PipelineCache::null(),
                    &[pipeline_create_info],
                    None,
                )
                .unwrap()
        };
        let pipeline = pipelines[0];
        unsafe {
            vk_device.destroy_shader_module(module, None);
        }
        Ok(Arc::new(Self {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            desc,
        }))
    }
}

impl Drop for RawKernel {
    fn drop(&mut self) {
        let device = &self.device.device;
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub struct Kernel {
    device: Arc<Device>,
    raw: Arc<RawKernel>,
}

impl super::DeviceOwned for Kernel {
    type Device = Device;
    fn device(&self) -> &Arc<Self::Device> {
        &self.device
    }
}

type BufferBinding = super::BufferBinding<Slice>;

struct CompileError {}

impl super::Kernel for Kernel {
    type Slice = Slice;
    fn get_or_create(
        device: Arc<Self::Device>,
        key: KernelKey,
        f: impl FnOnce() -> KernelCreateInfo,
    ) -> Result<Arc<Self>> {
        let mut kernels = device.kernels.lock();
        let raw_device = device.raw.clone();
        let result = kernels
            .entry(key)
            .or_insert_with(|| RawKernel::new(raw_device, f()));
        match result {
            Ok(raw) => Ok(Arc::new(Self {
                device: device.clone(),
                raw: raw.clone(),
            })),
            Err(err) => todo!(),
        }
    }
    fn desc(&self) -> &KernelDesc {
        &self.raw.desc
    }
    unsafe fn exec(
        self: &Arc<Self>,
        groups: u32,
        buffers: &[BufferBinding],
        push_constants: &[u8],
    ) -> Result<()> {
        #[cfg(debug_assertions)]
        {
            for (i, buffer) in buffers.iter().enumerate() {
                let mutable = self.raw.desc.mutability.get(i.try_into().unwrap());
                assert_eq!(buffer.mutable, mutable);
            }
        }
        self.device.poll()?;
        let (queue_index, encode_lock) =
            if let Some(queue_selector) = self.device.queue_selector.as_ref() {
                let (index, encode) = queue_selector.lock().select(&self.device.compute_queues)?;
                (index, Some(encode))
            } else {
                (0, None)
            };
        let queue = &self.device.compute_queues[queue_index];
        let command_buffer = queue.kernel(self.raw.clone(), groups, &buffers, push_constants)?;
        for buffer in buffers.iter() {
            let events = buffer.slice.buffer.events.read().clone();
            for event in events {
                event.wait()?;
            }
        }
        let event = command_buffer.submit()?;
        std::mem::drop(encode_lock);
        for buffer in buffers.iter() {
            if buffer.mutable {
                buffer.slice.buffer.events.write().push(event.clone());
            }
        }
        for buffer in buffers.iter() {
            buffer.slice.buffer.poll_events()?;
        }
        Ok(())
    }
}
