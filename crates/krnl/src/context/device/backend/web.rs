use super::{BufferRange, DeviceOwned, DeviceSpecifier, Properties};
use crate::{
    Result,
    context::device,
    kernel::{KernelCreateInfo, KernelDesc, KernelKey},
};
use fxhash::FxHashMap;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    pin::Pin,
    sync::Arc,
};
use tinyvec::ArrayVec;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    GpuBindGroupDescriptor, GpuBindGroupEntry, GpuBindGroupLayoutDescriptor,
    GpuBindGroupLayoutEntry, GpuBufferBinding, GpuBufferBindingLayout, GpuBufferBindingType,
    GpuComputePipeline, GpuComputePipelineDescriptor, GpuPipelineLayoutDescriptor,
    GpuProgrammableStage, GpuShaderModule, GpuShaderModuleDescriptor,
    js_sys::{Array, Int8Array, Promise},
    wasm_bindgen::{JsCast as _, JsValue},
};

pub struct Backend {
    gpu: web_sys::Gpu,
}

#[cfg(not(target_feature = "atomics"))]
unsafe impl Send for Backend {}
#[cfg(not(target_feature = "atomics"))]
unsafe impl Sync for Backend {}

impl super::Backend for Backend {
    type Device = Device;
    type Event = Event;
    type Buffer = Buffer;
    type Slice = Slice;
    type Kernel = Kernel;
    fn create() -> Result<Arc<Self>> {
        let window = web_sys::window().unwrap();
        let navigator = window.navigator();
        let gpu = navigator.gpu();
        Ok(Arc::new(Self { gpu }))
    }
}

struct RawDevice {
    device: web_sys::GpuDevice,
    adapter: web_sys::GpuAdapter,
    backend: Arc<Backend>,
    properties: Properties,
}

impl RawDevice {
    async fn new(backend: Arc<Backend>) -> Result<Arc<Self>> {
        let adapter: web_sys::GpuAdapter = JsFuture::from(backend.gpu.request_adapter())
            .await
            .unwrap()
            .dyn_into()
            .unwrap();
        let device: web_sys::GpuDevice = JsFuture::from(adapter.request_device())
            .await
            .unwrap()
            .dyn_into()
            .unwrap();
        let limits = device.limits();
        let max_buffer_size = limits.max_buffer_size() as u32;
        let properties = Properties {
            max_buffer_size,
            max_subgroup_threads: 128,
            min_subgroup_threads: 1,
        };
        Ok(Arc::new(Self {
            device,
            adapter,
            backend,
            properties,
        }))
    }
}

impl Drop for RawDevice {
    fn drop(&mut self) {
        self.device.destroy();
    }
}

#[cfg(not(target_feature = "atomics"))]
unsafe impl Send for RawDevice {}
#[cfg(not(target_feature = "atomics"))]
unsafe impl Sync for RawDevice {}

pub struct Device {
    raw: Arc<RawDevice>,
    push_constant_buffer: Arc<RawBuffer>,
    kernels: Mutex<FxHashMap<KernelKey, Result<Arc<RawKernel>, CompileError>>>,
}

impl super::Device for Device {
    type Backend = Backend;
    type Event = Event;
    async fn create_async(backend: Arc<Self::Backend>) -> Result<Arc<Self>> {
        let raw = RawDevice::new(backend).await?;
        let push_constant_buffer = unsafe { RawBuffer::uninit_push(raw.clone(), 128)? };
        let kernels = Mutex::default();
        Ok(Arc::new(Self {
            raw,
            push_constant_buffer,
            kernels,
        }))
    }
    fn event(self: &Arc<Self>) -> Arc<Event> {
        Arc::new(Event {
            device: self.clone(),
            promise: self.raw.device.queue().on_submitted_work_done(),
        })
    }
    fn properties(&self) -> &Properties {
        &self.raw.properties
    }
}

#[cfg(not(target_feature = "atomics"))]
unsafe impl Send for Device {}
#[cfg(not(target_feature = "atomics"))]
unsafe impl Sync for Device {}

pub struct Event {
    device: Arc<Device>,
    promise: Promise,
}

impl super::DeviceOwned for Event {
    type Device = Device;
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl super::Event for Event {
    async fn wait_async(&self) -> Result<()> {
        JsFuture::from(self.promise.clone()).await.unwrap();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
enum BufferUsage {
    CopySrc = 4,
    CopyDst = 8,
    MapRead = 1,
    MapWrite = 2,
    Storage = 128,
    Uniform = 64,
}

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
enum MapMode {
    Read = 1,
    Write = 2,
}

struct RawBuffer {
    device: Arc<RawDevice>,
    buffer: web_sys::GpuBuffer,
    len: usize,
}

#[cfg(not(target_feature = "atomics"))]
unsafe impl Send for RawBuffer {}
#[cfg(not(target_feature = "atomics"))]
unsafe impl Sync for RawBuffer {}

impl RawBuffer {
    unsafe fn uninit_device(device: Arc<RawDevice>, len: usize) -> Result<Arc<Self>> {
        debug_assert!(len <= device.properties.max_buffer_size as usize);
        let mut buffer_desc = web_sys::GpuBufferDescriptor::new(
            len as f64,
            BufferUsage::CopyDst as u32 | BufferUsage::CopySrc as u32 | BufferUsage::Storage as u32,
        );
        let buffer = device.device.create_buffer(&buffer_desc).unwrap();
        Ok(Arc::new(Self {
            device,
            buffer,
            len,
        }))
    }
    unsafe fn uninit_push(device: Arc<RawDevice>, len: usize) -> Result<Arc<Self>> {
        debug_assert!(len <= device.properties.max_buffer_size as usize);
        let mut usage = BufferUsage::CopyDst as u32;
        if PUSH_UNIFORM {
            usage |= BufferUsage::Uniform as u32;
        } else {
            usage |= BufferUsage::Storage as u32;
        }
        let mut buffer_desc = web_sys::GpuBufferDescriptor::new(len as f64, usage);
        let buffer = device.device.create_buffer(&buffer_desc).unwrap();
        Ok(Arc::new(Self {
            device,
            buffer,
            len,
        }))
    }
    unsafe fn uninit_download(device: Arc<RawDevice>, len: usize) -> Result<Arc<Self>> {
        debug_assert!(len <= device.properties.max_buffer_size as usize);
        let mut buffer_desc = web_sys::GpuBufferDescriptor::new(
            len as f64,
            BufferUsage::CopyDst as u32 | BufferUsage::MapRead as u32,
        );
        let buffer = device.device.create_buffer(&buffer_desc).unwrap();
        Ok(Arc::new(Self {
            device,
            buffer,
            len,
        }))
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        self.buffer.destroy();
    }
}

pub struct Buffer {
    device: Arc<Device>,
    raw: Option<Arc<RawBuffer>>,
    len: usize,
    promise: RwLock<Option<Promise>>,
}

impl DeviceOwned for Buffer {
    type Device = Device;
    fn device(&self) -> &Arc<Self::Device> {
        &self.device
    }
}

impl super::Buffer for Buffer {
    type Slice = Slice;
    unsafe fn uninit(device: Arc<Self::Device>, len: usize) -> Result<Arc<Self>> {
        let raw = if len > 0 {
            Some(unsafe { RawBuffer::uninit_device(device.raw.clone(), len)? })
        } else {
            None
        };
        Ok(Arc::new(Self {
            device,
            raw,
            len,
            promise: RwLock::default(),
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

const HOST_BUFFER_LEN: usize = 32_000_000;

struct HostBuffer {
    raw: Arc<RawBuffer>,
    promise: Option<Promise>,
}

impl HostBuffer {
    unsafe fn uninit_download(device: Arc<RawDevice>, len: usize) -> Result<Self> {
        debug_assert!(len <= device.properties.max_buffer_size as usize);
        let raw = unsafe { RawBuffer::uninit_download(device.clone(), len)? };
        Ok(Self { raw, promise: None })
    }
}

struct HostBufferPair {
    first: HostBuffer,
    second: Option<HostBuffer>,
}

impl HostBufferPair {
    unsafe fn uninit_download(device: Arc<RawDevice>, len: usize) -> Result<Self> {
        let max_len = HOST_BUFFER_LEN.min(device.properties.max_buffer_size as usize);
        let first_len = len.min(HOST_BUFFER_LEN);
        let first = unsafe { HostBuffer::uninit_download(device.clone(), first_len)? };
        let second_len = len.checked_sub(max_len).map(|x| x.min(max_len));
        let second = second_len
            .map(|x| unsafe { HostBuffer::uninit_download(device.clone(), x) })
            .transpose()?;
        Ok(Self { first, second })
    }
    fn swap(&mut self) {
        if let Some(second) = self.second.as_mut() {
            std::mem::swap(&mut self.first, second);
        }
    }
}

pub struct Slice {
    buffer: Arc<Buffer>,
    range: BufferRange,
}

impl DeviceOwned for Slice {
    type Device = Device;
    fn device(&self) -> &Arc<Self::Device> {
        self.buffer.device()
    }
}

impl super::Slice for Slice {
    fn len(&self) -> usize {
        self.buffer.len
    }
    fn range(&self) -> BufferRange {
        self.range
    }
    fn slice(self: Arc<Self>, _range: BufferRange) -> Arc<Self> {
        todo!()
    }
    unsafe fn upload(&self, bytes: &[u8]) -> Result<()> {
        assert_eq!(self.len(), bytes.len());
        let raw = if let Some(raw) = self.buffer.raw.as_ref() {
            raw
        } else {
            return Ok(());
        };
        let device = self.device();
        let queue = device.raw.device.queue();
        let mut offset = 0;
        let max_len = HOST_BUFFER_LEN.min(device.raw.properties.max_buffer_size as usize);
        for chunk in bytes.chunks(max_len) {
            queue
                .write_buffer_with_u32_and_u8_slice_and_u32_and_u32(
                    &raw.buffer,
                    (self.range.start + offset) as u32,
                    chunk,
                    0,
                    chunk.len() as u32,
                )
                .unwrap();
            offset += chunk.len();
        }
        Ok(())
    }
    async unsafe fn download_async(&self, bytes: &mut [u8]) -> Result<()> {
        assert_eq!(self.len(), bytes.len());
        if bytes.is_empty() {
            return Ok(());
        }
        let len = self.len();
        let device = self.device();
        let queue = device.raw.device.queue();
        let buffer = self.buffer.raw.as_ref().unwrap();
        let mut buffer_promise = self.buffer.promise.write();
        let mut host_buffers = unsafe { HostBufferPair::uninit_download(device.raw.clone(), len)? };
        let mut offset = 0;
        let host_buffer = &mut host_buffers.first;
        let encoder = device.raw.device.create_command_encoder();
        encoder
            .copy_buffer_to_buffer_with_u32_and_u32_and_u32(
                &buffer.buffer,
                (self.range.start + offset) as u32,
                &host_buffer.raw.buffer,
                0,
                host_buffer.raw.len as u32,
            )
            .unwrap();
        let command_buffer = encoder.finish();
        queue.submit(&JsValue::from(Array::of1(&command_buffer)));
        host_buffer.promise.replace(queue.on_submitted_work_done());
        let max_len = HOST_BUFFER_LEN.min(device.raw.properties.max_buffer_size as usize);
        let mut chunk_iter = bytes.chunks_mut(max_len).peekable();
        while let Some(chunk) = chunk_iter.next() {
            offset += chunk.len();
            JsFuture::from(queue.on_submitted_work_done())
                .await
                .unwrap();
            let command_buffer = if let Some((chunk, host_buffer)) =
                chunk_iter.peek().zip(host_buffers.second.as_mut())
            {
                let encoder = device.raw.device.create_command_encoder();
                encoder
                    .copy_buffer_to_buffer_with_u32_and_u32_and_u32(
                        &buffer.buffer,
                        (self.range.start + offset) as u32,
                        &host_buffer.raw.buffer,
                        0,
                        chunk.len() as u32,
                    )
                    .unwrap();
                let command_buffer = encoder.finish();
                Some(command_buffer)
            } else {
                None
            };
            if let Some(promise) = buffer_promise.take() {
                JsFuture::from(promise).await.unwrap();
            }
            let host_buffer = &mut host_buffers.first;
            JsFuture::from(host_buffer.promise.take().unwrap())
                .await
                .unwrap();
            JsFuture::from(host_buffer.raw.buffer.map_async(MapMode::Read as u32))
                .await
                .unwrap();
            let output = host_buffer
                .raw
                .buffer
                .get_mapped_range_with_u32_and_u32(0, chunk.len() as u32)
                .unwrap();
            let output = Int8Array::new(&output);
            output.copy_to(bytemuck::cast_slice_mut(chunk));
            host_buffer.raw.buffer.unmap();
            if let Some(command_buffer) = command_buffer {
                queue.submit(&JsValue::from(Array::of1(&command_buffer)));
                let host_buffer = host_buffers.second.as_mut().unwrap();
                host_buffer.promise.replace(queue.on_submitted_work_done());
                host_buffers.swap();
            }
        }
        Ok(())
    }
}

const PUSH_UNIFORM: bool = false;

struct RawKernel {
    device: Arc<RawDevice>,
    desc: KernelDesc,
    pipeline: GpuComputePipeline,
    has_push_constants: bool,
}

impl RawKernel {
    fn new(device: Arc<RawDevice>, info: KernelCreateInfo) -> Result<Arc<Self>, CompileError> {
        let KernelCreateInfo {
            spirv,
            spec_constants,
            desc,
        } = info;
        let mut has_push_constants = false;
        let wgsl = spirv_to_wgsl(spirv, &spec_constants, &mut has_push_constants)?;
        let module = device
            .device
            .create_shader_module(&GpuShaderModuleDescriptor::new(&wgsl));
        let bindgroup_layout_entries =
            Array::new_with_length(desc.buffers + has_push_constants as u32);
        for binding in 0..desc.buffers {
            let buffer_binding_layout = GpuBufferBindingLayout::new();
            let binding_type = if desc.mutability.get(binding) {
                GpuBufferBindingType::Storage
            } else {
                GpuBufferBindingType::ReadOnlyStorage
            };
            buffer_binding_layout.set_type(binding_type);
            let layout_entry =
                GpuBindGroupLayoutEntry::new(binding, web_sys::gpu_shader_stage::COMPUTE);
            layout_entry.set_buffer(&buffer_binding_layout);
            bindgroup_layout_entries.set(binding, layout_entry.into());
        }
        if has_push_constants && desc.push_constant_bytes > 0 {
            let buffer_binding_layout = GpuBufferBindingLayout::new();
            let binding_type = if PUSH_UNIFORM {
                GpuBufferBindingType::Uniform
            } else {
                GpuBufferBindingType::ReadOnlyStorage
            };
            buffer_binding_layout.set_type(binding_type);
            let binding = desc.buffers;
            let layout_entry =
                GpuBindGroupLayoutEntry::new(binding, web_sys::gpu_shader_stage::COMPUTE);
            layout_entry.set_buffer(&buffer_binding_layout);
            bindgroup_layout_entries.set(binding, layout_entry.into());
        }
        let bindgroup_layout = device
            .device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(
                &bindgroup_layout_entries,
            ))
            .unwrap();
        let bindgroup_layouts = Array::new_with_length(1);
        bindgroup_layouts.set(0, bindgroup_layout.into());
        let layout = device
            .device
            .create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&bindgroup_layouts));
        let stage = GpuProgrammableStage::new(&module);
        let pipeline = device
            .device
            .create_compute_pipeline(&GpuComputePipelineDescriptor::new(&layout, &stage));
        Ok(Arc::new(Self {
            device,
            desc,
            pipeline,
            has_push_constants,
        }))
    }
}

fn spirv_to_wgsl(
    spirv: &[u32],
    spec_constants: &BTreeMap<u32, ArrayVec<[u32; 2]>>,
    has_push_constants: &mut bool,
) -> Result<String, CompileError> {
    use naga::{
        AddressSpace, Expression, Module, Override, ResourceBinding, Scalar, ScalarKind,
        ShaderStage, Span, StorageAccess, Type, TypeInner,
        back::wgsl::{Writer as WgslWriter, WriterFlags as WgslWriterFlags},
        front::spv::{Frontend as SpvFrontend, Options as SpvOptions},
        valid::{Capabilities, ModuleInfo, ValidationFlags, Validator},
    };

    let mut module = SpvFrontend::new(
        spirv.iter().copied(),
        &SpvOptions {
            strict_capabilities: true,
            ..SpvOptions::default()
        },
    )
    .parse()
    .unwrap();
    let threads = spec_constants[&0][0];
    let mut pipeline_constants = naga::back::PipelineConstants::new();
    for (_handle, over) in module.overrides.iter() {
        let id = over.id.unwrap();
        let spec = spec_constants[&(id as u32)];
        let ty = &module.types[over.ty];
        let scalar = if let TypeInner::Scalar(scalar) = &ty.inner {
            *scalar
        } else {
            unreachable!()
        };
        let value = match scalar {
            Scalar::U32 => spec[0] as f64,
            _ => todo!(),
        };
        pipeline_constants.insert(id.to_string(), value);
    }
    let entry_point = module.entry_points.first_mut().unwrap();
    entry_point.workgroup_size[0] = threads;

    let buffers = module
        .global_variables
        .iter()
        .map(|x| x.1)
        .filter(|x| matches!(x.space, AddressSpace::Storage { .. }))
        .count() as u32;

    if let Some(var) = module
        .global_variables
        .iter_mut()
        .map(|x| x.1)
        .find(|x| x.space == AddressSpace::PushConstant)
    {
        // TODO: uniform imposes 16 byte alignment / stride for arrays
        if PUSH_UNIFORM {
            var.space = AddressSpace::Uniform;
        } else {
            var.space = AddressSpace::Storage {
                access: StorageAccess::LOAD,
            };
        }
        var.binding.replace(ResourceBinding {
            group: 0,
            binding: buffers,
        });
        *has_push_constants = true;
    }
    let capabilities = Capabilities::FLOAT64
        | Capabilities::SHADER_INT64
        | Capabilities::SUBGROUP
        | Capabilities::SUBGROUP_BARRIER
        | Capabilities::SHADER_FLOAT16
        | Capabilities::SHADER_FLOAT16_IN_FLOAT32;
    let mut validator = Validator::new(ValidationFlags::empty(), capabilities);
    let module_info = validator.validate(&module).unwrap();
    let (module, module_info) = naga::back::pipeline_constants::process_overrides(
        &module,
        &module_info,
        Some((ShaderStage::Compute, "main")),
        &pipeline_constants,
    )
    .unwrap();
    let mut module = module.into_owned();
    module.overrides.clear();
    let mut wgsl_writer = WgslWriter::new(String::new(), WgslWriterFlags::empty());
    wgsl_writer.write(&module, &module_info).unwrap();
    let wgsl = wgsl_writer.finish();
    Ok(wgsl)
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
        let queue = self.device.raw.device.queue();
        if self.raw.has_push_constants && !push_constants.is_empty() {
            queue
                .write_buffer_with_u32_and_u8_slice_and_u32_and_u32(
                    &self.device.push_constant_buffer.buffer,
                    0,
                    push_constants,
                    0,
                    push_constants.len() as u32,
                )
                .unwrap();
        }
        let encoder = self.device.raw.device.create_command_encoder();
        let compute_pass = encoder.begin_compute_pass();
        compute_pass.set_pipeline(&self.raw.pipeline);
        let bindgroup_entries =
            Array::new_with_length(buffers.len() as u32 + self.raw.has_push_constants as u32);
        for (binding, buffer) in buffers.iter().enumerate() {
            let resource = GpuBufferBinding::new(&buffer.slice.buffer.raw.as_ref().unwrap().buffer);
            resource.set_offset(buffer.slice.range.start as f64);
            resource.set_size(buffer.slice.range.len() as f64);
            let binding = binding as u32;
            let bindgroup_entry = GpuBindGroupEntry::new(binding, &resource);
            bindgroup_entries.set(binding, bindgroup_entry.into());
        }
        if self.raw.has_push_constants && !push_constants.is_empty() {
            let resource = GpuBufferBinding::new(&self.device.push_constant_buffer.buffer);
            resource.set_size(push_constants.len() as f64);
            let binding = buffers.len() as u32;
            let bindgroup_entry = GpuBindGroupEntry::new(binding, &resource);
            bindgroup_entries.set(binding, bindgroup_entry.into());
        }
        let bindgroup_layout = self.raw.pipeline.get_bind_group_layout(0);
        let bindgroup = self
            .device
            .raw
            .device
            .create_bind_group(&GpuBindGroupDescriptor::new(
                &bindgroup_entries,
                &bindgroup_layout,
            ));
        compute_pass.set_bind_group(0, Some(&bindgroup));
        compute_pass.dispatch_workgroups(groups);
        compute_pass.end();
        let command_buffer = encoder.finish();
        queue.submit(&Array::of1(&command_buffer));
        Ok(())
    }
}
