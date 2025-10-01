use super::{BufferRange, DeviceOwned, DeviceSpecifier};
use crate::{context::device, Result};
use parking_lot::RwLock;
use std::{future::Future, pin::Pin, sync::Arc};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
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

pub struct Device {
    raw: web_sys::GpuDevice,
    adapter: web_sys::GpuAdapter,
    backend: Arc<Backend>,
}

impl Drop for Device {
    fn drop(&mut self) {
        self.raw.destroy();
    }
}

#[cfg(not(target_feature = "atomics"))]
unsafe impl Send for Device {}
#[cfg(not(target_feature = "atomics"))]
unsafe impl Sync for Device {}

impl super::Device for Device {
    type Backend = Backend;
    type Event = Event;
    async fn create_async(backend: Arc<Self::Backend>) -> Result<Arc<Self>> {
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
        Ok(Arc::new(Self {
            raw: device,
            adapter,
            backend,
        }))
    }
    fn event(self: &Arc<Self>) -> Arc<Event> {
        Arc::new(Event {
            device: self.clone(),
            promise: self.raw.queue().on_submitted_work_done(),
        })
    }
}

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
    device: Arc<Device>,
    buffer: web_sys::GpuBuffer,
    len: usize,
}

#[cfg(not(target_feature = "atomics"))]
unsafe impl Send for RawBuffer {}
#[cfg(not(target_feature = "atomics"))]
unsafe impl Sync for RawBuffer {}

impl RawBuffer {
    unsafe fn uninit_device(device: Arc<Device>, len: usize) -> Result<Arc<Self>> {
        let mut buffer_desc = web_sys::GpuBufferDescriptor::new(
            len as f64,
            BufferUsage::CopyDst as u32 | BufferUsage::CopySrc as u32 | BufferUsage::Storage as u32,
        );
        let buffer = device.raw.create_buffer(&buffer_desc).unwrap();
        Ok(Arc::new(Self {
            device,
            buffer,
            len,
        }))
    }
    unsafe fn uninit_download(device: Arc<Device>, len: usize) -> Result<Arc<Self>> {
        let mut buffer_desc = web_sys::GpuBufferDescriptor::new(
            len as f64,
            BufferUsage::CopyDst as u32 | BufferUsage::MapRead as u32,
        );
        let buffer = device.raw.create_buffer(&buffer_desc).unwrap();
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
            Some(RawBuffer::uninit_device(device.clone(), len)?)
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
    unsafe fn uninit_download(device: Arc<Device>, len: usize) -> Result<Self> {
        let raw = unsafe { RawBuffer::uninit_download(device, len)? };
        Ok(Self { raw, promise: None })
    }
}

struct HostBufferPair {
    first: HostBuffer,
    second: Option<HostBuffer>,
}

impl HostBufferPair {
    unsafe fn uninit_download(device: Arc<Device>, len: usize) -> Result<Self> {
        let first_len = len.min(HOST_BUFFER_LEN);
        let first = unsafe { HostBuffer::uninit_download(device.clone(), first_len)? };
        let second_len = len
            .checked_sub(HOST_BUFFER_LEN)
            .map(|x| x.min(HOST_BUFFER_LEN));
        let second = second_len
            .map(|x| unsafe { HostBuffer::uninit_download(device, x) })
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
        let queue = self.device().raw.queue();
        let mut offset = 0;
        for chunk in bytes.chunks(HOST_BUFFER_LEN) {
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
        let queue = device.raw.queue();
        let buffer = self.buffer.raw.as_ref().unwrap();
        let mut buffer_promise = self.buffer.promise.write();
        let mut host_buffers = unsafe { HostBufferPair::uninit_download(device.clone(), len)? };
        let mut offset = 0;
        let host_buffer = &mut host_buffers.first;
        let encoder = device.raw.create_command_encoder();
        encoder
            .copy_buffer_to_buffer_with_u32_and_u32_and_u32(
                &buffer.buffer,
                (self.range.start + offset) as u32,
                &host_buffer.raw.buffer,
                0,
                len as u32,
            )
            .unwrap();
        let command_buffer = encoder.finish();
        queue.submit(&JsValue::from(Array::of1(&command_buffer)));
        host_buffer.promise.replace(queue.on_submitted_work_done());
        let mut chunk_iter = bytes.chunks_mut(HOST_BUFFER_LEN).peekable();
        while let Some(chunk) = chunk_iter.next() {
            let command_buffer = if let Some((chunk, host_buffer)) =
                chunk_iter.peek().zip(host_buffers.second.as_mut())
            {
                let encoder = device.raw.create_command_encoder();
                encoder
                    .copy_buffer_to_buffer_with_u32_and_u32_and_u32(
                        &buffer.buffer,
                        (self.range.start + offset) as u32,
                        &host_buffer.raw.buffer,
                        0,
                        len as u32,
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
            let output = host_buffer.raw.buffer.get_mapped_range().unwrap();
            let output = Int8Array::new(&output);
            output.copy_to(bytemuck::cast_slice_mut(chunk));
            if let Some(command_buffer) = command_buffer {
                queue.submit(&JsValue::from(Array::of1(&command_buffer)));
                host_buffer.promise.replace(queue.on_submitted_work_done());
                offset += chunk.len();
                host_buffers.swap();
            }
        }
        Ok(())
    }
}

pub enum Kernel {}

impl super::DeviceOwned for Kernel {
    type Device = Device;
    fn device(&self) -> &Arc<Self::Device> {
        todo!()
    }
}

impl super::Kernel for Kernel {
    type Slice = Slice;
    fn create(_device: Arc<Self::Device>, _spirv: &[u32]) -> Result<Arc<Self>> {
        todo!()
    }
    unsafe fn exec(
        self: Arc<Self>,
        _groups: u32,
        _slices: &[Self::Slice],
        _push_constants: &[u32],
    ) -> Result<()> {
        todo!()
    }
}
