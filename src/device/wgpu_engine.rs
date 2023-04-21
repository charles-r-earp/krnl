use super::{
    error::{DeviceIndexOutOfRange, DeviceUnavailable, OutOfDeviceMemory},
    DeviceEngine, DeviceEngineBuffer, DeviceEngineKernel, DeviceId, DeviceInfo, DeviceLost,
    DeviceOptions, Features, KernelDesc, KernelKey,
};
use anyhow::{Error, Result};
use futures_lite::future::yield_now;
use std::{
    future::Future,
    ops::Range,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{sync_channel, Receiver, SyncSender as Sender, TryRecvError},
        Arc,
    },
};
use dashmap::DashMap;
use wgpu::{
    Adapter, Backends, Buffer, BufferDescriptor, BufferUsages, Device, Instance,
    InstanceDescriptor, PowerPreference, Queue, RequestAdapterOptions, MapMode, ComputePipeline, ShaderModuleDescriptor, ShaderSource,
};

fn engine_handle() -> usize {
    static HANDLE: AtomicUsize = AtomicUsize::new(0);
    HANDLE.fetch_add(1, Ordering::Relaxed)
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

pub struct Engine {
    id: DeviceId,
    info: Arc<DeviceInfo>,
    sender: Sender<Op>,
    kernels: DashMap<KernelKey, (usize, Arc<KernelDesc>)>,
}

impl DeviceEngine for Engine {
    type DeviceBuffer = DeviceBuffer;
    type Kernel = Kernel;
    fn new(options: DeviceOptions) -> Pin<Box<dyn Future<Output = Result<Arc<Self>>>>> {
        Box::pin(async move {
            let index = options.index;
            let instance = Instance::new(InstanceDescriptor {
                backends: Backends::BROWSER_WEBGPU,
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await;
            let adapter = if let Some(adapter) = adapter {
                if index > 0 {
                    return Err(DeviceIndexOutOfRange { index, devices: 1 }.into());
                }
                adapter
            } else {
                return Err(DeviceIndexOutOfRange { index, devices: 0 }.into());
            };
            let (device, queue) = adapter.request_device(&Default::default(), None).await?;
            let (sender, receiver) = sync_channel(8);
            let id = DeviceId {
                index,
                handle: engine_handle(),
            };
            let upload_buffer = device.create_buffer(&BufferDescriptor {
                label: None,
                size: DeviceBuffer::HOST_BUFFER_SIZE.try_into().unwrap(),
                usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let download_buffer = device.create_buffer(&BufferDescriptor {
                label: None,
                size: DeviceBuffer::HOST_BUFFER_SIZE.try_into().unwrap(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let worker = Worker {
                id,
                device,
                queue,
                receiver,
                upload_buffer,
                download_buffer,
                buffers: Vec::new(),
                compute_pipelines: Vec::new(),
                adapter,
                instance,
            };
            wasm_bindgen_futures::spawn_local(worker.run());
            let info = Arc::new(DeviceInfo {
                index,
                name: "Web".to_string(),
                compute_queues: 1,
                transfer_queues: 0,
                features: Features::empty(),
            });
            let kernels = DashMap::new();
            Ok(Arc::new(Self { id, info, sender, kernels, }))
        })
    }
    fn id(&self) -> DeviceId {
        self.id
    }
    fn info(&self) -> &Arc<DeviceInfo> {
        &self.info
    }
    fn wait(&self) -> Result<(), DeviceLost> {
        todo!()
    }
}

pub(super) struct DeviceBuffer {
    engine: Arc<Engine>,
    id: usize,
    len: usize,
    offset: usize,
}

impl DeviceBuffer {
    const MAX_LEN: usize = i32::MAX as usize;
    const MAX_SIZE: usize = aligned_ceil(Self::MAX_LEN, Self::ALIGN);
    const ALIGN: usize = 256;
    const HOST_BUFFER_SIZE: usize = 32_000_000;
}

impl DeviceEngineBuffer for DeviceBuffer {
    type Engine = Engine;
    unsafe fn uninit(engine: Arc<Self::Engine>, len: usize) -> Result<Self> {
        let (sender, receiver) = sync_channel(1);
        engine.sender.send(Op::Uninit {
            len: aligned_ceil(len, Self::ALIGN),
            sender,
        })?;
        let id = receiver.recv()?.unwrap();
        Ok(Self {
            engine,
            id,
            len,
            offset: 0,
        })
    }
    fn upload(&self, data: &[u8]) -> Result<()> {
        let mut offset = self.offset;
        for data in data.chunks(DeviceBuffer::HOST_BUFFER_SIZE) {
            self.engine.sender.send(Op::Upload {
                id: self.id,
                offset,
                data: data.to_vec(),
            }).map_err(|_| DeviceLost(self.engine.id))?;
            offset += data.len();
        }
        Ok(())
    }
    fn download(&self, data: &mut [u8]) -> Result<()> {
        let mut offset = self.offset;
        for data in data.chunks_mut(DeviceBuffer::HOST_BUFFER_SIZE) {
            let (sender, receiver) = sync_channel(1);
            self.engine.sender.send(Op::Download {
                id: self.id,
                offset: self.offset,
                len: data.len(),
                sender,
            }).map_err(|_| DeviceLost(self.engine.id))?;
            let vec = receiver.recv().map_err(|_| DeviceLost(self.engine.id))??;
            data.copy_from_slice(&vec);
            offset += data.len();
        }
        Ok(())
    }
    fn transfer(&self, dst: &Self) -> Result<()> {
        let mut vec = vec![0u8; self.len];
        self.download(&mut vec)?;
        dst.upload(&vec)
    }
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn len(&self) -> usize {
        self.len
    }
    fn slice(self: &Arc<Self>, range: Range<usize>) -> Option<Arc<Self>> {
        todo!()
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let (sender, receiver) = sync_channel(1);
        let _ = self.engine.sender.send(Op::DropBuffer {
            id: self.id,
            sender,
        });
        let _ = receiver.recv();
    }
}

pub(super) struct Kernel {
    engine: Arc<Engine>,
    id: usize,
    desc: Arc<KernelDesc>,
}

impl DeviceEngineKernel for Kernel {
    type Engine = Engine;
    type DeviceBuffer = DeviceBuffer;
    fn cached(
        engine: Arc<Self::Engine>,
        key: KernelKey,
        desc_fn: impl FnOnce() -> Result<Arc<KernelDesc>>,
    ) -> Result<Arc<Self>> {
        if let Some((id, desc)) = engine.kernels.get(&key).map(|x| x.clone()) {
            return Ok(Arc::new(Self {
                engine,
                id,
                desc,
            }));
        }
        let desc = desc_fn()?;
        let naga_module = naga::front::spv::parse_u8_slice(bytemuck::cast_slice(desc.spirv.as_slice()), &Default::default()).unwrap();
        todo!();
        let (sender, receiver) = sync_channel(1);
        engine.sender.send(Op::Compile {
            desc: desc.clone(),
            sender,
        }).map_err(|_| DeviceLost(engine.id))?;
        let id = receiver.recv().map_err(|_| DeviceLost(engine.id))??;
        engine.kernels.insert(key, (id, desc.clone()));
        Ok(Arc::new(Self {
            engine,
            id,
            desc,
        }))
    }
    unsafe fn dispatch(
        &self,
        groups: [u32; 3],
        buffers: &[Arc<Self::DeviceBuffer>],
        push_consts: Vec<u8>,
    ) -> Result<()> {
        todo!()
    }
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    fn desc(&self) -> &Arc<KernelDesc> {
        &self.desc
    }
}

struct Worker {
    id: DeviceId,
    device: Device,
    queue: Queue,
    receiver: Receiver<Op>,
    upload_buffer: Buffer,
    download_buffer: Buffer,
    buffers: Vec<Option<Buffer>>,
    compute_pipelines: Vec<ComputePipeline>,
    adapter: Adapter,
    instance: Instance,
}

impl Worker {
    async fn run(mut self) {
        loop {
            match self.receiver.try_recv() {
                Ok(op) => {
                    self.op(op).await;
                }
                Err(TryRecvError::Empty) => {
                    yield_now().await;
                }
                Err(TryRecvError::Disconnected) => {
                    return;
                }
            }
        }
    }
    async fn op(&mut self, op: Op) {
        match op {
            Op::Uninit { len, sender } => {
                let buffer = self.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: len.try_into().unwrap(),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // TODO: handle out of memory
                for (i, slot) in self.buffers.iter_mut().enumerate() {
                    if slot.is_none() {
                        slot.replace(buffer);
                        let _ = sender.send(Ok(i));
                        return;
                    }
                }
                let id = self.buffers.len();
                self.buffers.push(Some(buffer));
                let _ = sender.send(Ok(id));
            }
            Op::DropBuffer { id, sender } => {
                self.buffers[id].take();
                let _ = sender.send(());
            }
            Op::Upload { id, offset, data, } => {
                let offset = offset.try_into().unwrap();
                let size: u64 = data.len().try_into().unwrap();
                let slice = self.upload_buffer.slice(..size);
                let (map_sender, map_receiver) = sync_channel(1);
                slice.map_async(MapMode::Write, move |result| {
                    let _ = map_sender.send(result);
                });
                loop {
                    match map_receiver.try_recv() {
                        Ok(result) => {
                            let _ = result.unwrap();
                            break;
                        }
                        Err(TryRecvError::Empty) => {
                            yield_now().await;
                        }
                        err => {
                            let _ = err.unwrap();
                        }
                    }
                }
                slice.get_mapped_range_mut().copy_from_slice(&data);
                self.upload_buffer.unmap();
                let mut encoder = self.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(
                    &self.upload_buffer,
                    0,
                    &self.buffers[id].as_ref().unwrap(),
                    offset,
                    size,
                );
                self.queue.submit([encoder.finish()]);
            }
            Op::Download { id, offset, len, sender, } => {
                let offset = offset.try_into().unwrap();
                let size: u64 = len.try_into().unwrap();
                let mut encoder = self.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(
                    &self.buffers[id].as_ref().unwrap(),
                    offset,
                    &self.download_buffer,
                    0,
                    size,
                );
                self.queue.submit([encoder.finish()]);
                let slice = self.download_buffer.slice(offset..offset+size);
                let (map_sender, map_receiver) = sync_channel(1);
                slice.map_async(MapMode::Read, move |result| {
                    let _ = map_sender.send(result);
                });
                loop {
                    match map_receiver.try_recv() {
                        Ok(result) => {
                            let _ = result.unwrap();
                            break;
                        }
                        Err(TryRecvError::Empty) => {
                            yield_now().await;
                        }
                        err => {
                            let _ = err.unwrap();
                        }
                    }
                }
                let data = slice.get_mapped_range().to_vec();
                self.download_buffer.unmap();
                let _ = sender.send(Ok(data));
            }
            Op::Compile { desc, sender, } => {
                todo!();
                let module = self.device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: ShaderSource::SpirV(desc.spirv.as_slice().into()),
                });
                todo!()
            }
            Op::Dispatch { id, groups, buffers, push_consts, } => {
                todo!()
            }
        }
    }
}

enum Op {
    Uninit {
        len: usize,
        sender: Sender<Result<usize>>,
    },
    DropBuffer {
        id: usize,
        sender: Sender<()>,
    },
    Upload {
        id: usize,
        offset: usize,
        data: Vec<u8>,
    },
    Download {
        id: usize,
        offset: usize,
        len: usize,
        sender: Sender<Result<Vec<u8>>>,
    },
    Compile {
        desc: Arc<KernelDesc>,
        sender: Sender<Result<usize>>,
    },
    Dispatch {
        id: usize,
        groups: [u32; 3],
        buffers: Vec<(usize, (usize, usize))>,
        push_consts: Vec<u8>,
    },
}
