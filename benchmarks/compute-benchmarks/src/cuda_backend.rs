#[cfg(debug_assertions)]
use crate::saxpy_host;
#[cfg(debug_assertions)]
use approx::assert_relative_eq;
use cust::{
    context::Context,
    device::Device,
    launch,
    memory::{DeviceBuffer, DeviceSlice},
    module::Module,
    stream::{Stream, StreamFlags},
};
use krnl::anyhow::Result;
use std::sync::Arc;

struct Cuda {
    module: Module,
    stream: Stream,
    #[allow(unused)]
    context: Context,
    #[allow(unused)]
    device: Device,
}

impl Cuda {
    fn new(index: usize) -> Result<Self> {
        cust::init(cust::CudaFlags::empty())?;
        let device = Device::get_device(index.try_into().unwrap())?;
        let context = Context::new(device)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        let ptx = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
        let ptx = String::from_utf8(ptx.as_ref().to_vec())?;
        let module = Module::from_ptx(&ptx, &[])?;
        Ok(Self {
            module,
            stream,
            context,
            device,
        })
    }
    fn sync(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct CudaBackend {
    cuda: Arc<Cuda>,
}

impl CudaBackend {
    pub fn new(index: usize) -> Result<Self> {
        Ok(Self {
            cuda: Arc::new(Cuda::new(index)?),
        })
    }
    pub fn alloc(&self, len: usize) -> Result<Alloc> {
        let x_device = unsafe { DeviceBuffer::<f32>::uninitialized(len)? };
        Ok(Alloc { x_device })
    }
    pub fn upload(&self, x: &[f32]) -> Result<Upload> {
        #[allow(unused)]
        let x_device = DeviceBuffer::from_slice(x)?;
        self.cuda.sync()?;
        #[cfg(debug_assertions)]
        {
            let x_device = x_device.as_host_vec()?;
            assert_eq!(x, x_device.as_slice());
        }
        Ok(Upload { x_device })
    }
    pub fn download(&self, x: &[f32]) -> Result<Download> {
        let x_device = DeviceBuffer::from_slice(x)?;
        Ok(Download {
            x_device,
            #[cfg(debug_assertions)]
            x_host: x.to_vec(),
        })
    }
    pub fn saxpy(&self, x: &[f32], alpha: f32, y: &[f32]) -> Result<Saxpy> {
        assert_eq!(x.len(), y.len());
        let x_device = DeviceBuffer::from_slice(x)?;
        let y_device = DeviceBuffer::from_slice(y)?;
        #[cfg(debug_assertions)]
        let y_host = {
            let mut y_host = y.to_vec();
            saxpy_host(x, alpha, &mut y_host);
            y_host
        };
        Ok(Saxpy {
            cuda: self.cuda.clone(),
            x_device,
            alpha,
            y_device,
            #[cfg(debug_assertions)]
            y_host,
        })
    }
}

pub struct Alloc {
    #[allow(dead_code)]
    x_device: DeviceBuffer<f32>,
}

pub struct Upload {
    #[allow(dead_code)]
    x_device: DeviceBuffer<f32>,
}

pub struct Download {
    x_device: DeviceBuffer<f32>,
    #[cfg(debug_assertions)]
    x_host: Vec<f32>,
}

impl Download {
    pub fn run(&self) -> Result<()> {
        #[allow(unused)]
        let x_device = self.x_device.as_host_vec()?;
        #[cfg(debug_assertions)]
        {
            assert_eq!(x_device, self.x_host);
        }
        Ok(())
    }
}

pub struct Saxpy {
    cuda: Arc<Cuda>,
    x_device: DeviceBuffer<f32>,
    alpha: f32,
    y_device: DeviceBuffer<f32>,
    #[cfg(debug_assertions)]
    y_host: Vec<f32>,
}

impl Saxpy {
    pub fn run(&mut self) -> Result<()> {
        unsafe {
            saxpy(&self.cuda, &self.x_device, self.alpha, &mut self.y_device)?;
        }
        self.cuda.sync()?;
        #[cfg(debug_assertions)]
        {
            let y_device = self.y_device.as_host_vec()?;
            assert_relative_eq!(self.y_host.as_slice(), y_device.as_slice());
        }
        Ok(())
    }
}

unsafe fn saxpy(
    cuda: &Cuda,
    x: &DeviceSlice<f32>,
    alpha: f32,
    y: &mut DeviceSlice<f32>,
) -> Result<()> {
    let n = x.len() as u32;
    let block = 256;
    let grid = n / block + if n % block != 0 { 1 } else { 0 };
    let shared_memory_size = 0;
    let stream = &cuda.stream;
    let function = cuda.module.get_function("saxpy")?;
    launch!(function<<<grid, block, shared_memory_size, stream>>>(
        n,
        x.as_device_ptr(),
        alpha,
        y.as_device_ptr(),
    ))?;
    Ok(())
}
