use anyhow::Result;
use futures::future::Future;
use ocl::{
    flags::MemFlags,
    r#async::{BufferSink, BufferStream},
    Buffer, Device, OclPrm, Platform, ProQue, Queue,
};

trait BufferExt {
    type Elem: OclPrm;
    fn from_slice(queue: Queue, data: &[Self::Elem]) -> Result<Self>
    where
        Self: Sized;
    fn to_vec(&self) -> Result<Vec<Self::Elem>>;
}

impl<T: OclPrm> BufferExt for Buffer<T> {
    type Elem = T;
    fn from_slice(queue: Queue, data: &[Self::Elem]) -> Result<Self>
    where
        Self: Sized,
    {
        let x_host = Buffer::builder()
            .queue(queue.clone())
            .len(data.len())
            .copy_host_slice(data)
            .build()?;
        let x_device = Buffer::builder()
            .queue(queue.clone())
            .len(data.len())
            .flags(MemFlags::READ_WRITE | MemFlags::HOST_NO_ACCESS)
            .build()?;
        x_host.copy(&x_device, None, None).enq()?;
        queue.finish()?;
        Ok(x_device)
    }
    fn to_vec(&self) -> Result<Vec<Self::Elem>> {
        let queue = self.default_queue().unwrap();
        let y_host = Buffer::builder()
            .queue(queue.clone())
            .len(self.len())
            .flags(MemFlags::HOST_READ_ONLY)
            .build()?;
        self.copy(&y_host, None, None).enq()?;
        queue.finish()?;
        let mut y_vec = vec![T::default(); y_host.len()];
        y_host.read(&mut y_vec).enq()?;
        Ok(y_vec)
    }
}

#[derive(Clone)]
pub struct OclBackend {
    pro_que: ProQue,
}

impl OclBackend {
    pub fn new(platform_index: usize, device_index: usize) -> Result<Self> {
        let platform = Platform::list()[platform_index];
        let device = Device::list_all(&platform)?[device_index];
        let pro_que = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(KERNELS)
            .build()?;
        Ok(Self { pro_que })
    }
    pub fn alloc(&self, len: usize) -> Result<Alloc> {
        let x_device = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .len(len)
            .flags(MemFlags::READ_WRITE | MemFlags::HOST_NO_ACCESS)
            .build()?;
        Ok(Alloc {
            _x_device: x_device,
        })
    }
    pub fn upload(&self, x: &[f32]) -> Result<Upload> {
        let y_device = Buffer::from_slice(self.pro_que.queue().clone(), &vec![0f32; x.len()])?;
        Ok(Upload {
            pro_que: self.pro_que.clone(),
            x_host: x.to_vec(),
            y_device,
        })
        /*
        let x_host = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .len(x.len())
            .copy_host_slice(x)
            .build()?;
        let x_device = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .len(x.len())
            .flags(MemFlags::READ_WRITE | MemFlags::HOST_NO_ACCESS)
            .build()?;
        assert_eq!(x_device.len(), x.len());
        x_host.copy(&x_device, None, None).enq()?;
        self.pro_que.finish()?;
        #[cfg(debug_assertions)]
        {
            let y_host = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .len(x.len())
                .build()?;
            x_device.copy(&y_host, None, None).enq()?;
            let mut y_vec = vec![0f32; x.len()];
            y_host.read(&mut y_vec).enq()?;
            assert_eq!(x, y_vec.as_slice());
        }
        Ok(Upload { x_device })*/
    }
    pub fn download(&self, x: &[f32]) -> Result<Download> {
        let queue = self.pro_que.queue();
        let x_host = Buffer::builder()
            .queue(queue.clone())
            .len(x.len())
            .copy_host_slice(x)
            .build()?;
        let x_device = BufferStream::new(queue.clone(), x.len())?;
        x_host.copy(x_device.buffer(), None, None).enq()?;
        queue.finish()?;
        #[cfg(debug_assertions)]
        {
            let y_host = x_device.buffer().to_vec()?;
            assert_eq!(y_host.as_slice(), x);
        }
        Ok(Download {
            pro_que: self.pro_que.clone(),
            #[cfg(debug_assertions)]
            x_host: x.to_vec(),
            x_device,
            y_host: vec![0f32; x.len()],
        })
    }
    pub fn saxpy(&self, x: &[f32], alpha: f32, y: &[f32]) -> Result<Saxpy> {
        assert_eq!(x.len(), y.len());
        let queue = self.pro_que.queue();
        let x_device = Buffer::from_slice(queue.clone(), x)?;
        let y_device = Buffer::from_slice(queue.clone(), y)?;
        #[cfg(debug_assertions)]
        let y_host = {
            let mut y_host = y.to_vec();
            crate::saxpy_host(x, alpha, &mut y_host);
            y_host
        };
        Ok(Saxpy {
            pro_que: self.pro_que.clone(),
            x_device,
            alpha,
            y_device,
            #[cfg(debug_assertions)]
            y_host,
        })
    }
}

pub struct Alloc {
    _x_device: Buffer<f32>,
}

pub struct Upload {
    pro_que: ProQue,
    x_host: Vec<f32>,
    y_device: Buffer<f32>,
}

impl Upload {
    pub fn run(&mut self) -> Result<()> {
        let y_device = unsafe {
            BufferSink::from_buffer(
                self.y_device.clone(),
                Some(self.pro_que.queue().clone()),
                0,
                self.y_device.len(),
            )?
        };
        y_device
            .clone()
            .write()
            .wait()?
            .copy_from_slice(&self.x_host);
        y_device.flush().enq()?.wait()?;
        self.pro_que.finish()?;
        #[cfg(debug_assertions)]
        {
            let y_host = self.y_device.to_vec()?;
            assert_eq!(self.x_host, y_host);
        }
        Ok(())
    }
}

pub struct Download {
    pro_que: ProQue,
    #[cfg(debug_assertions)]
    x_host: Vec<f32>,
    x_device: BufferStream<f32>,
    y_host: Vec<f32>,
}

impl Download {
    pub fn run(&mut self) -> Result<()> {
        self.x_device.clone().flood().enq()?.wait()?;
        self.pro_que.finish()?;
        self.y_host
            .copy_from_slice(&self.x_device.clone().read().wait()?);
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.y_host, self.x_host);
        }
        Ok(())
    }
}

pub struct Saxpy {
    pro_que: ProQue,
    x_device: Buffer<f32>,
    alpha: f32,
    y_device: Buffer<f32>,
    #[cfg(debug_assertions)]
    y_host: Vec<f32>,
}

impl Saxpy {
    pub fn run(&mut self) -> Result<()> {
        let n = self.x_device.len() as u32;
        let lws = 256;
        let wgs = n / lws + if n % lws != 0 { 1 } else { 0 };
        let kernel = self
            .pro_que
            .kernel_builder("saxpy")
            .arg(&n)
            .arg(&self.x_device)
            .arg(&self.alpha)
            .arg(&self.y_device)
            .global_work_size(wgs * lws)
            .local_work_size(lws)
            .build()?;
        unsafe {
            kernel.enq()?;
        }
        self.pro_que.finish()?;
        #[cfg(debug_assertions)]
        {
            let y_host = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .len(self.x_device.len())
                .flags(MemFlags::HOST_READ_ONLY)
                .build()?;
            self.y_device.copy(&y_host, None, None).enq()?;
            self.pro_que.finish()?;
            let mut y_vec = vec![0f32; y_host.len()];
            y_host.read(&mut y_vec).enq()?;
            assert_eq!(y_vec, self.y_host);
        }
        Ok(())
    }
}

static KERNELS: &'static str = r#"
kernel void saxpy(uint n, global float* const x, float alpha, global float* __restrict__ y) {
    uint idx = get_global_id(0);
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}
"#;
