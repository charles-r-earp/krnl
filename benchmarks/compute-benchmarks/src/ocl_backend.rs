use krnl::anyhow::Result;
use ocl::{flags::MemFlags, Buffer, Device, Platform, ProQue};

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
    pub fn upload(&self, x: &[f32]) -> Result<Upload> {
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
        Ok(Upload { x_device })
    }
    pub fn download(&self, x: &[f32]) -> Result<Download> {
        let x_host = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .copy_host_slice(x)
            .len(x.len())
            .build()?;
        let x_device = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .len(x.len())
            .flags(MemFlags::READ_WRITE | MemFlags::HOST_NO_ACCESS)
            .build()?;
        x_host.copy(&x_device, None, None).enq()?;
        self.pro_que.finish()?;
        Ok(Download {
            pro_que: self.pro_que.clone(),
            x_device,
            #[cfg(debug_assertions)]
            x_host: x.to_vec(),
        })
    }
    pub fn saxpy(&self, x: &[f32], alpha: f32, y: &[f32]) -> Result<Saxpy> {
        assert_eq!(x.len(), y.len());
        let queue = self.pro_que.queue();
        let x_host = Buffer::builder()
            .queue(queue.clone())
            .copy_host_slice(x)
            .len(x.len())
            .build()?;
        let x_device = Buffer::builder()
            .queue(queue.clone())
            .len(x.len())
            .flags(MemFlags::READ_WRITE | MemFlags::HOST_NO_ACCESS)
            .build()?;
        x_host.copy(&x_device, None, None).enq()?;
        let y_host = Buffer::builder()
            .queue(queue.clone())
            .copy_host_slice(y)
            .len(y.len())
            .build()?;
        let y_device = Buffer::builder()
            .queue(queue.clone())
            .len(x.len())
            .flags(MemFlags::READ_WRITE | MemFlags::HOST_NO_ACCESS)
            .build()?;
        y_host.copy(&y_device, None, None).enq()?;
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

pub struct Upload {
    #[allow(dead_code)]
    x_device: Buffer<f32>,
}

pub struct Download {
    pro_que: ProQue,
    x_device: Buffer<f32>,
    #[cfg(debug_assertions)]
    x_host: Vec<f32>,
}

impl Download {
    pub fn run(&self) -> Result<()> {
        let x_host = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .len(self.x_device.len())
            .flags(MemFlags::HOST_READ_ONLY)
            .build()?;
        self.x_device.copy(&x_host, None, None).enq()?;
        self.pro_que.finish()?;
        let mut x_vec = vec![0f32; x_host.len()];
        x_host.read(&mut x_vec).enq()?;
        #[cfg(debug_assertions)]
        {
            assert_eq!(x_vec, self.x_host);
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
