#![allow(dead_code, unused_variables)]

#[cfg(debug_assertions)]
use crate::saxpy_host;
#[cfg(debug_assertions)]
use approx::assert_relative_eq;
use krnl::{
    anyhow::Result,
    buffer::{Buffer, Slice},
    device::Device,
    krnl_macros::module,
};

#[derive(Clone)]
pub struct KrnlBackend {
    device: Device,
}

impl KrnlBackend {
    pub fn new(index: usize) -> Result<Self> {
        Ok(Self {
            device: Device::builder().index(index).build()?,
        })
    }
    pub fn upload(&self, x: &[f32]) -> Result<Upload> {
        let x_device = Slice::from(x).to_device(self.device.clone())?;
        self.device.wait()?;
        #[cfg(debug_assertions)]
        {
            let x_device = x_device.to_vec()?;
            assert_eq!(x, x_device.as_slice());
        }
        Ok(Upload { x_device })
    }
    pub fn download(&self, x: &[f32]) -> Result<Download> {
        let x_device = Slice::from(x).to_device(self.device.clone())?;
        Ok(Download {
            x_device,
            #[cfg(debug_assertions)]
            x_host: x.to_vec(),
        })
    }
    pub fn saxpy(&self, x: &[f32], alpha: f32, y: &[f32]) -> Result<Saxpy> {
        assert_eq!(x.len(), y.len());
        let device = self.device.clone();
        let x_device = Slice::from(x).to_device(device.clone())?;
        let y_device = Slice::from(y).to_device(device.clone())?;
        device.wait()?;
        #[cfg(debug_assertions)]
        let y_host = {
            let mut y_host = y.to_vec();
            saxpy_host(x, alpha, &mut y_host);
            y_host
        };
        Ok(Saxpy {
            device,
            x_device,
            alpha,
            y_device,
            #[cfg(debug_assertions)]
            y_host,
        })
    }
}

pub struct Upload {
    x_device: Buffer<f32>,
}

pub struct Download {
    x_device: Buffer<f32>,
    #[cfg(debug_assertions)]
    x_host: Vec<f32>,
}

impl Download {
    pub fn run(&self) -> Result<()> {
        #[allow(unused)]
        let x_device = self.x_device.to_vec()?;
        #[cfg(debug_assertions)]
        {
            assert_eq!(x_device, self.x_host);
        }
        Ok(())
    }
}

pub struct Saxpy {
    device: Device,
    x_device: Buffer<f32>,
    alpha: f32,
    y_device: Buffer<f32>,
    #[cfg(debug_assertions)]
    y_host: Vec<f32>,
}

impl Saxpy {
    pub fn run(&mut self) -> Result<()> {
        let n = self.x_device.len() as u32;
        let threads = 128;
        let groups = n / threads + if n % threads != 0 { 1 } else { 0 };
        kernels::saxpy::builder()?
            .build(self.device.clone())?
            .dispatch(
                [groups],
                self.x_device.as_slice(),
                self.alpha,
                self.y_device.as_slice_mut(),
            )?;
        self.device.wait()?;
        #[cfg(debug_assertions)]
        {
            let y_device = self.y_device.to_vec()?;
            assert_relative_eq!(self.y_host.as_slice(), y_device.as_slice());
        }
        Ok(())
    }
}

#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::krnl_macros::kernel;

    #[kernel(threads(128))]
    pub fn saxpy(#[global] x: &[f32], alpha: f32, #[global] y: &mut [f32]) {
        let idx = global_id as usize;
        if idx < x.len().min(y.len()) {
            unsafe {
                *y.unsafe_index_mut(idx) += alpha * x[idx];
            }
        }
    }
}
