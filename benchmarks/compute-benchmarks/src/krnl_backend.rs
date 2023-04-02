#![allow(dead_code, unused_variables)]

#[cfg(debug_assertions)]
use crate::saxpy_host;
#[cfg(debug_assertions)]
use approx::assert_relative_eq;
use krnl::{
    anyhow::Result,
    buffer::{Buffer, Slice, SliceMut},
    device::Device,
    macros::module,
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
    pub fn alloc(&self, len: usize) -> Result<Alloc> {
        let x_device = unsafe { Buffer::uninit(self.device.clone(), len)? };
        Ok(Alloc { x_device })
    }
    pub fn upload(&self, x: &[f32]) -> Result<Upload> {
        let y_device = Buffer::zeros(self.device.clone(), x.len())?;
        self.device.wait()?;
        Ok(Upload {
            x_host: x.to_vec(),
            y_device,
        })
    }
    pub fn download(&self, x: &[f32]) -> Result<Download> {
        let x_device = Slice::from(x).to_device(self.device.clone())?;
        Ok(Download {
            #[cfg(debug_assertions)]
            x_host: x.to_vec(),
            x_device,
            y_host: vec![0f32; x.len()],
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

pub struct Alloc {
    x_device: Buffer<f32>,
}

pub struct Upload {
    x_host: Vec<f32>,
    y_device: Buffer<f32>,
}

impl Upload {
    pub fn run(&mut self) -> Result<()> {
        self.y_device
            .copy_from_slice(&self.x_host.as_slice().into())?;
        #[cfg(debug_assertions)]
        {
            let y_host = self.y_device.to_vec()?;
            assert_eq!(self.x_host, y_host);
        }
        Ok(())
    }
}

pub struct Download {
    #[cfg(debug_assertions)]
    x_host: Vec<f32>,
    x_device: Buffer<f32>,
    y_host: Vec<f32>,
}

impl Download {
    pub fn run(&mut self) -> Result<()> {
        SliceMut::from_host_slice_mut(&mut self.y_host)
            .copy_from_slice(&self.x_device.as_slice())?;
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.y_host, self.x_host);
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
        kernels::saxpy::builder()?
            .specialize(128)?
            .build(self.device.clone())?
            .dispatch(
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

    #[kernel(threads(TS))]
    pub fn saxpy<#[spec] const TS: u32>(#[item] x: f32, alpha: f32, #[item] y: &mut f32) {
        *y += alpha * x;
    }
}
