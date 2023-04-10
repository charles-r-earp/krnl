#[cfg(debug_assertions)]
use crate::saxpy_host;
use anyhow::Result;
#[cfg(debug_assertions)]
use approx::assert_relative_eq;
use autograph::{
    buffer::{Buffer, Slice},
    device::Device,
    shader::Module,
};
use blocker::Blocker;

#[derive(Clone)]
pub struct AutographBackend {
    device: Device,
}

impl AutographBackend {
    pub fn new(index: usize) -> Result<Self> {
        Ok(Self {
            device: Device::from_index(index)?,
        })
    }
    pub fn alloc(&self, len: usize) -> Result<Alloc> {
        let x_device = unsafe { Buffer::alloc(self.device.clone(), len)? };
        Ok(Alloc {
            _x_device: x_device,
        })
    }
    pub fn upload(&self, x: &[f32]) -> Result<Upload> {
        Ok(Upload {
            device: self.device.clone(),
            x_host: x.to_vec(),
        })
    }
    pub fn download(&self, x: &[f32]) -> Result<Download> {
        let x_device = Slice::from(x).into_device(self.device.clone()).block()?;
        Ok(Download {
            #[cfg(debug_assertions)]
            x_host: x.to_vec(),
            x_device,
        })
    }
    pub fn saxpy(&self, x: &[f32], alpha: f32, y: &[f32]) -> Result<Saxpy> {
        assert_eq!(x.len(), y.len());
        let device = self.device.clone();
        let x_device = Slice::from(x).into_device(device.clone()).block()?;
        let y_device = Slice::from(y).into_device(device.clone()).block()?;
        device.sync().block()?;
        #[cfg(debug_assertions)]
        let y_host = {
            let mut y_host = y.to_vec();
            saxpy_host(x, alpha, &mut y_host);
            y_host
        };
        let spirv = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader-builder/shader.spv"
        ));
        let module = Module::from_spirv(spirv.to_vec()).unwrap();
        Ok(Saxpy {
            device,
            module,
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
    device: Device,
    x_host: Vec<f32>,
}

impl Upload {
    pub fn run(&self) -> Result<()> {
        let y_device = Slice::from(self.x_host.as_slice())
            .into_device(self.device.clone())
            .block()?;
        self.device.sync().block()?;
        #[cfg(debug_assertions)]
        {
            let y_host = y_device.read().block()?;
            assert_eq!(self.x_host.as_slice(), y_host.as_slice());
        }
        #[cfg(not(debug_assertions))]
        {
            let _ = y_device;
        }
        Ok(())
    }
}

pub struct Download {
    #[cfg(debug_assertions)]
    x_host: Vec<f32>,
    x_device: Buffer<f32>,
}

impl Download {
    pub fn run(&self) -> Result<()> {
        let y_host = self.x_device.as_slice().read().block()?.to_vec();
        #[cfg(debug_assertions)]
        {
            assert_eq!(y_host.as_slice(), self.x_host.as_slice());
        }
        #[cfg(not(debug_assertions))]
        {
            let _ = y_host;
        }
        Ok(())
    }
}

#[allow(dead_code)]
pub struct Saxpy {
    device: Device,
    module: Module,
    x_device: Buffer<f32>,
    alpha: f32,
    y_device: Buffer<f32>,
    #[cfg(debug_assertions)]
    y_host: Vec<f32>,
}

impl Saxpy {
    pub fn run(&mut self) -> Result<()> {
        let n = self.y_device.len() as u32;
        unsafe {
            self.module
                .compute_pass("saxpy")?
                .push(n)?
                .slice(self.x_device.as_slice())?
                .push(self.alpha)?
                .slice_mut(self.y_device.as_slice_mut())?
                .submit([n, 1, 1])?;
        }
        self.device.sync().block()?;
        #[cfg(debug_assertions)]
        {
            let y_device = self.y_device.as_slice().read().block()?;
            assert_relative_eq!(self.y_host.as_slice(), y_device.as_slice());
        }
        Ok(())
    }
}
