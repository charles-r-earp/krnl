#[cfg(feature = "device")]
use crate::device::{Compute, KernelCache};
use crate::{buffer::ScalarSlice, device::Device};
use anyhow::{format_err, Result};
use std::{borrow::Cow, sync::Arc};

#[doc(inline)]
pub use krnl_macros::module;

#[doc(hidden)]
#[path = "../krnl-macros/src/kernel_info.rs"]
pub mod kernel_info;
use kernel_info::{KernelInfo, Spirv};

#[doc(hidden)]
pub struct KernelBase {
    device: Device,
    #[cfg(feature = "device")]
    cache: Arc<KernelCache>,
}

#[doc(hidden)]
pub fn __build(
    device: Device,
    kernel_info: Arc<KernelInfo>,
    spirv: Arc<Spirv>,
) -> Result<KernelBase> {
    #[cfg(feature = "device")]
    if let Some(device_base) = device.as_device() {
        //dbg!(&kernel_info);
        let cache = device_base.kernel_cache(kernel_info, spirv)?;
        return Ok(KernelBase { device, cache });
    }
    let path = &kernel_info.path;
    Err(format_err!("Cannot build kernel `{path}` for {device:?}!"))
}

#[doc(hidden)]
pub unsafe fn __dispatch(
    kernel_base: &KernelBase,
    dispatch_dim: DispatchDim<&[u32]>,
    slices: &[(&'static str, ScalarSlice)],
    push_consts: &[u8],
) -> Result<()> {
    let device = &kernel_base.device;
    #[cfg(feature = "device")]
    if let Some(device_base) = device.as_device() {
        let kernel_info = kernel_base.cache.kernel_info();
        let threads = kernel_info.threads;
        let kernel_path = &kernel_info.path;
        let elements = if kernel_info.elementwise {
            if let DispatchDim::GlobalThreads(&[elements]) = dispatch_dim {
                Some(elements)
            } else {
                unreachable!()
            }
        } else {
            None
        };
        let groups = dispatch_dim.to_dispatch_groups(threads);
        if groups.iter().any(|x| *x == 0) {
            return Ok(());
        }
        let mut buffers = Vec::with_capacity(slices.len());
        for ((name, slice), slice_info) in slices.into_iter().zip(kernel_info.slice_infos()) {
            let slice = slice.clone().into_raw_slice();
            let slice_device = slice.device_ref();
            if slice_device == device {
                let device_slice = slice.as_device_slice().unwrap();
                if slice_info.elementwise {
                    let slice_len = slice.len();
                    let elements = elements.unwrap() as usize;
                    if slice_len != elements {
                        return Err(format_err!("Kernel `{kernel_path}` elementwise slice `{name}` has len {slice_len}, expected {elements}!"));
                    }
                }
                if let Some(device_buffer) = device_slice.device_buffer() {
                    buffers.push(device_buffer.inner());
                } else {
                    return Err(format_err!(
                        "Kernel `{kernel_path}` slice `{name}` is empty!"
                    ));
                }
            } else {
                return Err(format_err!("Kernel `{kernel_path}` slice `{name}` is on {slice_device:?}, expected {device:?}!"));
            }
        }
        let cache = kernel_base.cache.clone();
        let push_consts = bytemuck::cast_slice(push_consts).to_vec();
        return device_base.compute(Compute {
            cache,
            groups,
            buffers,
            push_consts,
        });
    }
    unreachable!()
}

#[derive(Debug)]
pub enum DispatchDim<T> {
    GlobalThreads(T),
    Groups(T),
}

#[doc(hidden)]
impl<T: AsRef<[u32]>> DispatchDim<T> {
    pub fn as_ref(&self) -> DispatchDim<&[u32]> {
        match self {
            Self::GlobalThreads(x) => DispatchDim::GlobalThreads(x.as_ref()),
            Self::Groups(x) => DispatchDim::Groups(x.as_ref()),
        }
    }
}

impl DispatchDim<&[u32]> {
    fn to_dispatch_groups(&self, threads: [u32; 3]) -> [u32; 3] {
        let mut output = [1; 3];
        match self {
            Self::GlobalThreads(global_threads) => {
                for ((gt, t), y) in global_threads
                    .iter()
                    .zip(threads.iter())
                    .zip(output.iter_mut())
                {
                    *y = *gt / *t + if *gt % *t != 0 { 1 } else { 0 };
                }
            }
            Self::Groups(groups) => {
                for (g, y) in groups.iter().zip(output.iter_mut()) {
                    *y = *g;
                }
            }
        }
        output
    }
}
