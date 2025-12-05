use core::ops::RangeBounds;

use crate::Result;
use bytemuck::Pod;
use derive_more::From;

pub mod device;
pub use device::Device;

#[derive(Clone, From, PartialEq, Eq, derive_more::IsVariant)]
pub enum Context {
    Host,
    Device(Device),
}

impl Context {
    pub fn new_default() -> Result<Self> {
        #[cfg(all(feature = "device", not(target_family = "wasm")))]
        {
            Device::builder().build().map(Self::Device)
        }
        #[cfg(any(not(feature = "device"), target_family = "wasm"))]
        {
            Ok(Self::Host)
        }
    }
}

pub(crate) enum Buffer<T> {
    Host(Vec<T>),
    #[cfg(feature = "device")]
    Device(device::Buffer<T>),
}

impl<T: Pod> Buffer<T> {
    pub(crate) unsafe fn uninit(context: Context, len: usize) -> Result<Self> {
        match context {
            Context::Host => Ok(Self::Host(vec![T::zeroed(); len])),
            #[cfg(feature = "device")]
            Context::Device(device) => Ok(Self::Device(unsafe {
                device::Buffer::uninit(device, len)?
            })),
            #[cfg(not(feature = "device"))]
            Context::Device(_) => unreachable!(),
        }
    }
    /*
    pub(crate) fn into_context(self, context: Context) -> Result<Self> {
        if self.context() == context {
            Ok(self)
        } else {
            self.as_slice().to_context(context)
        }
    }
    */
}

impl<T> Buffer<T> {
    /*
    pub(crate) fn context(&self) -> Context {
        match self {
            Self::Host(_) => Context::Host,
            #[cfg(feature = "device")]
            Self::Device(x) => Context::Device(x.device()),
        }
    }
    */
    pub(crate) fn as_slice(&self) -> Slice<'_, T> {
        match self {
            Self::Host(x) => Slice::Host(x.as_slice()),
            #[cfg(feature = "device")]
            Self::Device(x) => Slice::Device(x.as_slice()),
        }
    }
    pub(crate) fn as_slice_mut(&mut self) -> SliceMut<'_, T> {
        match self {
            Self::Host(x) => SliceMut::Host(x.as_mut_slice()),
            #[cfg(feature = "device")]
            Self::Device(x) => SliceMut::Device(x.as_slice_mut()),
        }
    }
}

pub(crate) enum Slice<'a, T> {
    Host(&'a [T]),
    #[cfg(feature = "device")]
    Device(device::Slice<'a, T>),
}

impl<T> Clone for Slice<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Host(x) => Self::Host(x),
            #[cfg(feature = "device")]
            Self::Device(x) => Self::Device(x.clone()),
        }
    }
}

impl<T> Slice<'_, T> {
    pub(crate) fn context(&self) -> Context {
        match self {
            Self::Host(_) => Context::Host,
            #[cfg(feature = "device")]
            Self::Device(x) => Context::Device(x.device()),
        }
    }
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Host(x) => x.len(),
            #[cfg(feature = "device")]
            Self::Device(x) => x.len(),
        }
    }
    /*
    pub(crate) fn as_slice(&self) -> Slice<'_, T> {
        match self {
            Self::Host(x) => Slice::Host(x),
            #[cfg(feature = "device")]
            Self::Device(x) => Slice::Device(x.as_slice()),
        }
    }
    */
    pub(crate) fn slice(self, bounds: impl RangeBounds<usize>) -> Self {
        match self {
            Self::Host(x) => {
                let start_bound = bounds.start_bound().map(|x| *x);
                let end_bound = bounds.end_bound().map(|x| *x);
                Self::Host(&x[(start_bound, end_bound)])
            }
            #[cfg(feature = "device")]
            Self::Device(x) => Self::Device(x.slice(bounds)),
        }
    }
}

impl<T: Pod> Slice<'_, T> {
    pub(crate) fn to_context(&self, context: Context) -> Result<Buffer<T>> {
        match (self, context) {
            (Self::Host(_), Context::Host) => self.to_buffer(),
            #[cfg(feature = "device")]
            (Self::Host(slice), Context::Device(device)) => {
                let mut output = unsafe { device::Buffer::uninit(device, slice.len())? };
                output.as_slice_mut().upload(slice)?;
                Ok(Buffer::Device(output))
            }
            #[cfg(not(feature = "device"))]
            (Self::Host(_), Context::Device(_)) => unreachable!(),
            #[cfg(feature = "device")]
            (Self::Device(slice), Context::Device(device)) => {
                if slice.device() == device {
                    self.to_buffer()
                } else {
                    // TODO: device to device copies
                    self.to_context(Context::Host)?
                        .as_slice()
                        .to_context(device.into())
                }
            }
            #[cfg(all(feature = "device", not(target_family = "wasm")))]
            (Self::Device(slice), Context::Host) => {
                let mut output = vec![T::zeroed(); slice.len()];
                slice.download(&mut output)?;
                Ok(Buffer::Host(output))
            }
            #[cfg(all(feature = "device", target_family = "wasm"))]
            (Self::Device(slice), Context::Host) => todo!(),
        }
    }
    pub(crate) fn to_buffer(&self) -> Result<Buffer<T>> {
        match self {
            Self::Host(x) => Ok(Buffer::Host(x.to_vec())),
            #[cfg(feature = "device")]
            Self::Device(_) => {
                let mut output = unsafe { Buffer::uninit(self.context(), self.len())? };
                let slice = crate::buffer::Slice::from_context_slice(self.clone());
                crate::buffer::SliceMut::from_context_slice_mut(output.as_slice_mut())
                    .copy_from_slice(slice)?;
                Ok(output)
            }
        }
    }
    #[cfg(target_family = "wasm")]
    pub(crate) async fn to_vec_async(&self) -> Result<Vec<T>> {
        match self {
            Self::Host(slice) => Ok(slice.to_vec()),
            #[cfg(feature = "device")]
            Self::Device(slice) => {
                let mut output = vec![T::zeroed(); slice.len()];
                slice.download_async(&mut output).await?;
                Ok(output)
            }
        }
    }
}

impl<'a, T: Pod> Slice<'a, T> {
    pub(crate) fn try_bitcast<Y: Pod>(self) -> Option<Slice<'a, Y>> {
        match self {
            Self::Host(x) => bytemuck::try_cast_slice(x).ok().map(Slice::Host),
            #[cfg(feature = "device")]
            Self::Device(x) => x.try_bitcast().map(Slice::Device),
        }
    }
}

pub(crate) enum SliceMut<'a, T> {
    Host(&'a mut [T]),
    #[cfg(feature = "device")]
    Device(device::SliceMut<'a, T>),
}

impl<T> SliceMut<'_, T> {
    pub(crate) fn as_slice(&self) -> Slice<'_, T> {
        match self {
            Self::Host(x) => Slice::Host(x),
            #[cfg(feature = "device")]
            Self::Device(x) => Slice::Device(x.as_slice()),
        }
    }
    pub(crate) fn as_slice_mut(&mut self) -> SliceMut<'_, T> {
        match self {
            Self::Host(x) => SliceMut::Host(x),
            #[cfg(feature = "device")]
            Self::Device(x) => SliceMut::Device(x.as_slice_mut()),
        }
    }
    pub(crate) fn slice_mut(self, bounds: impl RangeBounds<usize>) -> Self {
        match self {
            Self::Host(x) => {
                let start_bound = bounds.start_bound().map(|x| *x);
                let end_bound = bounds.end_bound().map(|x| *x);
                Self::Host(&mut x[(start_bound, end_bound)])
            }
            #[cfg(feature = "device")]
            Self::Device(x) => Self::Device(x.slice_mut(bounds)),
        }
    }
}

impl<T: Pod> SliceMut<'_, T> {
    pub(crate) fn copy_from_slice(&mut self, slice: Slice<T>) -> Result<()> {
        match (slice, self) {
            (Slice::Host(x), Self::Host(y)) => {
                y.copy_from_slice(x);
                Ok(())
            }
            #[cfg(not(feature = "device"))]
            _ => unreachable!(),
            #[cfg(feature = "device")]
            (Slice::Host(x), Self::Device(y)) => y.upload(x),
            #[cfg(all(feature = "device", not(target_family = "wasm")))]
            (Slice::Device(x), Self::Host(y)) => x.download(y),
            #[cfg(all(feature = "device", target_family = "wasm"))]
            (Slice::Device(x), Self::Host(y)) => todo!(),
            #[cfg(feature = "device")]
            (x @ Slice::Device(_), y @ Self::Device(_)) => {
                crate::buffer::SliceMut::from_context_slice_mut(y.as_slice_mut())
                    .copy_from_slice(crate::buffer::Slice::from_context_slice(x))
            }
        }
    }
}

impl<'a, T: Pod> SliceMut<'a, T> {
    pub(crate) fn try_bitcast_mut<Y: Pod>(self) -> Option<SliceMut<'a, Y>> {
        match self {
            Self::Host(x) => bytemuck::try_cast_slice_mut(x).ok().map(SliceMut::Host),
            #[cfg(feature = "device")]
            Self::Device(x) => x.try_bitcast_mut().map(SliceMut::Device),
        }
    }
}
