use crate::{
    device::{Device, DeviceInner},
    scalar::{Scalar, ScalarType},
};
use anyhow::Result;
use dry::macro_for;
use std::{
    marker::PhantomData,
    mem::{forget, size_of},
};

#[cfg(feature = "device")]
use crate::device::DeviceBuffer;

#[derive(Copy, Clone)]
struct RawHostSlice {
    ptr: *mut u8,
    len: usize,
}

#[derive(Clone)]
pub struct RawSlice {
    inner: RawSliceInner,
}

impl RawSlice {
    fn device(&self) -> Device {
        match &self.inner {
            RawSliceInner::Host(_) => Device::host(),
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => buffer.device().into(),
        }
    }
    fn len(&self) -> usize {
        match &self.inner {
            RawSliceInner::Host(raw) => raw.len,
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => buffer.len(),
        }
    }
    /*fn to_device(&self, device: Device) -> Result<RawBuffer> {
        match (&self.inner, device.inner()) {
            #[cfg(feature = "device")]
            (RawSliceInner::Host(raw), DeviceInner::Device(device)) => {
                let data = unsafe { std::slice::from_raw_parts(raw.ptr, raw.len) };
                let device_buffer = DeviceBuffer::upload(device.clone(), data)?;
                let slice = RawSlice {
                    inner: RawSliceInner::Device(device_buffer),
                    width: self.width,
                };
                Ok(RawBuffer {
                    slice,
                    cap: raw.len,
                })
            }
            _ => todo!(),
        }
    }
    fn as_host_slice<T: Scalar>(&self) -> Option<&[T]> {
         #[cfg(feature = "device")]

    }
    fn to_vec<T: Scalar>(&self) -> Result<Vec<T>> {
        match &self.inner {
            #[cfg(feature = "device")]
            RawSliceInner::Host(_) => Ok(self.as_host_slice().unwrap().to_vec()),
            _ => todo!(),
        }
    }*/
}

#[derive(Clone)]
enum RawSliceInner {
    Host(RawHostSlice),
    #[cfg(feature = "device")]
    Device(DeviceBuffer),
}

#[derive(derive_more::Deref, derive_more::DerefMut)]
pub struct RawBuffer {
    #[deref]
    #[deref_mut]
    slice: RawSlice,
    cap: usize,
    width: usize,
}

impl RawBuffer {
    /*fn into_device(self, device: Device) -> Result<Self> {
        if self.device() == device {
            Ok(self)
        } else {
            self.slice.to_device(device)
        }
    }
    fn into_vec<T: Scalar>(self) -> Result<Vec<T>> {
        match self.slice.inner {
            RawSliceInner::Host(raw) => {
                let width = size_of::<T>();
                let vec =
                    unsafe { Vec::from_raw_parts(raw.ptr as _, raw.len / width, self.cap / width) };
                std::mem::forget(self);
                Ok(vec)
            }
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => self.slice.to_vec(),
        }
    }*/
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        if let RawSliceInner::Host(slice) = self.inner {
            let ptr = slice.ptr;
            let len = slice.len * self.width;
            let cap = self.cap * self.width;
            unsafe {
                match self.width {
                    1 => {
                        Vec::<u8>::from_raw_parts(ptr as _, len, cap);
                    }
                    2 => {
                        Vec::<u16>::from_raw_parts(ptr as _, len, cap);
                    }
                    4 => {
                        Vec::<u32>::from_raw_parts(ptr as _, len, cap);
                    }
                    8 => {
                        Vec::<u64>::from_raw_parts(ptr as _, len, cap);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

macro_for!($S in [ScalarBufferRepr] {
    impl Sealed for $S {}
    unsafe impl Send for $S {}
    unsafe impl Sync for $S {}
});

macro_for!($S in [ScalarSliceRepr] {
    impl Sealed for $S<'_> {}
    unsafe impl Send for $S<'_> {}
    unsafe impl Sync for $S<'_> {}
});

pub trait ScalarData: Sealed {
    fn as_scalar_slice(&self) -> ScalarSliceRepr;
    fn device(&self) -> Device {
        self.as_scalar_slice().raw.device()
    }
    fn scalar_type(&self) -> ScalarType {
        self.as_scalar_slice().scalar_type
    }
    fn len(&self) -> usize {
        let slice = self.as_scalar_slice();
        slice.raw.len() * slice.scalar_type().size()
    }
}

pub trait ScalarDataOwned: ScalarData + From<ScalarBufferRepr> {}

pub struct ScalarBufferRepr {
    raw: RawBuffer,
    scalar_type: ScalarType,
}

impl ScalarBufferRepr {
    fn from_buffer_repr<T: Scalar>(buffer: BufferRepr<T>) -> Self {
        Self {
            raw: buffer.raw,
            scalar_type: T::scalar_type(),
        }
    }
    fn zeros(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        let buffer = match device.inner() {
            DeviceInner::Host => {
                let width = scalar_type.size();
                match width {
                    1 => Self::from_buffer_repr(BufferRepr::from_vec(vec![0u8; len])),
                    2 => Self::from_buffer_repr(BufferRepr::from_vec(vec![0u16; len])),
                    4 => Self::from_buffer_repr(BufferRepr::from_vec(vec![0u32; len])),
                    8 => Self::from_buffer_repr(BufferRepr::from_vec(vec![0u64; len])),
                    _ => unreachable!(),
                }
            }
            #[cfg(feature = "device")]
            DeviceInner::Device(device) => todo!(),
        };
        Ok(buffer)
    }
}

impl ScalarData for ScalarBufferRepr {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData::default(),
        }
    }
}

impl ScalarDataOwned for ScalarBufferRepr {}

#[derive(Clone)]
pub struct ScalarSliceRepr<'a> {
    raw: RawSlice,
    scalar_type: ScalarType,
    _m: PhantomData<&'a ()>,
}

impl ScalarData for ScalarSliceRepr<'_> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData::default(),
        }
    }
}

pub struct ScalarBufferBase<S: ScalarData> {
    data: S,
}

pub type ScalarBuffer = ScalarBufferBase<ScalarBufferRepr>;

impl<S: ScalarDataOwned> ScalarBufferBase<S> {
    pub fn zeros(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        let data = ScalarBufferRepr::zeros(device, len, scalar_type)?.into();
        Ok(Self { data })
    }
}

macro_for!($S in [BufferRepr] {
    impl<T: Scalar> Sealed for $S<T> {}
    unsafe impl<T: Scalar> Send for $S<T> {}
    unsafe impl<T: Scalar> Sync for $S<T> {}
});

macro_for!($S in [SliceRepr] {
    impl<T: Scalar> Sealed for $S<'_, T> {}
    unsafe impl<T: Scalar> Send for $S<'_, T> {}
    unsafe impl<T: Scalar> Sync for $S<'_, T> {}
});

pub trait Data: ScalarData {
    type Elem: Scalar;
    fn as_slice(&self) -> SliceRepr<Self::Elem>;
}

pub trait DataOwned: Data + From<BufferRepr<Self::Elem>> {}

pub struct BufferRepr<T: Scalar> {
    raw: RawBuffer,
    _m: PhantomData<T>,
}

impl<T: Scalar> BufferRepr<T> {
    fn from_vec(vec: Vec<T>) -> Self {
        let width = std::mem::size_of::<T>();
        let ptr = vec.as_ptr() as *mut u8;
        let len = vec.len() * width;
        let cap = vec.capacity() * width;
        forget(vec);
        let raw = RawBuffer {
            slice: RawSlice {
                inner: RawSliceInner::Host(RawHostSlice { ptr, len }),
            },
            cap,
            width,
        };
        Self {
            raw,
            _m: PhantomData::default(),
        }
    }
    unsafe fn uninit(device: Device, len: usize) -> Result<Self> {
        match device.inner() {
            DeviceInner::Host => Ok(Self::from_vec(vec![T::zero(); len])),
            #[cfg(feature = "device")]
            DeviceInner::Device(device) => {
                let width = size_of::<T>();
                let device_buffer = unsafe { DeviceBuffer::uninit(device.clone(), len * width)? };
                let raw = RawBuffer {
                    slice: RawSlice {
                        inner: RawSliceInner::Device(device_buffer),
                    },
                    cap: len,
                    width,
                };
                Ok(Self {
                    raw,
                    _m: PhantomData::default(),
                })
            }
        }
    }
    fn into_device(mut self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self);
        }
        match (&self.raw.inner, device.inner()) {
            #[cfg(feature = "device")]
            (RawSliceInner::Device(device_buffer), DeviceInner::Host) => {
                Ok(Self::from_vec(self.into_vec()?))
            }
            _ => todo!(),
        }
    }
    fn into_vec(self) -> Result<Vec<T>> {
        match self.raw.inner {
            RawSliceInner::Host(raw) => {
                let width = size_of::<T>();
                if width == self.raw.width {
                    let vec = unsafe {
                        Vec::from_raw_parts(raw.ptr as _, raw.len / width, self.raw.cap / width)
                    };
                    std::mem::forget(self.raw);
                    Ok(vec)
                } else {
                    Ok(
                        unsafe { std::slice::from_raw_parts(raw.ptr as _, raw.len / width) }
                            .to_vec(),
                    )
                }
            }
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => self.as_slice().to_vec(),
        }
    }
}

impl<T: Scalar> ScalarData for BufferRepr<T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: T::scalar_type(),
            _m: PhantomData::default(),
        }
    }
}

impl<T: Scalar> Data for BufferRepr<T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<Self::Elem> {
        SliceRepr {
            raw: self.raw.slice.clone(),
            _m: PhantomData::default(),
        }
    }
}

impl<T: Scalar> DataOwned for BufferRepr<T> {}

#[derive(Clone)]
pub struct SliceRepr<'a, T: Scalar> {
    raw: RawSlice,
    _m: PhantomData<&'a T>,
}

impl<'a, T: Scalar> SliceRepr<'a, T> {
    fn from_host_slice(host_slice: &'a [T]) -> Self {
        let width = std::mem::size_of::<T>();
        let ptr = host_slice.as_ptr() as *mut u8;
        let len = host_slice.len() * width;
        let raw = RawSlice {
            inner: RawSliceInner::Host(RawHostSlice { ptr, len }),
        };
        Self {
            raw,
            _m: PhantomData::default(),
        }
    }
    fn as_host_slice(&self) -> Option<&'a [T]> {
        match &self.raw.inner {
            RawSliceInner::Host(raw) => {
                let slice =
                    unsafe { std::slice::from_raw_parts(raw.ptr as _, raw.len / size_of::<T>()) };
                Some(slice)
            }
            #[cfg(feature = "device")]
            _ => None,
        }
    }
    fn to_vec(&self) -> Result<Vec<T>> {
        if let Some(slice) = self.as_host_slice() {
            Ok(slice.to_vec())
        } else {
            self.to_device(Device::host())?.into_vec()
        }
    }
    fn to_device(&self, device: Device) -> Result<BufferRepr<T>> {
        match (&self.raw.inner, device.inner()) {
            #[cfg(feature = "device")]
            (RawSliceInner::Host(raw), DeviceInner::Device(device)) => {
                let data = unsafe { std::slice::from_raw_parts(raw.ptr, raw.len) };
                let device_buffer = DeviceBuffer::upload(device.clone(), data)?;
                let slice = RawSlice {
                    inner: RawSliceInner::Device(device_buffer),
                };
                let raw = RawBuffer {
                    slice,
                    cap: raw.len,
                    width: size_of::<T>(),
                };
                Ok(BufferRepr {
                    raw,
                    _m: PhantomData::default(),
                })
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Device(device_buffer), DeviceInner::Host) => {
                let mut vec = vec![T::zero(); device_buffer.len() / size_of::<T>()];
                device_buffer.download(bytemuck::cast_slice_mut(vec.as_mut_slice()))?;
                Ok(BufferRepr::from_vec(vec))
            }
            _ => todo!(),
        }
    }
}

impl<T: Scalar> ScalarData for SliceRepr<'_, T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: T::scalar_type(),
            _m: PhantomData::default(),
        }
    }
}

impl<T: Scalar> Data for SliceRepr<'_, T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<T> {
        self.clone()
    }
}

pub struct BufferBase<S: Data> {
    data: S,
}

pub type Buffer<T> = BufferBase<BufferRepr<T>>;
pub type Slice<'a, T> = BufferBase<SliceRepr<'a, T>>;

impl<T: Scalar, S: DataOwned<Elem = T>> From<Vec<T>> for BufferBase<S> {
    fn from(vec: Vec<T>) -> Self {
        let data = BufferRepr::from_vec(vec).into();
        Self { data }
    }
}

impl<'a, T: Scalar> From<&'a [T]> for Slice<'a, T> {
    fn from(host_slice: &'a [T]) -> Self {
        let data = SliceRepr::from_host_slice(host_slice);
        Self { data }
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> BufferBase<S> {
    pub unsafe fn uninit(device: Device, len: usize) -> Result<Self> {
        let data = unsafe { BufferRepr::uninit(device, len)?.into() };
        Ok(Self { data })
    }
}

impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
    pub fn to_device(&self, device: Device) -> Result<Buffer<T>> {
        let data = self.data.as_slice().to_device(device)?;
        Ok(Buffer { data })
    }
    pub fn to_vec(&self) -> Result<Vec<T>> {
        self.data.as_slice().to_vec()
    }
}
