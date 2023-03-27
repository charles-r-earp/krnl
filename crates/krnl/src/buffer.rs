use crate::{
    device::{Device, DeviceInner},
    scalar::{Scalar, ScalarType, ScalarElem},
};
use krnl_macros::module;
use half::{f16, bf16};
use anyhow::Result;
use dry::{macro_wrap, macro_for};
use paste::paste;
use std::{
    marker::PhantomData,
    mem::{forget, size_of},
    sync::Arc,
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

#[derive(Clone, derive_more::Unwrap)]
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

macro_for!($S in [ScalarSliceRepr,ScalarSliceMutRepr] {
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
        slice.raw.len() / slice.scalar_type().size()
    }
}

pub trait ScalarDataMut: ScalarData {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr;
}

pub trait ScalarDataOwned: ScalarData + From<ScalarBufferRepr> {}

pub struct ScalarBufferRepr {
    raw: RawBuffer,
    scalar_type: ScalarType,
}

/*
impl ScalarBufferRepr {
    fn zeros(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        let buffer = match device.inner() {
            DeviceInner::Host => {
                let width = scalar_type.size();
                match width {
                    1 => BufferRepr::from_vec(vec![0u8; len]).into(),
                    2 => BufferRepr::from_vec(vec![0u16; len]).into(),
                    4 => BufferRepr::from_vec(vec![0u32; len]).into(),
                    8 => BufferRepr::from_vec(vec![0u64; len]).into(),
                    _ => unreachable!(),
                }
            }
            #[cfg(feature = "device")]
            DeviceInner::Device(device) => todo!(),
        };
        Ok(buffer)
    }
}*/

impl<T: Scalar> From<BufferRepr<T>> for ScalarBufferRepr {
    fn from(buffer: BufferRepr<T>) -> Self {
        Self {
            raw: buffer.raw,
            scalar_type: T::scalar_type(),
        }
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

impl ScalarDataMut for ScalarBufferRepr {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr {
        ScalarSliceMutRepr {
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

#[derive(Clone)]
pub struct ScalarSliceMutRepr<'a> {
    raw: RawSlice,
    scalar_type: ScalarType,
    _m: PhantomData<&'a ()>,
}

impl ScalarData for ScalarSliceMutRepr<'_> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData::default(),
        }
    }
}

impl ScalarDataMut for ScalarSliceMutRepr<'_> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr {
        ScalarSliceMutRepr {
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
pub type ScalarSlice<'a> = ScalarBufferBase<ScalarSliceRepr<'a>>;
pub type ScalarSliceMut<'a> = ScalarBufferBase<ScalarSliceMutRepr<'a>>;

impl<S: ScalarDataOwned> ScalarBufferBase<S> {
    unsafe fn uninit(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        macro_wrap!(paste! { 
            match scalar_type {
                macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    ScalarType::[<$T:upper>] => {
                        unsafe {
                            Ok(unsafe { Buffer::<$T>::uninit(device, len)? }.into())
                        } 
                    }
                    _ => unreachable!(),  
                })
            }
        })
    }
    pub fn from_elem(device: Device, len: usize, elem: ScalarElem) -> Result<Self> {
        macro_wrap!(paste! { 
            match elem {
                macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                  ScalarElem::[<$T:upper>](elem) => {
                      unsafe {
                          Ok(Buffer::from_elem(device, len, elem)?.into())
                      }
                  }
                })
                _ => unreachable!(),
            }
        })
    }
    pub fn zeros(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Self::from_elem(device, len, ScalarElem::zero(scalar_type))
    }
    pub fn ones(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Self::from_elem(device, len, ScalarElem::one(scalar_type))
    }
}

impl<S: ScalarData> ScalarBufferBase<S> {
    pub fn device(&self) -> Device {
        self.data.device()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.data.scalar_type()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn as_scalar_slice(&self) -> ScalarSlice {
        let data = self.data.as_scalar_slice();
        ScalarSlice {
            data
        }
    }
    pub fn as_scalar_slice_mut(&mut self) -> ScalarSliceMut where S: ScalarDataMut {
        let data = self.data.as_scalar_slice_mut();
        ScalarSliceMut {
            data
        }
    }
}

impl<S: ScalarData> ScalarBufferBase<S> {
    pub fn fill(&mut self, elem: ScalarElem) -> Result<()> where S: ScalarDataMut {
        let slice = self.as_scalar_slice_mut();
        macro_wrap!(paste! { match elem {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                ScalarElem::[<$T:upper>](elem) => SliceMut::<$T>::try_from(slice).ok().unwrap().fill(elem),
            })
            _ => unreachable!(),
        }})
    }
}

#[cfg(feature = "device")]
impl ScalarSlice<'_> {
    pub(crate) fn device_buffer(&self) -> Option<&DeviceBuffer> {
        if let RawSliceInner::Device(buffer) = &self.data.raw.inner {
            Some(buffer)
        } else {
            None
        }
    }
}

#[cfg(feature = "device")]
impl ScalarSliceMut<'_> {
    pub(crate) fn device_buffer_mut(&self) -> Option<&DeviceBuffer> {
        if let RawSliceInner::Device(buffer) = &self.data.raw.inner {
            Some(buffer)
        } else {
            None
        }
    }
}

impl<T: Scalar, S: ScalarDataOwned> From<Buffer<T>> for ScalarBufferBase<S> {
    fn from(buffer: Buffer<T>) -> Self {
        let data = ScalarBufferRepr::from(buffer.data).into();
        Self {
            data
        }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for ScalarSlice<'a> {
    fn from(slice: Slice<'a, T>) -> Self {
        let data = ScalarSliceRepr {
            raw: slice.data.raw,
            scalar_type: T::scalar_type(),
            _m: PhantomData::default(),
        };
        Self {
            data
        }
    }
}

impl<'a, T: Scalar> From<SliceMut<'a, T>> for ScalarSliceMut<'a> {
    fn from(slice: SliceMut<'a, T>) -> Self {
        let data = ScalarSliceMutRepr {
            raw: slice.data.raw,
            scalar_type: T::scalar_type(),
            _m: PhantomData::default(),
        };
        Self {
            data
        }
    }
}

macro_for!($S in [BufferRepr] {
    impl<T: Scalar> Sealed for $S<T> {}
    unsafe impl<T: Scalar> Send for $S<T> {}
    unsafe impl<T: Scalar> Sync for $S<T> {}
});

macro_for!($S in [SliceRepr, SliceMutRepr] {
    impl<T: Scalar> Sealed for $S<'_, T> {}
    unsafe impl<T: Scalar> Send for $S<'_, T> {}
    unsafe impl<T: Scalar> Sync for $S<'_, T> {}
});

pub trait Data: ScalarData {
    type Elem: Scalar;
    fn as_slice(&self) -> SliceRepr<Self::Elem>;
    fn as_host_slice(&self) -> Option<&[Self::Elem]> {
        self.as_slice().into_host_slice()
    }
}

pub trait DataMut: Data + ScalarDataMut {
    fn as_slice_mut(&mut self) -> SliceMutRepr<Self::Elem>;
    fn as_host_slice_mut(&mut self) -> Option<&mut [Self::Elem]> {
        self.as_slice_mut().into_host_slice_mut()
    }
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
                let cap = len * width;
                let device_buffer = unsafe { DeviceBuffer::uninit(device.clone(), cap)? };
                let raw = RawBuffer {
                    slice: RawSlice {
                        inner: RawSliceInner::Device(device_buffer),
                    },
                    cap,
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
            (RawSliceInner::Host(raw), DeviceInner::Host) => Ok(Self::from_vec(self.into_vec()?)),
            _ => self.as_slice().to_device(device),
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

impl<T: Scalar> ScalarDataMut for BufferRepr<T> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr {
        ScalarSliceMutRepr {
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

impl<T: Scalar> DataMut for BufferRepr<T> {
    fn as_slice_mut(&mut self) -> SliceMutRepr<Self::Elem> {
        SliceMutRepr {
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
    fn into_host_slice(self) -> Option<&'a [T]> {
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
            (RawSliceInner::Host(raw), DeviceInner::Host) => {
                Ok(BufferRepr::from_vec(self.to_vec()?))
            }
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
            #[cfg(feature = "device")]
            (RawSliceInner::Device(device_buffer), DeviceInner::Device(device)) => {
                let len = device_buffer.len();
                let device_buffer = device_buffer.transfer(device.clone())?;
                let slice = RawSlice {
                    inner: RawSliceInner::Device(device_buffer),
                };
                let raw = RawBuffer {
                    slice,
                    cap: len,
                    width: size_of::<T>(),
                };
                Ok(BufferRepr {
                    raw,
                    _m: PhantomData::default(),
                })
            }
            _ => todo!(),
        }
    }
}

impl<'a, T: Scalar> ScalarData for SliceRepr<'a, T> {
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


impl<'a, T: Scalar> TryFrom<ScalarSliceRepr<'a>> for SliceRepr<'a, T> {
    type Error = ScalarSliceRepr<'a>;
    fn try_from(slice: ScalarSliceRepr<'a>) -> Result<Self, Self::Error> {
        if slice.scalar_type() == T::scalar_type() {
            Ok(Self {
                raw: slice.raw,
                _m: Default::default()
            })
        } else {
            Err(slice)
        }
    }
}

pub struct SliceMutRepr<'a, T> {
    raw: RawSlice,
    _m: PhantomData<&'a T>,
}

impl<'a, T: Scalar> SliceMutRepr<'a, T> {
    fn into_host_slice_mut(self) -> Option<&'a mut [T]> {
        match &self.raw.inner {
            RawSliceInner::Host(raw) => {
                let slice =
                    unsafe { std::slice::from_raw_parts_mut(raw.ptr as _, raw.len / size_of::<T>()) };
                Some(slice)
            }
            #[cfg(feature = "device")]
            _ => None,
        }
    }
}

impl<T: Scalar> ScalarData for SliceMutRepr<'_, T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: T::scalar_type(),
            _m: PhantomData::default(),
        }
    }
}

impl<T: Scalar> ScalarDataMut for SliceMutRepr<'_, T> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr {
        ScalarSliceMutRepr {
            raw: self.raw.clone(),
            scalar_type: T::scalar_type(),
            _m: PhantomData::default(),
        }
    }
}

impl<T: Scalar> Data for SliceMutRepr<'_, T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<T> {
        SliceRepr {
            raw: self.raw.clone(),
            _m: Default::default(),
        }
    }
}


impl<T: Scalar> DataMut for SliceMutRepr<'_, T> {
    fn as_slice_mut(&mut self) -> SliceMutRepr<T> {
        SliceMutRepr {
            raw: self.raw.clone(),
            _m: Default::default(),
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarSliceMutRepr<'a>> for SliceMutRepr<'a, T> {
    type Error = ScalarSliceMutRepr<'a>;
    fn try_from(slice: ScalarSliceMutRepr<'a>) -> Result<Self, Self::Error> {
        if slice.scalar_type() == T::scalar_type() {
            Ok(Self {
                raw: slice.raw,
                _m: Default::default()
            })
        } else {
            Err(slice)
        }
    }
}

pub struct BufferBase<S: Data> {
    data: S,
}

pub type Buffer<T> = BufferBase<BufferRepr<T>>;
pub type Slice<'a, T> = BufferBase<SliceRepr<'a, T>>;
pub type SliceMut<'a, T> = BufferBase<SliceMutRepr<'a, T>>;

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

impl<'a, T: Scalar> TryFrom<ScalarSlice<'a>> for Slice<'a, T> {
    type Error = ScalarSlice<'a>;
    fn try_from(slice: ScalarSlice<'a>) -> Result<Self, Self::Error> {
        match slice.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarSlice { data })
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarSliceMut<'a>> for SliceMut<'a, T> {
    type Error = ScalarSliceMut<'a>;
    fn try_from(slice: ScalarSliceMut<'a>) -> Result<Self, Self::Error> {
        match slice.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarSliceMut { data })
        }
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> BufferBase<S> {
    unsafe fn uninit(device: Device, len: usize) -> Result<Self> {
        let data = unsafe { BufferRepr::uninit(device, len)?.into() };
        Ok(Self { data })
    }
    pub fn from_elem(device: Device, len: usize, elem: T) -> Result<Self> {
        let mut output = unsafe { Buffer::uninit(device, len)? };
        output.fill(elem)?;
        Ok(Self {
            data: output.data.into(),
        })
    }
    pub fn zeros(device: Device, len: usize) -> Result<Self> {
        Self::from_elem(device, len, T::zero())
    }
    pub fn ones(device: Device, len: usize) -> Result<Self> {
        Self::from_elem(device, len, T::one())
    }
}

impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
    pub fn device(&self) -> Device {
        self.data.device()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.data.scalar_type()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_slice(&self) -> Slice<T> {
        let data = self.data.as_slice();
        Slice { data }
    }
    pub fn as_slice_mut(&mut self) -> SliceMut<T>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut();
        SliceMut { data }
    }
    pub fn as_host_slice(&self) -> Option<&[T]> {
        self.data.as_host_slice()
    }
    pub fn as_host_slice_mut(&mut self) -> Option<&mut [T]> where S: DataMut {
        self.data.as_host_slice_mut()
    }
    pub fn as_scalar_slice(&self) -> ScalarSlice
    {
        let data = self.data.as_scalar_slice();
        ScalarSlice { data }
    }
    pub fn as_scalar_slice_mut(&mut self) -> ScalarSliceMut
    where
        S: DataMut,
    {
        let data = self.data.as_scalar_slice_mut();
        ScalarSliceMut { data }
    }
    pub fn to_device(&self, device: Device) -> Result<Buffer<T>> {
        let data = self.data.as_slice().to_device(device)?;
        Ok(Buffer { data })
    }
    pub fn to_vec(&self) -> Result<Vec<T>> {
        self.data.as_slice().to_vec()
    }
}

impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
    pub fn fill(&mut self, elem: T) -> Result<()> where S: DataMut {
        use dry::macro_wrap;
        use paste::paste;
        
        if self.is_empty() {
            return Ok(());
        }
        if let Some(y) = self.as_host_slice_mut() {
            for y in y.iter_mut() {
                *y = elem;
            }  
            return Ok(());
        } 
        #[cfg(feature = "device")] {
            let x = ScalarElem::from(elem);
            let y = self.as_scalar_slice_mut();
            macro_wrap!(
                paste! {
                    match x {
                        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                            ScalarElem::[<$T:upper>](x) => {
                                kernels::[<fill_ $T>]::builder()?
                                .build(y.device())?
                                .dispatch(x, y.try_into().ok().unwrap())?;
                            }
                        })
                    _ => unreachable!(),
                }
            });
        }
        Ok(())
    }
    pub fn cast<Y: Scalar>(&self) -> Result<Buffer<Y>> {
        if self.is_empty() {
            return Buffer::zeros(self.device(), 0);
        }
        if let Some(x) = self.as_host_slice() {
            let y: Vec<Y> = x.iter().map(|x| x.cast()).collect();
            return Ok(Buffer::from(y));
        }
        #[cfg(feature = "device")] {
            let mut output = unsafe { 
                Buffer::uninit(self.device(), self.len())?
            };
            let x = self.as_scalar_slice();
            let y = output.as_scalar_slice_mut();
            
            macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    if T::scalar_type() == $X::scalar_type() && Y::scalar_type() == $Y::scalar_type() {
                        paste! {
                            kernels::[<cast_ $X _ $Y>]::builder()?.build(y.device())?.dispatch(x.try_into().ok().unwrap(), y.try_into().ok().unwrap())?;
                        }
                        return Ok(output);
                    }
                });
            });
        }
        unreachable!()
    }
}

//#[cfg(feature = "device")]
#[module]
#[krnl(crate=crate)]
//#[krnl(no_build)]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl_core; 
    use krnl_core::{half::{f16, bf16}, macros::kernel};
    #[cfg(target_arch = "spirv")]
    use krnl_core::scalar::Scalar;
    use dry::macro_for;
    use paste::paste;
    
    macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64]  {
       paste! {
           #[kernel(threads(128))]
           pub fn [<fill_ $T>](x: $T, #[item] y: &mut $T) {
               *y = x;
           }
       } 
    });
    
    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                #[kernel(threads(128))]
                pub fn [<cast_ $X _ $Y>](#[item] x: $X, #[item] y: &mut $Y) {
                    *y = Default::default();
                    *y = x.cast();
                }
            }
        });     
    });   
}

/*
#[cfg(feature = "cast")]        
#[module]
#[krnl(crate=crate)]
mod cast_kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl_core;
    use krnl_core::{scalar::Scalar, dry::macro_for, paste::paste, half::{f16, bf16}};
    use krnl_core::krnl_macros::kernel;
            
    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                #[kernel(threads(128))]
                pub fn [<cast_ $X _ $Y>](#[item] x: &$X, #[item] y: &mut $Y) {
                    *y = x.cast();
                }
            }
        });     
    });   
} */  


/*
#[cfg(feature = "device")]
pub fn saxpy(x: Slice<f32>, alpha: f32, y: SliceMut<f32>) -> Result<()> {
    use crate::device::RawKernel;
    use once_cell::sync::Lazy;

    assert_eq!(x.device(), y.device());
    assert_eq!(x.len(), y.len());

    static SPIRV: Lazy<Arc<[u32]>> = Lazy::new(|| {
        let glsl = r#"#version 450
            layout(local_size_x=128) in;
            
            readonly buffer X {
                float x[];
            };
            
            buffer Y {
                float y[];
            };
            
            layout(push_constant) uniform PushConsts {
                float alpha;
                uint x_offset;
                uint x_len;
                uint y_offset;
                uint y_len;  
            };
            
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx < min(x_len, y_len)) {
                    y[y_offset + idx] += alpha * x[x_offset + idx];
                }
            }
        "#;
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_auto_bind_uniforms(true);
        let binary_result = compiler
            .compile_into_spirv(
                &glsl,
                shaderc::ShaderKind::Compute,
                "saxpy",
                "main",
                Some(&options),
            )
            .unwrap();
        Arc::from(binary_result.as_binary())
    });
    let device = x.device();
    let kernel = RawKernel::new(device.inner().clone().unwrap_device(), SPIRV.clone(), &[])?;
    let n = x.data.len() as u32;
    let threads = kernel.threads()[0];
    let groups = n / threads + if n % threads != 0 { 1 } else { 0 };
    let groups = [groups, 1, 1];
    let buffers = [
        x.data.raw.inner.unwrap_device(),
        y.data.raw.inner.unwrap_device(),
    ];
    let push_consts = [alpha.into()];
    unsafe { kernel.dispatch(groups, buffers.as_ref(), push_consts.as_ref()) }
}

#[cfg(all(test, feature = "device"))]
#[test]
fn test_saxpy() -> Result<()> {
    let device = Device::builder().build()?;
    let n = 1000;
    let x = Buffer::from(vec![1f32; n]).to_device(device.clone())?;
    let mut y = Buffer::from(vec![1f32; n]).to_device(device.clone())?;
    saxpy(x.as_slice(), 2f32, y.as_slice_mut())?;
    let y = y.to_vec()?;
    if y.iter().any(|x| *x != 3f32) {
        panic!();
    }
    Ok(())
}

#[cfg(feature = "device")]
pub fn fill(y: SliceMut<f32>) -> Result<()> {
    use crate::device::RawKernel;
    use once_cell::sync::Lazy;

    static SPIRV: Lazy<Arc<[u32]>> = Lazy::new(|| {
        let glsl = r#"#version 450
            layout(local_size_x=256) in;
            
            buffer Y {
                float y[];
            };
            
            /*layout(push_constant) uniform PushConsts {
                uint y_offset;
                uint y_len;  
            };*/
            
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                y[idx] = 1;
            }
        "#;
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_auto_bind_uniforms(true);
        let binary_result = compiler
            .compile_into_spirv(
                &glsl,
                shaderc::ShaderKind::Compute,
                "fill",
                "main",
                Some(&options),
            )
            .unwrap();
        Arc::from(binary_result.as_binary())
    });
    let device = y.device();
    let kernel = RawKernel::new(device.inner().clone().unwrap_device(), SPIRV.clone(), &[])?;
    let n = y.data.len() as u32;
    let threads = kernel.threads()[0];
    let groups = n / threads + if n % threads != 0 { 1 } else { 0 };
    let groups = [groups, 1, 1];
    let buffers = [y.data.raw.inner.unwrap_device()];
    let push_consts = [];
    unsafe { kernel.dispatch(groups, buffers.as_ref(), push_consts.as_ref()) }
}

#[cfg(all(test, feature = "device"))]
#[test]
fn test_fill1() -> Result<()> {
    let device = Device::builder().build()?;
    let n = 256_000_000;
    let mut y = Buffer::from(vec![0f32; n]).to_device(device.clone())?;
    for _ in 0..10 {
        fill(y.as_slice_mut())?;
        device.wait()?;
    }
    Ok(())
}

#[cfg(feature = "device")]
pub fn fill2(y: SliceMut<f32>) -> Result<()> {
    use crate::device::RawKernel;
    use once_cell::sync::Lazy;

    static SPIRV: Lazy<Arc<[u32]>> = Lazy::new(|| {
        let bytes = include_bytes!("../krnl-device-builder/fill.spv").to_vec();
        let words = bytemuck::cast_slice(bytes.as_slice());
        Arc::from(words)
    });
    let device = y.device();
    let kernel = RawKernel::new(device.inner().clone().unwrap_device(), SPIRV.clone(), &[])?;
    let n = y.data.len() as u32;
    let threads = kernel.threads()[0];
    let groups = n / threads + if n % threads != 0 { 1 } else { 0 };
    let groups = [groups, 1, 1];
    let buffers = [y.data.raw.inner.unwrap_device()];
    let push_consts = [];
    unsafe { kernel.dispatch(groups, buffers.as_ref(), push_consts.as_ref()) }
}

#[cfg(all(test, feature = "device"))]
#[test]
fn test_fill2() -> Result<()> {
    let device = Device::builder().build()?;
    let n = 256_000_000;
    let mut y = unsafe { Buffer::uninit(device.clone(), n)? };
    for _ in 0..10 {
        fill2(y.as_slice_mut())?;
        device.wait()?;
    }
    Ok(())
}
*/