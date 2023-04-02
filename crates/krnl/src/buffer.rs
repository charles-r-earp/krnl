use crate::{
    device::{Device, DeviceInner},
    scalar::{Scalar, ScalarElem, ScalarType},
};
#[cfg(feature = "device")]
use anyhow::bail;
use anyhow::Result;
use bytemuck::PodCastError;
use dry::{macro_for, macro_wrap};
use half::{bf16, f16};
#[cfg(feature = "device")]
use krnl_macros::module;
use paste::paste;
use std::{
    marker::PhantomData,
    mem::{forget, size_of},
    ops::{Bound, RangeBounds},
};

#[cfg(feature = "device")]
use crate::device::DeviceBuffer;

#[derive(Copy, Clone)]
struct RawHostSlice {
    ptr: *mut u8,
    len: usize,
}

impl RawHostSlice {
    unsafe fn as_bytes<'a>(&self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    unsafe fn as_bytes_mut<'a>(&mut self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
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
    /*#[cfg(feature = "device")]
    fn offset(&self) -> usize {
        match &self.inner {
            RawSliceInner::Host(_) => 0,
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => buffer.offset(),
        }
    }*/
    fn len(&self) -> usize {
        match &self.inner {
            RawSliceInner::Host(raw) => raw.len,
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => buffer.len(),
        }
    }
    fn bitcast(self, scalar_type: ScalarType) -> Result<Self, PodCastError> {
        let (index, len) = match &self.inner {
            RawSliceInner::Host(raw) => (raw.ptr as usize, raw.len),
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => (buffer.offset(), buffer.len()),
        };
        let width = scalar_type.size();
        if index % width != 0 {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else if len % width != 0 {
            Err(PodCastError::OutputSliceWouldHaveSlop)
        } else {
            Ok(self)
        }
    }
    fn slice(mut self, range: impl RangeBounds<usize>, scalar_type: ScalarType) -> Option<Self> {
        let width = scalar_type.size();
        let start = match range.start_bound() {
            Bound::Included(x) => x.checked_mul(width)?,
            Bound::Excluded(x) => x.checked_mul(width)?.checked_add(width)?,
            Bound::Unbounded => 0,
        };
        if start > self.len() {
            return None;
        }
        let end = match range.end_bound() {
            Bound::Included(x) => x.checked_mul(width)?.checked_add(width)?,
            Bound::Excluded(x) => x.checked_mul(width)?,
            Bound::Unbounded => self.len(),
        };
        if end > self.len() {
            return None;
        }
        match &mut self.inner {
            RawSliceInner::Host(raw) => {
                let slice = unsafe { raw.as_bytes().get(start..end)? };
                raw.ptr = slice.as_ptr() as *mut u8;
                raw.len = slice.len();
            }
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => {
                *buffer = buffer.slice(start..end)?;
            }
        }
        Some(self)
    }
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

impl Drop for RawBuffer {
    fn drop(&mut self) {
        #[cfg_attr(not(feature = "device"), allow(irrefutable_let_patterns))]
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
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self>
    where
        Self: Sized,
    {
        Err(self)
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
    fn try_into_scalar_buffer(self) -> Result<Self, Self>
    where
        Self: Sized,
    {
        Ok(self)
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

impl<'a> ScalarSliceRepr<'a> {
    fn bitcast(
        self,
        scalar_type: ScalarType,
    ) -> Result<ScalarSliceRepr<'a>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(scalar_type)?;
        Ok(ScalarSliceRepr {
            raw,
            scalar_type,
            _m: Default::default(),
        })
    }
    fn slice(mut self, range: impl RangeBounds<usize>) -> Option<Self> {
        self.raw = self.raw.slice(range, self.scalar_type)?;
        Some(self)
    }
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

impl<'a> ScalarSliceMutRepr<'a> {
    fn bitcast_mut(
        self,
        scalar_type: ScalarType,
    ) -> Result<ScalarSliceMutRepr<'a>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(scalar_type)?;
        Ok(ScalarSliceMutRepr {
            raw,
            scalar_type,
            _m: Default::default(),
        })
    }
    fn slice(mut self, range: impl RangeBounds<usize>) -> Option<Self> {
        self.raw = self.raw.slice(range, self.scalar_type)?;
        Some(self)
    }
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
    pub unsafe fn uninit(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        macro_wrap!(paste! {
            match scalar_type {
                macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    ScalarType::[<$T:upper>] => Ok(unsafe { Buffer::<$T>::uninit(device, len)? }.into()),
                })
                _ => unreachable!(),
            }
        })
    }
    pub fn from_elem(device: Device, len: usize, elem: ScalarElem) -> Result<Self> {
        macro_wrap!(paste! {
            match elem {
                macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                  ScalarElem::[<$T:upper>](elem) => Ok(Buffer::from_elem(device, len, elem)?.into()),
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
        ScalarSlice { data }
    }
    pub fn as_scalar_slice_mut(&mut self) -> ScalarSliceMut
    where
        S: ScalarDataMut,
    {
        let data = self.data.as_scalar_slice_mut();
        ScalarSliceMut { data }
    }
    pub fn into_owned(self) -> Result<ScalarBuffer> {
        match self.data.try_into_scalar_buffer() {
            Ok(data) => Ok(ScalarBuffer { data }),
            Err(data) => Self { data }.to_owned(),
        }
    }
    pub fn to_owned(self) -> Result<ScalarBuffer> {
        self.cast(self.scalar_type())
    }
    pub fn into_device(self, device: Device) -> Result<ScalarBuffer> {
        if device == self.device() {
            self.into_owned()
        } else {
            self.to_device(device)
        }
    }
    pub fn to_device(&self, device: Device) -> Result<ScalarBuffer> {
        let slice = self.as_scalar_slice();
        macro_wrap!(paste! { match slice.scalar_type() {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                ScalarType::[<$T:upper>] => Ok(Slice::<$T>::try_from(slice).ok().unwrap().to_device(device)?.into()),
            })
            _ => unreachable!(),
        }})
    }
    pub fn fill(&mut self, elem: ScalarElem) -> Result<()>
    where
        S: ScalarDataMut,
    {
        let slice = self.as_scalar_slice_mut();
        macro_wrap!(paste! { match elem {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                ScalarElem::[<$T:upper>](elem) => SliceMut::<$T>::try_from(slice).ok().unwrap().fill(elem),
            })
            _ => unreachable!(),
        }})
    }
    pub fn cast(&self, scalar_type: ScalarType) -> Result<ScalarBuffer> {
        let slice = self.as_scalar_slice();
        macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                paste! {
                    if self.scalar_type() == $X::scalar_type() && scalar_type == $Y::scalar_type() {
                         return Ok(Slice::<$X>::try_from(slice).ok().unwrap().cast::<$Y>()?.into());
                    }
                }
            });
        });
        unreachable!()
    }
    pub fn bitcast(&self, scalar_type: ScalarType) -> Result<ScalarSlice, PodCastError> {
        let data = self.data.as_scalar_slice().bitcast(scalar_type)?;
        Ok(ScalarSlice { data })
    }
    pub fn bitcast_mut(&mut self, scalar_type: ScalarType) -> Result<ScalarSliceMut, PodCastError>
    where
        S: DataMut,
    {
        let data = self.data.as_scalar_slice_mut().bitcast_mut(scalar_type)?;
        Ok(ScalarSliceMut { data })
    }
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<ScalarSlice> {
        let data = self.data.as_scalar_slice().slice(range)?;
        Some(ScalarSlice { data })
    }
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<ScalarSliceMut>
    where
        S: ScalarDataMut,
    {
        let data = self.data.as_scalar_slice_mut().slice(range)?;
        Some(ScalarSliceMut { data })
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
        Self { data }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for ScalarSlice<'a> {
    fn from(slice: Slice<'a, T>) -> Self {
        let data = ScalarSliceRepr {
            raw: slice.data.raw,
            scalar_type: T::scalar_type(),
            _m: PhantomData::default(),
        };
        Self { data }
    }
}

impl<'a, T: Scalar> From<SliceMut<'a, T>> for ScalarSliceMut<'a> {
    fn from(slice: SliceMut<'a, T>) -> Self {
        let data = ScalarSliceMutRepr {
            raw: slice.data.raw,
            scalar_type: T::scalar_type(),
            _m: PhantomData::default(),
        };
        Self { data }
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
    fn try_into_buffer(self) -> Result<BufferRepr<Self::Elem>, Self>
    where
        Self: Sized,
    {
        Err(self)
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
    /*#[cfg(feature = "device")]
    fn into_device(self, device: Device) -> Result<Self> {
        let this_device = self.device();
        if this_device == device {
            Ok(self)
        } else if this_device.is_host() {
            Ok(Self::from_vec(self.into_vec()?))
        } else {
            self.as_slice().to_device(device)
        }
    }*/
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
    fn try_into_buffer(self) -> Result<Self, Self>
    where
        Self: Sized,
    {
        Ok(self)
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

impl<T: Scalar> TryFrom<ScalarBufferRepr> for BufferRepr<T> {
    type Error = ScalarBufferRepr;
    fn try_from(buffer: ScalarBufferRepr) -> Result<Self, Self::Error> {
        if buffer.scalar_type() == T::scalar_type() {
            Ok(Self {
                raw: buffer.raw,
                _m: Default::default(),
            })
        } else {
            Err(buffer)
        }
    }
}

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
    /*fn to_buffer(&self) -> Result<BufferRepr<T>> {
        self.to_device(self.device())
    }*/
    fn to_device(&self, device: Device) -> Result<BufferRepr<T>> {
        let mut output = unsafe { BufferRepr::uninit(device, self.len())? };
        output.as_slice_mut().copy_from_slice(self)?;
        Ok(output)
    }
    fn bitcast<Y: Scalar>(self) -> Result<SliceRepr<'a, Y>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(Y::scalar_type())?;
        Ok(SliceRepr {
            raw,
            _m: PhantomData::default(),
        })
    }
    fn slice(mut self, range: impl RangeBounds<usize>) -> Option<Self> {
        self.raw = self.raw.slice(range, T::scalar_type())?;
        Some(self)
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
                _m: Default::default(),
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
    fn from_host_slice_mut(host_slice: &'a mut [T]) -> Self {
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
    fn into_host_slice_mut(self) -> Option<&'a mut [T]> {
        match &self.raw.inner {
            RawSliceInner::Host(raw) => {
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(raw.ptr as _, raw.len / size_of::<T>())
                };
                Some(slice)
            }
            #[cfg(feature = "device")]
            _ => None,
        }
    }
    fn copy_from_slice(&mut self, src: &SliceRepr<T>) -> Result<()> {
        match (&mut self.raw.inner, &src.raw.inner) {
            (RawSliceInner::Host(dst), RawSliceInner::Host(src)) => {
                unsafe {
                    dst.as_bytes_mut().copy_from_slice(src.as_bytes());
                }
                Ok(())
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Host(dst), RawSliceInner::Device(src)) => {
                src.download(unsafe { dst.as_bytes_mut() })
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Device(dst), RawSliceInner::Host(src)) => {
                dst.upload(unsafe { src.as_bytes() })
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Device(dst), RawSliceInner::Device(src_buffer)) => {
                if dst.device() != src_buffer.device() {
                    return src_buffer.transfer(dst);
                }
                Slice { data: src.clone() }.cast_impl(&mut SliceMut::<T> {
                    data: SliceMutRepr {
                        raw: self.raw.clone(),
                        _m: PhantomData::default(),
                    },
                })
            }
        }
    }
    fn bitcast<Y: Scalar>(self) -> Result<SliceMutRepr<'a, Y>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(Y::scalar_type())?;
        Ok(SliceMutRepr {
            raw,
            _m: PhantomData::default(),
        })
    }
    fn slice(mut self, range: impl RangeBounds<usize>) -> Option<Self> {
        self.raw = self.raw.slice(range, T::scalar_type())?;
        Some(self)
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
                _m: Default::default(),
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
        Self::from_vec(vec)
    }
}

impl<'a, T: Scalar> From<&'a [T]> for Slice<'a, T> {
    fn from(host_slice: &'a [T]) -> Self {
        Self::from_host_slice(host_slice)
    }
}

impl<'a, T: Scalar> From<&'a mut [T]> for SliceMut<'a, T> {
    fn from(host_slice: &'a mut [T]) -> Self {
        Self::from_host_slice_mut(host_slice)
    }
}

impl<T: Scalar> TryFrom<ScalarBuffer> for Buffer<T> {
    type Error = ScalarBuffer;
    fn try_from(buffer: ScalarBuffer) -> Result<Self, Self::Error> {
        match buffer.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarBuffer { data }),
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarSlice<'a>> for Slice<'a, T> {
    type Error = ScalarSlice<'a>;
    fn try_from(slice: ScalarSlice<'a>) -> Result<Self, Self::Error> {
        match slice.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarSlice { data }),
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarSliceMut<'a>> for SliceMut<'a, T> {
    type Error = ScalarSliceMut<'a>;
    fn try_from(slice: ScalarSliceMut<'a>) -> Result<Self, Self::Error> {
        match slice.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarSliceMut { data }),
        }
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> BufferBase<S> {
    pub unsafe fn uninit(device: Device, len: usize) -> Result<Self> {
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
    pub fn from_vec(vec: Vec<T>) -> Self {
        let data = BufferRepr::from_vec(vec).into();
        Self { data }
    }
}

impl<'a, T: Scalar> Slice<'a, T> {
    pub fn from_host_slice(host_slice: &'a [T]) -> Self {
        let data = SliceRepr::from_host_slice(host_slice);
        Self { data }
    }
}

impl<'a, T: Scalar> SliceMut<'a, T> {
    pub fn from_host_slice_mut(host_slice: &'a mut [T]) -> Self {
        let data = SliceMutRepr::from_host_slice_mut(host_slice);
        Self { data }
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
    pub fn as_host_slice_mut(&mut self) -> Option<&mut [T]>
    where
        S: DataMut,
    {
        self.data.as_host_slice_mut()
    }
    pub fn as_scalar_slice(&self) -> ScalarSlice {
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
    pub fn into_owned(self) -> Result<Buffer<T>> {
        match self.data.try_into_buffer() {
            Ok(data) => Ok(Buffer { data }),
            Err(data) => Self { data }.to_owned(),
        }
    }
    pub fn to_owned(&self) -> Result<Buffer<T>> {
        self.cast()
    }
    pub fn into_device(self, device: Device) -> Result<Buffer<T>> {
        if device == self.device() {
            self.into_owned()
        } else {
            self.to_device(device)
        }
    }
    pub fn to_device(&self, device: Device) -> Result<Buffer<T>> {
        let data = self.data.as_slice().to_device(device)?;
        Ok(Buffer { data })
    }
    pub fn into_vec(self) -> Result<Vec<T>> {
        match self.data.try_into_buffer() {
            Ok(data) => data.into_vec(),
            Err(data) => data.as_slice().to_vec(),
        }
    }
    pub fn to_vec(&self) -> Result<Vec<T>> {
        self.data.as_slice().to_vec()
    }
    pub fn fill(&mut self, elem: T) -> Result<()>
    where
        S: DataMut,
    {
        if self.is_empty() {
            return Ok(());
        }
        if let Some(y) = self.as_host_slice_mut() {
            for y in y.iter_mut() {
                *y = elem;
            }
            return Ok(());
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            fn copied_bytes<X: Scalar, Y: Scalar>(x: X) -> Y {
                assert!(size_of::<Y>() >= size_of::<X>());
                let mut y = Y::default();
                for (y, x) in bytemuck::bytes_of_mut(&mut y)
                    .iter_mut()
                    .zip(bytemuck::bytes_of(&x).iter().cycle())
                {
                    *y = *x;
                }
                y
            }

            let device = self.device();
            let features = device.info().unwrap().features();
            if let Ok(y) = self.bitcast_mut::<u64>() {
                let x = copied_bytes(elem);
                return kernels::fill_u64::builder()?.build(device)?.dispatch(x, y);
            }
            if let Ok(y) = self.bitcast_mut::<u32>() {
                let x = copied_bytes(elem);
                return kernels::fill_u32::builder()?.build(device)?.dispatch(x, y);
            }
            if features.shader_int16() {
                if let Ok(y) = self.bitcast_mut::<u16>() {
                    let x = copied_bytes(elem);
                    return kernels::fill_u16::builder()?.build(device)?.dispatch(x, y);
                }
            }
            if features.shader_int8() {
                if let Ok(y) = self.bitcast_mut::<u8>() {
                    let x = copied_bytes(elem);
                    return kernels::fill_u8::builder()?.build(device)?.dispatch(x, y);
                }
            }
            if self.bitcast::<u16>().is_ok() {
                bail!("Device {device:?} does not support 16 bit operations!");
            } else {
                bail!("Device {device:?} does not support 16 bit operations!");
            }
        }
    }
    fn cast_impl<Y: Scalar>(&self, output: &mut SliceMut<Y>) -> Result<()> {
        debug_assert_eq!(self.len(), output.len());
        if output.is_empty() {
            return Ok(());
        }
        if let Some((x, y)) = self.as_host_slice().zip(output.as_host_slice_mut()) {
            for (x, y) in x.iter().zip(y.iter_mut()) {
                *y = x.cast();
            }
            return Ok(());
        }
        #[cfg(feature = "device")]
        {
            let x = self.as_scalar_slice();
            let y = output.as_scalar_slice_mut();
            macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    if T::scalar_type() == $X::scalar_type() && Y::scalar_type() == $Y::scalar_type() {
                        paste! {
                            kernels::[<cast_ $X _ $Y>]::builder()?.build(y.device())?.dispatch(x.try_into().ok().unwrap(), y.try_into().ok().unwrap())?;
                        }
                        return Ok(());
                    }
                });
            });
        }
        unreachable!()
    }
    pub fn cast<Y: Scalar>(&self) -> Result<Buffer<Y>> {
        let mut output = unsafe { Buffer::uninit(self.device(), self.len())? };
        self.cast_impl(&mut output.as_slice_mut())?;
        Ok(output)
    }
    pub fn cast_into<Y: Scalar>(self) -> Result<Buffer<Y>> {
        if T::scalar_type() == Y::scalar_type() {
            let buffer = self.into_owned()?;
            Ok(ScalarBuffer::from(buffer).try_into().ok().unwrap())
        } else {
            self.cast()
        }
    }
    pub fn bitcast<Y: Scalar>(&self) -> Result<Slice<Y>, bytemuck::PodCastError> {
        let data = self.data.as_slice().bitcast()?;
        Ok(Slice { data })
    }
    pub fn bitcast_mut<Y: Scalar>(&mut self) -> Result<SliceMut<Y>, bytemuck::PodCastError>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut().bitcast()?;
        Ok(SliceMut { data })
    }
    pub fn copy_from_slice(&mut self, src: &Slice<T>) -> Result<()>
    where
        S: DataMut,
    {
        self.data.as_slice_mut().copy_from_slice(&src.data)
    }
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<Slice<T>> {
        let data = self.data.as_slice().slice(range)?;
        Some(Slice { data })
    }
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<SliceMut<T>>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut().slice(range)?;
        Some(SliceMut { data })
    }
}

#[cfg(feature = "device")]
#[module]
#[krnl(crate=crate)]
mod kernels {
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{
        half::{bf16, f16},
        scalar::Scalar,
    };
    use paste::paste;

    macro_for!($T in [u8, u16, u32, u64]  {
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
