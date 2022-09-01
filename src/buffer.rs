use crate::{
    device::{Device, DeviceInner, DeviceKind},
    future::BlockableFuture,
    scalar::{Scalar, ScalarType, ScalarElem},
};
#[cfg(feature = "device")]
use crate::{
    device::{DeviceBase, DeviceBuffer, HostBuffer},
    kernel::{module, Kernel},
    krnl_core,
};
use core::{marker::PhantomData, mem::size_of};
use futures_util::future::ready;
use std::{pin::Pin, sync::Arc};
use anyhow::Result;

#[doc(inline)]
pub use krnl_types::kernel::{Module, KernelInfo};

type PinBox<T> = Pin<Box<T>>;

pub mod error {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("The slice is on Device({}), not the host!", index)]
    pub struct SliceOnDeviceError {
        index: usize,
    }

    impl SliceOnDeviceError {
        pub(super) fn new(index: usize) -> Self {
            Self { index }
        }
    }
}
use error::*;

#[derive(Clone, Copy, Debug)]
struct HostSlice {
    ptr: *mut u8,
    len: usize,
}

impl HostSlice {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr() as *mut u8,
            len: bytes.len(),
        }
    }
    /*
    fn new<T: Scalar>(ptr: *mut T, len: usize) -> Self {
        assert_eq!(len % size_of::<T>(), 0);
        Self {
            ptr: ptr as *mut u8,
            len: len / size_of::<T>(),
        }
    }*/
    unsafe fn as_slice<T: Scalar>(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr as *const T, self.len / size_of::<T>()) }
    }
    unsafe fn as_slice_mut<T: Scalar>(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut T, self.len / size_of::<T>()) }
    }
}

#[derive(Clone, derive_more::Unwrap)]
enum RawSliceInner {
    Host(HostSlice),
    #[cfg(feature = "device")]
    Device(Option<Arc<DeviceBuffer>>),
}

impl RawSliceInner {
    fn len(&self) -> usize {
        match self {
            Self::Host(slice) => slice.len,
            #[cfg(feature = "device")]
            Self::Device(buffer) => buffer.as_ref().map_or(0, |x| x.len()),
        }
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct RawSlice {
    device: Device,
    scalar_type: ScalarType,
    inner: RawSliceInner,
}

impl RawSlice {
    pub(crate) fn len(&self) -> usize {
        self.inner.len() / self.scalar_type.size()
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }
    fn from_bytes(scalar_type: ScalarType, bytes: &[u8]) -> Self {
        Self {
            device: Device::host(),
            scalar_type,
            inner: RawSliceInner::Host(HostSlice::from_bytes(bytes)),
        }
    }
    fn as_host_slice<T: Scalar>(&self) -> Result<&[T], SliceOnDeviceError> {
        assert_eq!(T::scalar_type(), self.scalar_type);
        match &self.inner {
            RawSliceInner::Host(slice) => Ok(unsafe { slice.as_slice() }),
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => Err(SliceOnDeviceError::new(
                self.device.inner.clone().unwrap_device().index(),
            )),
        }
    }
    fn as_host_slice_mut<T: Scalar>(&mut self) -> Result<&mut [T], SliceOnDeviceError> {
        assert_eq!(T::scalar_type(), self.scalar_type);
        match &mut self.inner {
            RawSliceInner::Host(slice) => Ok(unsafe { slice.as_slice_mut() }),
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => Err(SliceOnDeviceError::new(
                self.device.inner.clone().unwrap_device().index(),
            )),
        }
    }
    fn to_vec<T: Scalar>(&self) -> Result<Vec<T>, SliceOnDeviceError> {
        Ok(self.as_host_slice()?.to_vec())
    }
    fn to_raw_buffer(&self) -> Result<RawBuffer> {
        match &self.inner {
            RawSliceInner::Host(_) => match self.scalar_type.size() {
                1 => Ok(RawBuffer::from_vec(self.to_vec::<u8>()?)),
                2 => Ok(RawBuffer::from_vec(self.to_vec::<u16>()?)),
                4 => Ok(RawBuffer::from_vec(self.to_vec::<u32>()?)),
                8 => Ok(RawBuffer::from_vec(self.to_vec::<u64>()?)),
                _ => unreachable!(),
            },
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => todo!(),
        }
    }
    #[cfg(feature = "device")]
    pub(crate) fn device_buffer(&self) -> Option<&Arc<DeviceBuffer>> {
        if let RawSliceInner::Device(Some(x)) = &self.inner {
            Some(x)
        } else {
            None
        }
    }
    fn split_at(&self, mid: usize) -> (Self, Self) {
        todo!()
        /*
        let mid = mid * self.scalar_type.size();
        assert!(mid < self.len);
        let a = Self {
            device: self.device.clone(),
            scalar_type: self.scalar_type,
            ptr: self.ptr.clone(),
            offset: self.offset,
            len: mid,
        };
        let b = Self {
            device: self.device.clone(),
            scalar_type: self.scalar_type,
            ptr: self.ptr.clone(),
            offset: self.offset + mid,
            len: self.len - mid,
        };
        (a, b)*/
    }
    fn split_at_mut(&mut self, mid: usize) -> (Self, Self) {
        todo!() // self.split_at(mid)
    }
    fn to_device(
        &self,
        device: Device,
    ) -> Result<PinBox<dyn BlockableFuture<Output = Result<RawBuffer>>>> {
        if self.len() == 0 {
            Ok(Box::pin(ready(unsafe {
                RawBuffer::uninit(device, self.scalar_type, 0)
            })))
        } else if self.device == device {
            Ok(Box::pin(ready(self.to_raw_buffer())))
        } else {
            #[cfg(feature = "device")]
            let scalar_type = self.scalar_type;
            match (&self.device.inner, &device.inner) {
                #[cfg(feature = "device")]
                (DeviceInner::Host, DeviceInner::Device(dst_device)) => {
                    let host_slice = self.inner.clone().unwrap_host();
                    let bytes = unsafe { host_slice.as_slice() };
                    let device_buffer = dst_device.upload(bytes)?;
                    let buffer = RawBuffer {
                        slice: RawSlice {
                            device,
                            scalar_type,
                            inner: RawSliceInner::Device(device_buffer),
                        },
                        cap: bytes.len(),
                    };
                    Ok(Box::pin(ready(Ok(buffer))))
                }
                #[cfg(feature = "device")]
                (DeviceInner::Device(src_device), DeviceInner::Host) => {
                    let device_buffer = self.inner.clone().unwrap_device().unwrap();
                    let host_buffer_fut = src_device.download(device_buffer)?;
                    Ok(Box::pin(async move {
                        let host_buffer = host_buffer_fut.await?;
                        Ok(RawSlice::from_bytes(scalar_type, host_buffer.read()?)
                            .to_raw_buffer()?)
                    }))
                }
                _ => unreachable!("{:?} => {:?}", self.device, device),
            }
        }
    }
}

#[doc(hidden)]
#[derive(derive_more::Deref, derive_more::DerefMut)]
pub struct RawBuffer {
    #[deref]
    #[deref_mut]
    slice: RawSlice,
    cap: usize,
}

impl RawBuffer {
    unsafe fn uninit(device: Device, scalar_type: ScalarType, len: usize) -> Result<Self> {
        match &device.inner {
            DeviceInner::Host => {
                let mut buffer = match scalar_type.size() {
                    1 => Self::from_vec(vec![0u8; len]),
                    2 => Self::from_vec(vec![0u16; len]),
                    4 => Self::from_vec(vec![0u32; len]),
                    8 => Self::from_vec(vec![0u64; len]),
                    _ => unreachable!(),
                };
                buffer.slice.scalar_type = scalar_type;
                Ok(buffer)
            }
            #[cfg(feature = "device")]
            DeviceInner::Device(device_base) => {
                let len = len * scalar_type.size();
                let cap = len;
                let buffer = unsafe { device_base.alloc(len)? };
                let inner = RawSliceInner::Device(buffer);
                Ok(Self {
                    slice: RawSlice {
                        device,
                        scalar_type,
                        inner,
                    },
                    cap,
                })
            }
        }
    }
    fn from_vec<T: Scalar>(vec: Vec<T>) -> Self {
        let device = Device::host();
        let scalar_type = T::scalar_type();
        let bytes = bytemuck::cast_slice(vec.as_slice());
        let inner = RawSliceInner::Host(HostSlice::from_bytes(bytes));
        let cap = vec.capacity() * scalar_type.size();
        core::mem::forget(vec);
        Self {
            slice: RawSlice {
                device,
                scalar_type,
                inner,
            },
            cap,
        }
    }
    fn into_device(
        self,
        device: Device,
    ) -> Result<PinBox<dyn BlockableFuture<Output = Result<Self>>>> {
        if device == self.device {
            Ok(Box::pin(ready(Ok(self))))
        } else {
            self.slice.to_device(device)
        }
    }
    fn into_vec<T: Scalar>(mut self) -> Result<Vec<T>, SliceOnDeviceError> {
        assert_eq!(T::scalar_type(), self.scalar_type);
        match &mut self.inner {
            RawSliceInner::Host(slice) => {
                let slice = std::mem::replace(slice, HostSlice::from_bytes(&[]));
                let width = self.scalar_type.size();
                let len = slice.len / width;
                let cap = self.cap / width;
                let ptr = slice.ptr as *mut T;
                self.cap = 0;
                Ok(unsafe { Vec::from_raw_parts(ptr as *mut T, len, cap) })
            }
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => {
                let index = self.device.as_device().unwrap().index();
                Err(SliceOnDeviceError::new(index))
            }
        }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        if let RawSliceInner::Host(slice) = &self.inner {
            let width = self.scalar_type.size();
            let ptr = slice.ptr;
            let len = slice.len / width;
            let cap = self.cap / width;
            match width {
                1 => unsafe {
                    Vec::from_raw_parts(ptr as *mut u8, len, cap);
                },
                2 => unsafe {
                    Vec::from_raw_parts(ptr as *mut u16, len, cap);
                },
                4 => unsafe {
                    Vec::from_raw_parts(ptr as *mut u32, len, cap);
                },
                8 => unsafe {
                    Vec::from_raw_parts(ptr as *mut u64, len, cap);
                },
                _ => unreachable!(),
            }
        }
    }
}

mod sealed {
    use super::*;

    #[doc(hidden)]
    pub trait RawData: Sized {
        #[doc(hidden)]
        fn as_raw_slice(&self) -> &RawSlice;
        #[doc(hidden)]
        fn into_raw_buffer(self) -> Result<RawBuffer> {
            self.as_raw_slice().to_raw_buffer()
        }
        #[doc(hidden)]
        fn to_arc_raw_buffer(&self) -> Result<Arc<RawBuffer>> {
            self.as_raw_slice().to_raw_buffer().map(Arc::new)
        }
        #[doc(hidden)]
        fn into_arc_raw_buffer(self) -> Result<Arc<RawBuffer>> {
            self.into_raw_buffer().map(Arc::new)
        }
        #[doc(hidden)]
        fn into_device(
            self,
            device: Device,
        ) -> Result<PinBox<dyn BlockableFuture<Output = Result<RawBuffer>>>> {
            if self.as_raw_slice().device == device {
                Ok(Box::pin(ready(self.into_raw_buffer())))
            } else {
                self.as_raw_slice().to_device(device)
            }
        }
    }

    pub trait RawDataMut: RawData {
        #[doc(hidden)]
        fn as_raw_slice_mut(&mut self) -> &mut RawSlice;
    }

    pub trait RawDataOwned: RawData {
        #[doc(hidden)]
        fn from_raw_buffer(raw: RawBuffer) -> Self;
        #[doc(hidden)]
        fn set_raw_buffer(&mut self, raw: RawBuffer);
        #[doc(hidden)]
        fn to_device_mut(
            &mut self,
            device: Device,
        ) -> Result<PinBox<dyn BlockableFuture<Output = Result<()>> + '_>> {
            if self.as_raw_slice().device == device {
                Ok(Box::pin(ready(Ok(()))))
            } else {
                let fut = self.as_raw_slice().to_device(device)?;
                Ok(Box::pin(async move {
                    self.set_raw_buffer(fut.await?);
                    Ok(())
                }))
            }
        }
    }
}
use sealed::*;

pub trait Data: RawData {
    type Elem: Scalar;
}

pub trait DataMut: Data + RawDataMut {}

pub trait DataOwned: Data + RawDataOwned {}

pub struct BufferRepr<T: Scalar> {
    raw: RawBuffer,
    _m: PhantomData<T>,
}

impl<T: Scalar> RawData for BufferRepr<T> {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
    fn into_raw_buffer(self) -> Result<RawBuffer> {
        Ok(self.raw)
    }
}

impl<T: Scalar> Data for BufferRepr<T> {
    type Elem = T;
}

impl<T: Scalar> RawDataMut for BufferRepr<T> {
    fn as_raw_slice_mut(&mut self) -> &mut RawSlice {
        &mut self.raw
    }
}

impl<T: Scalar> DataMut for BufferRepr<T> {}

impl<T: Scalar> RawDataOwned for BufferRepr<T> {
    fn from_raw_buffer(raw: RawBuffer) -> Self {
        debug_assert_eq!(raw.scalar_type, T::scalar_type());
        Self {
            raw,
            _m: PhantomData::default(),
        }
    }
    fn set_raw_buffer(&mut self, raw: RawBuffer) {
        debug_assert_eq!(raw.scalar_type, T::scalar_type());
        self.raw = raw;
    }
}

impl<T: Scalar> DataOwned for BufferRepr<T> {}

#[derive(Clone)]
pub struct SliceRepr<'a, T: Scalar> {
    raw: RawSlice,
    _m: PhantomData<&'a T>,
}

impl<T: Scalar> RawData for SliceRepr<'_, T> {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
}

impl<T: Scalar> Data for SliceRepr<'_, T> {
    type Elem = T;
}

pub struct SliceMutRepr<'a, T: Scalar> {
    raw: RawSlice,
    _m: PhantomData<&'a mut T>,
}

impl<T: Scalar> RawData for SliceMutRepr<'_, T> {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
}

impl<T: Scalar> Data for SliceMutRepr<'_, T> {
    type Elem = T;
}

impl<T: Scalar> RawDataMut for SliceMutRepr<'_, T> {
    fn as_raw_slice_mut(&mut self) -> &mut RawSlice {
        &mut self.raw
    }
}

impl<T: Scalar> DataMut for SliceMutRepr<'_, T> {}

#[derive(Clone)]
pub struct ArcBufferRepr<T: Scalar> {
    raw: Arc<RawBuffer>,
    _m: PhantomData<T>,
}

impl<T: Scalar> RawData for ArcBufferRepr<T> {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
    fn into_raw_buffer(self) -> Result<RawBuffer> {
        match Arc::try_unwrap(self.raw) {
            Ok(raw) => Ok(raw),
            Err(raw) => raw.to_raw_buffer(),
        }
    }
    fn to_arc_raw_buffer(&self) -> Result<Arc<RawBuffer>> {
        Ok(self.raw.clone())
    }
    fn into_arc_raw_buffer(self) -> Result<Arc<RawBuffer>> {
        Ok(self.raw)
    }
}

impl<T: Scalar> Data for ArcBufferRepr<T> {
    type Elem = T;
}

impl<T: Scalar> RawDataOwned for ArcBufferRepr<T> {
    fn from_raw_buffer(raw: RawBuffer) -> Self {
        debug_assert_eq!(raw.scalar_type, T::scalar_type());
        Self {
            raw: Arc::new(raw),
            _m: PhantomData::default(),
        }
    }
    fn set_raw_buffer(&mut self, raw: RawBuffer) {
        debug_assert_eq!(raw.scalar_type, T::scalar_type());
        self.raw = Arc::new(raw);
    }
}

impl<T: Scalar> DataOwned for ArcBufferRepr<T> {}

pub enum CowBufferRepr<'a, T: Scalar> {
    Buffer(BufferRepr<T>),
    Slice(SliceRepr<'a, T>),
}

impl<T: Scalar> RawData for CowBufferRepr<'_, T> {
    fn as_raw_slice(&self) -> &RawSlice {
        match self {
            Self::Buffer(buffer) => buffer.as_raw_slice(),
            Self::Slice(slice) => slice.as_raw_slice(),
        }
    }
    fn into_raw_buffer(self) -> Result<RawBuffer> {
        match self {
            Self::Buffer(buffer) => buffer.into_raw_buffer(),
            Self::Slice(slice) => slice.into_raw_buffer(),
        }
    }
}

impl<T: Scalar> Data for CowBufferRepr<'_, T> {
    type Elem = T;
}

impl<T: Scalar> RawDataOwned for CowBufferRepr<'_, T> {
    fn from_raw_buffer(raw: RawBuffer) -> Self {
        Self::Buffer(BufferRepr::from_raw_buffer(raw))
    }
    fn set_raw_buffer(&mut self, raw: RawBuffer) {
        *self = Self::Buffer(BufferRepr::from_raw_buffer(raw));
    }
}

impl<T: Scalar> DataOwned for CowBufferRepr<'_, T> {}


pub trait ScalarData: RawData {}
pub trait ScalarDataMut: ScalarData + RawDataMut {}
pub trait ScalarDataOwned: ScalarData + RawDataOwned {}

pub struct ScalarBufferRepr {
    raw: RawBuffer,
}

impl RawData for ScalarBufferRepr {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
    fn into_raw_buffer(self) -> Result<RawBuffer> {
        Ok(self.raw)
    }
}

impl ScalarData for ScalarBufferRepr {}

impl RawDataMut for ScalarBufferRepr {
    fn as_raw_slice_mut(&mut self) -> &mut RawSlice {
        &mut self.raw
    }
}

impl ScalarDataMut for ScalarBufferRepr {}

impl RawDataOwned for ScalarBufferRepr {
    fn from_raw_buffer(raw: RawBuffer) -> Self {
        Self {
            raw,
        }
    }
    fn set_raw_buffer(&mut self, raw: RawBuffer) {
        self.raw = raw;
    }
}

impl ScalarDataOwned for ScalarBufferRepr {}

#[derive(Clone)]
pub struct ScalarSliceRepr<'a> {
    raw: RawSlice,
    _m: PhantomData<&'a ()>,
}

impl RawData for ScalarSliceRepr<'_> {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
}

impl ScalarData for ScalarSliceRepr<'_> {}

pub struct ScalarSliceMutRepr<'a> {
    raw: RawSlice,
    _m: PhantomData<&'a ()>,
}

impl RawData for ScalarSliceMutRepr<'_> {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
}

impl ScalarData for ScalarSliceMutRepr<'_> {}

impl RawDataMut for ScalarSliceMutRepr<'_> {
    fn as_raw_slice_mut(&mut self) -> &mut RawSlice {
        &mut self.raw
    }
}

impl ScalarDataMut for ScalarSliceMutRepr<'_> {}

#[derive(Clone)]
pub struct ScalarArcBufferRepr {
    raw: Arc<RawBuffer>,
}

impl RawData for ScalarArcBufferRepr {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
    fn into_raw_buffer(self) -> Result<RawBuffer> {
        match Arc::try_unwrap(self.raw) {
            Ok(raw) => Ok(raw),
            Err(raw) => raw.to_raw_buffer(),
        }
    }
    fn to_arc_raw_buffer(&self) -> Result<Arc<RawBuffer>> {
        Ok(self.raw.clone())
    }
    fn into_arc_raw_buffer(self) -> Result<Arc<RawBuffer>> {
        Ok(self.raw)
    }
}

impl ScalarData for ScalarArcBufferRepr {}

impl RawDataOwned for ScalarArcBufferRepr {
    fn from_raw_buffer(raw: RawBuffer) -> Self {
        Self {
            raw: Arc::new(raw),
        }
    }
    fn set_raw_buffer(&mut self, raw: RawBuffer) {
        self.raw = Arc::new(raw);
    }
}

impl ScalarDataOwned for ScalarArcBufferRepr {}

pub enum ScalarCowBufferRepr<'a> {
    Buffer(ScalarBufferRepr),
    Slice(ScalarSliceRepr<'a>),
}

impl RawData for ScalarCowBufferRepr<'_> {
    fn as_raw_slice(&self) -> &RawSlice {
        match self {
            Self::Buffer(buffer) => buffer.as_raw_slice(),
            Self::Slice(slice) => slice.as_raw_slice(),
        }
    }
    fn into_raw_buffer(self) -> Result<RawBuffer> {
        match self {
            Self::Buffer(buffer) => buffer.into_raw_buffer(),
            Self::Slice(slice) => slice.into_raw_buffer(),
        }
    }
}

impl ScalarData for ScalarCowBufferRepr<'_> {}

impl RawDataOwned for ScalarCowBufferRepr<'_> {
    fn from_raw_buffer(raw: RawBuffer) -> Self {
        Self::Buffer(ScalarBufferRepr::from_raw_buffer(raw))
    }
    fn set_raw_buffer(&mut self, raw: RawBuffer) {
        *self = Self::Buffer(ScalarBufferRepr::from_raw_buffer(raw));
    }
}

impl ScalarDataOwned for ScalarCowBufferRepr<'_> {}

#[cfg(feature = "device")]
#[module(vulkan(1, 1))]
mod kernels {
    #[allow(unused_imports)]
    use krnl_core::kernel;

    #[kernel(threads(256), elementwise)]
    pub fn fill_u32(y: &mut u32, x: u32) {
        *y = x;
    }
}


#[derive(Clone)]
pub struct BufferBase<S: Data> {
    data: S,
}

pub type Buffer<T> = BufferBase<BufferRepr<T>>;
pub type Slice<'a, T> = BufferBase<SliceRepr<'a, T>>;
pub type SliceMut<'a, T> = BufferBase<SliceMutRepr<'a, T>>;
pub type ArcBuffer<T> = BufferBase<ArcBufferRepr<T>>;
pub type CowBuffer<'a, T> = BufferBase<CowBufferRepr<'a, T>>;

impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
    pub fn device(&self) -> &Device {
        &self.data.as_raw_slice().device
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.data.as_raw_slice().scalar_type
    }
    pub fn len(&self) -> usize {
        self.data.as_raw_slice().len() * self.scalar_type().size()
    }
    pub fn is_empty(&self) -> bool {
        self.data.as_raw_slice().is_empty()
    }
    pub fn as_slice(&self) -> Slice<T> {
        Slice {
            data: SliceRepr {
                raw: self.data.as_raw_slice().clone(),
                _m: PhantomData::default(),
            },
        }
    }
    pub fn as_host_slice(&self) -> Result<&[T], SliceOnDeviceError> {
        self.data.as_raw_slice().as_host_slice()
    }
    pub fn to_buffer(&self) -> Result<Buffer<T>> {
        self.as_slice().into_buffer()
    }
    pub fn into_buffer(self) -> Result<Buffer<T>> {
        Ok(Buffer {
            data: BufferRepr::from_raw_buffer(self.data.into_raw_buffer()?),
        })
    }
    pub fn to_arc_buffer(&self) -> Result<ArcBuffer<T>> {
        Ok(ArcBuffer {
            data: ArcBufferRepr {
                raw: self.data.to_arc_raw_buffer()?,
                _m: PhantomData::default(),
            },
        })
    }
    pub fn into_arc_buffer(self) -> Result<ArcBuffer<T>> {
        Ok(ArcBuffer {
            data: ArcBufferRepr {
                raw: self.data.into_arc_raw_buffer()?,
                _m: PhantomData::default(),
            },
        })
    }
    pub fn into_device(
        self,
        device: Device,
    ) -> Result<impl BlockableFuture<Output = Result<Buffer<T>>>> {
        let fut = self.data.into_device(device)?;
        Ok(async move {
            Ok(Buffer {
                data: BufferRepr::from_raw_buffer(fut.await?),
            })
        })
    }
    pub fn to_vec(&self) -> Result<impl BlockableFuture<Output = Result<Vec<T>>> + '_> {
        self.as_slice().into_vec()
    }
    pub fn into_vec(self) -> Result<impl BlockableFuture<Output = Result<Vec<T>>>> {
        let fut = self.data.into_device(Device::host())?;
        Ok(async move { Ok(fut.await?.into_vec().unwrap()) })
    }

    /// Divides one slice into two at an index.
    ///
    /// Equivalent to <https://doc.rust-lang.org/std/primitive.slice.html#method.split_at>.
    ///
    /// # Panics
    /// Panics if `mid` > [`.len()`](BufferBase::len).
    pub fn split_at(&self, mid: usize) -> (Slice<T>, Slice<T>) {
        let (a, b) = self.data.as_raw_slice().split_at(mid);
        let a = Slice {
            data: SliceRepr {
                raw: a,
                _m: PhantomData::default(),
            },
        };
        let b = Slice {
            data: SliceRepr {
                raw: b,
                _m: PhantomData::default(),
            },
        };
        (a, b)
    }
}

impl<T: Scalar, S: DataMut<Elem = T>> BufferBase<S> {
    pub fn as_slice_mut(&mut self) -> SliceMut<T> {
        SliceMut {
            data: SliceMutRepr {
                raw: self.data.as_raw_slice_mut().clone(),
                _m: PhantomData::default(),
            },
        }
    }
    pub fn as_host_slice_mut(&mut self) -> Result<&mut [T], SliceOnDeviceError> {
        self.data.as_raw_slice_mut().as_host_slice_mut()
    }
    /// Divides one mutable slice into two at an index.
    ///
    /// Equivalent to <https://doc.rust-lang.org/std/primitive.slice.html#method.split_at_mut>.
    ///
    /// # Panics
    /// Panics if `mid` > [`.len()`](BufferBase::len).
    pub fn split_at_mut(&mut self, mid: usize) -> (SliceMut<T>, SliceMut<T>) {
        let (a, b) = self.data.as_raw_slice_mut().split_at_mut(mid);
        let a = SliceMut {
            data: SliceMutRepr {
                raw: a,
                _m: PhantomData::default(),
            },
        };
        let b = SliceMut {
            data: SliceMutRepr {
                raw: b,
                _m: PhantomData::default(),
            },
        };
        (a, b)
    }
    pub fn fill(&mut self, elem: T) -> Result<()> {
        if !self.is_empty() {
            match self.device().kind() {
                DeviceKind::Host => {
                    for y in self.as_host_slice_mut()?.iter_mut() {
                        *y = elem;
                    }
                }
                #[cfg(feature = "device")]
                DeviceKind::Device => {
                    let kernel_info = kernels::module()?
                        .kernel_info("fill_u32")?;
                    Kernel::builder(self.device().clone(), kernel_info)
                        .build()?
                        .dispatch_builder()
                        .slice_mut("y", self.as_slice_mut())
                        .push("x", elem)
                        .build()?
                        .dispatch()?;
                }
            }
        }
        Ok(())
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> BufferBase<S> {
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self {
            data: S::from_raw_buffer(RawBuffer::from_vec(vec)),
        }
    }
    pub unsafe fn uninit(device: Device, len: usize) -> Result<Self> {
        Ok(Self {
            data: S::from_raw_buffer(unsafe { RawBuffer::uninit(device, T::scalar_type(), len)? }),
        })
    }
    pub fn from_elem(device: Device, len: usize, elem: T) -> Result<Self> {
        match device.kind() {
            DeviceKind::Host => {
                Ok(Self::from_vec(vec![elem; len]))
            }
            #[cfg(feature = "device")]
            DeviceKind::Device => {
                let mut buffer = unsafe {
                    Buffer::uninit(device, len)?
                };
                buffer.fill(elem)?;
                Ok(Self {
                    data: S::from_raw_buffer(buffer.data.raw),
                })
            }
        }
    }
    pub fn to_device_mut(
        &mut self,
        device: Device,
    ) -> Result<impl BlockableFuture<Output = Result<()>> + '_> {
        self.data.to_device_mut(device)
    }
}

#[derive(Clone)]
pub struct ScalarBufferBase<S: ScalarData> {
    data: S,
}

pub type ScalarBuffer = ScalarBufferBase<ScalarBufferRepr>;
pub type ScalarSlice<'a> = ScalarBufferBase<ScalarSliceRepr<'a>>;
pub type ScalarSliceMut<'a> = ScalarBufferBase<ScalarSliceMutRepr<'a>>;
pub type ScalarArcBuffer = ScalarBufferBase<ScalarArcBufferRepr>;
pub type ScalarCowBuffer<'a> = ScalarBufferBase<ScalarCowBufferRepr<'a>>;

/*
impl<S: ScalarDataOwned> ScalarBufferBase<S> {
    pub unsafe fn uninit(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        todo!()
    }
    pub fn from_vec<T: Scalar>(vec: Vec<T>) -> Self {
        Self {
            data: S::from_raw_buffer(RawBuffer::from_vec(vec)),
        }
    }
    pub fn from_scalar_elem(device: Device, len: usize, elem: ScalarElem) -> Result<Self> {
        match device.kind() {
            DeviceKind::Host => {
                use ScalarElem::*;
                match elem {
                    U32(x) => Ok(Self::from_vec(vec![x; len])),
                    _ => todo!(),
                }
            }
            #[cfg(feature = "device")]
            DeviceKind::Device => {
                unsafe {
                    let mut buffer = Self::uninit(device.clone(), len, elem.scalar_type())?;
                    let kernel_info = kernels::module()?
                        .kernel_info("fill_u32")?;
                    Kernel::builder(device, kernel_info)
                        .build()?
                        .dispatch_builder()
                        .global_threads(len)
                        .scalar_slice_mut("y", buffer.as_scalar_slice_mut())
                        .push("n", len as u32)
                        .push("x", elem)
                        .unsafe_()
                        .build()?
                        .dispatch()?;
                    Ok(buffer)
                }
            }
        }
    }
}*/

impl ScalarSlice<'_> {
    // for DispatchBuilder
    pub(crate) fn into_raw_slice(self) -> RawSlice {
        self.data.raw
    }
}


impl ScalarSliceMut<'_> {
    // for DispatchBuilder
    pub(crate) fn into_raw_slice_mut(self) -> RawSlice {
        self.data.raw
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for ScalarSlice<'a> {
    fn from(slice: Slice<'a, T>) -> Self {
        Self {
            data: ScalarSliceRepr {
                raw: slice.data.raw,
                _m: PhantomData::default(),
            }
        }
    }
}

impl<'a, T: Scalar> From<SliceMut<'a, T>> for ScalarSliceMut<'a> {
    fn from(slice: SliceMut<'a, T>) -> Self {
        Self {
            data: ScalarSliceMutRepr {
                raw: slice.data.raw,
                _m: PhantomData::default(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;

    #[test]
    fn buffer_from_vec() -> Result<()> {
        let x_vec = vec![1u32, 2, 3, 4];
        let buffer = Buffer::from_vec(x_vec.clone());
        let y_vec = buffer.into_vec()?.block()?;
        assert_eq!(x_vec, y_vec);
        Ok(())
    }

    #[cfg(feature = "device")]
    #[test]
    fn buffer_into_device() -> Result<()> {
        let device = Device::new(0)?;
        let x_vec = vec![1u32, 2, 3, 4];
        let buffer = Buffer::from_vec(x_vec.clone())
            .into_device(device)?
            .block()?;
        let y_vec = buffer.into_vec()?.block()?;
        assert_eq!(x_vec, y_vec);
        Ok(())
    }

    #[cfg(feature = "device")]
    #[test]
    fn buffer_zero_device() -> Result<()> {
        let device = Device::new(0)?;
        let x_vec = vec![1u32, 2, 3, 4];
        let mut buffer = Buffer::from_vec(x_vec.clone())
            .into_device(device)?
            .block()?;
        buffer.fill(0)?;
        let y_vec = buffer.into_vec()?.block()?;
        assert_eq!(&y_vec, &vec![0; y_vec.len()]);
        Ok(())
    }

}
