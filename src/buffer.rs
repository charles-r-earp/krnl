use crate::{
    device::{Device, DeviceInner, DeviceKind},
    future::BlockableFuture,
    scalar::{Scalar, ScalarElem, ScalarType},
};
#[cfg(feature = "device")]
use crate::{
    device::{DeviceBase, DeviceBuffer, HostBuffer},
    kernel::module,
    krnl_core,
};
use anyhow::{bail, format_err, Result};
use paste::paste;
use core::{marker::PhantomData, mem::size_of};
use futures_util::future::ready;
use num_traits::AsPrimitive;
use std::{pin::Pin, sync::Arc};

type PinBox<T> = Pin<Box<T>>;

pub mod error {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("The buffer is on Device({}), not the host!", index)]
    pub struct BufferOnDeviceError {
        index: usize,
    }

    impl BufferOnDeviceError {
        pub(super) fn new(index: usize) -> Self {
            Self { index }
        }
    }
}
use error::*;

#[doc(hidden)]
pub mod future {
    use super::*;
    #[cfg(feature = "device")]
    use crate::device::future::HostBufferFuture;
    use core::{
        future::Future,
        mem::replace,
        pin::Pin,
        task::{Context, Poll},
    };

    enum RawBufferIntoDeviceFutureInner {
        Ready(Option<Result<RawBuffer>>),
        #[cfg(feature = "device")]
        Future {
            scalar_type: ScalarType,
            host_buffer_fut: HostBufferFuture,
        },
    }

    impl RawBufferIntoDeviceFutureInner {
        #[cfg(feature = "device")]
        fn future_mut(&mut self) -> Option<&mut HostBufferFuture> {
            #[cfg(feature = "device")]
            if let Self::Future {
                host_buffer_fut, ..
            } = self
            {
                return Some(host_buffer_fut);
            }
            None
        }
    }

    pub struct RawBufferIntoDeviceFuture {
        inner: RawBufferIntoDeviceFutureInner,
    }

    impl RawBufferIntoDeviceFuture {
        pub(super) fn ready(buffer: Result<RawBuffer>) -> Self {
            Self {
                inner: RawBufferIntoDeviceFutureInner::Ready(Some(buffer)),
            }
        }
        #[cfg(feature = "device")]
        pub(super) fn future(scalar_type: ScalarType, host_buffer_fut: HostBufferFuture) -> Self {
            Self {
                inner: RawBufferIntoDeviceFutureInner::Future {
                    scalar_type,
                    host_buffer_fut,
                },
            }
        }
    }

    impl Future for RawBufferIntoDeviceFuture {
        type Output = Result<RawBuffer>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            use RawBufferIntoDeviceFutureInner as Inner;
            match &mut self.inner {
                Inner::Ready(buffer) => Poll::Ready(buffer.take().unwrap()),
                #[cfg(feature = "device")]
                Inner::Future { scalar_type, .. } => {
                    let scalar_type = *scalar_type;
                    let host_buffer_fut =
                        unsafe { Pin::map_unchecked_mut(self, |x| x.inner.future_mut().unwrap()) };
                    match Future::poll(host_buffer_fut, cx) {
                        Poll::Ready(host_buffer) => {
                            let buffer = host_buffer.and_then(|host_buffer| {
                                RawSlice::from_bytes(scalar_type, host_buffer.read()?)
                                    .to_raw_buffer()
                            });
                            Poll::Ready(buffer)
                        }
                        Poll::Pending => Poll::Pending,
                    }
                }
            }
        }
    }
}
use future::*;

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
    unsafe fn as_slice<T: Scalar>(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr as *const T, self.len / size_of::<T>()) }
    }
    unsafe fn as_slice_mut<T: Scalar>(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut T, self.len / size_of::<T>()) }
    }
}

#[cfg(feature = "device")]
#[derive(Clone)]
pub(crate) struct DeviceSlice {
    buffer: Option<Arc<DeviceBuffer>>,
    len: usize,
}

#[cfg(feature = "device")]
impl DeviceSlice {
    pub(crate) fn device_buffer(&self) -> Option<&Arc<DeviceBuffer>> {
        self.buffer.as_ref()
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, derive_more::Unwrap)]
enum RawSliceInner {
    Host(HostSlice),
    #[cfg(feature = "device")]
    Device(DeviceSlice),
}

impl RawSliceInner {
    fn len(&self) -> usize {
        match self {
            Self::Host(slice) => slice.len,
            #[cfg(feature = "device")]
            Self::Device(buffer) => buffer.len(),
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
    pub(crate) fn device_ref(&self) -> &Device {
        &self.device
    }
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
    fn as_host_slice<T: Scalar>(&self) -> Result<&[T], BufferOnDeviceError> {
        assert_eq!(T::scalar_type(), self.scalar_type);
        match &self.inner {
            RawSliceInner::Host(slice) => Ok(unsafe { slice.as_slice() }),
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => Err(BufferOnDeviceError::new(
                self.device.inner.clone().unwrap_device().index(),
            )),
        }
    }
    fn as_host_slice_mut<T: Scalar>(&mut self) -> Result<&mut [T], BufferOnDeviceError> {
        assert_eq!(T::scalar_type(), self.scalar_type);
        match &mut self.inner {
            RawSliceInner::Host(slice) => Ok(unsafe { slice.as_slice_mut() }),
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => Err(BufferOnDeviceError::new(
                self.device.inner.clone().unwrap_device().index(),
            )),
        }
    }
    fn to_vec<T: Scalar>(&self) -> Result<Vec<T>, BufferOnDeviceError> {
        Ok(self.as_host_slice()?.to_vec())
    }
    fn to_raw_buffer(&self) -> Result<RawBuffer> {
        if self.is_empty() {
            return Ok(RawBuffer::new(self.device.clone(), self.scalar_type));
        }
        match &self.inner {
            RawSliceInner::Host(host_slice) => {
                let mut buffer = match self.scalar_type.size() {
                    1 => RawBuffer::from_vec(unsafe { host_slice.as_slice::<u8>() }.to_vec()),
                    2 => RawBuffer::from_vec(unsafe { host_slice.as_slice::<u16>() }.to_vec()),
                    4 => RawBuffer::from_vec(unsafe { host_slice.as_slice::<u32>() }.to_vec()),
                    8 => RawBuffer::from_vec(unsafe { host_slice.as_slice::<u64>() }.to_vec()),
                    _ => unreachable!(),
                };
                buffer.scalar_type = self.scalar_type;
                Ok(buffer)
            }
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => {
                use ScalarType::*;
                let device = &self.device;
                let cast_u8 = buffer_cast::cast_u8_u8::build(device.clone());
                let cast_u16 = buffer_cast::cast_u16_u16::build(device.clone());
                let cast_u32 = buffer_cast::cast_u32_u32::build(device.clone());
                let cast_u64 = buffer_cast::cast_u64_u64::build(device.clone());
                let n = self.inner.len();
                let scalar_type = if n % 8 == 0 && cast_u64.is_ok() {
                    U64
                } else if n % 4 == 0 && cast_u32.is_ok() {
                    U32
                } else if n % 2 == 0 && cast_u16.is_ok() {
                    U16
                } else {
                    U8
                };
                let x = ScalarSlice {
                    data: ScalarSliceRepr::from_raw(self.clone().bitcast(scalar_type)),
                };
                let mut output = unsafe {
                    ScalarBuffer::uninit(device.clone(), self.len(), self.scalar_type)?
                };
                let mut y = output.bitcast_mut(scalar_type);
                match scalar_type {
                    U8 => {
                        cast_u8?.dispatch(x.try_as_slice().unwrap(), y.try_as_slice_mut().unwrap())?;
                    }
                    U16 => {
                        cast_u16?.dispatch(x.try_as_slice().unwrap(), y.try_as_slice_mut().unwrap())?;
                    }
                    U32 => {
                        cast_u32?.dispatch(x.try_as_slice().unwrap(), y.try_as_slice_mut().unwrap())?;
                    }
                    U64 => {
                        cast_u64?.dispatch(x.try_as_slice().unwrap(), y.try_as_slice_mut().unwrap())?;
                    }
                    _ => unreachable!(),
                }
                Ok(output.data.raw)
            }
        }
    }
    #[cfg(feature = "device")]
    pub(crate) fn as_device_slice(&self) -> Option<&DeviceSlice> {
        if let RawSliceInner::Device(device_slice) = &self.inner {
            Some(device_slice)
        } else {
            None
        }
    }
    fn bitcast(mut self, scalar_type: ScalarType) -> Self {
        self.scalar_type = scalar_type;
        self
    }
    fn to_device(&self, device: Device) -> Result<RawBufferIntoDeviceFuture> {
        if self.is_empty() {
            return Ok(RawBufferIntoDeviceFuture::ready(Ok(RawBuffer::new(self.device.clone(), self.scalar_type))));
        }
        #[cfg(feature = "device")]
        let scalar_type = self.scalar_type;
        match (&self.inner, &device.inner) {
            (RawSliceInner::Host(_), DeviceInner::Host) => {
                Ok(RawBufferIntoDeviceFuture::ready(self.to_raw_buffer()))
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Device(_), DeviceInner::Device(_)) => {
                Ok(RawBufferIntoDeviceFuture::ready(self.to_raw_buffer()))
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Host(host_slice), DeviceInner::Device(dst_device)) => {
                let bytes = unsafe { host_slice.as_slice() };
                let device_buffer = dst_device.upload(bytes)?;
                let device_slice = DeviceSlice {
                    buffer: device_buffer,
                    len: bytes.len(),
                };
                let buffer = RawBuffer {
                    slice: RawSlice {
                        device,
                        scalar_type,
                        inner: RawSliceInner::Device(device_slice),
                    },
                    cap: bytes.len(),
                };
                Ok(RawBufferIntoDeviceFuture::ready(Ok(buffer)))
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Device(device_slice), DeviceInner::Host) => {
                let dst_device = self.device.inner.device().unwrap();
                let device_buffer = device_slice.device_buffer().unwrap();
                let host_buffer_fut = dst_device.download(device_buffer.clone())?;
                Ok(RawBufferIntoDeviceFuture::future(
                    scalar_type,
                    host_buffer_fut,
                ))
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
                let device_slice = DeviceSlice { buffer, len };
                let inner = RawSliceInner::Device(device_slice);
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
    fn new(device: Device, scalar_type: ScalarType) -> Self {
        unsafe {
            Self::uninit(device, scalar_type, 0).unwrap()
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
    fn into_device(self, device: Device) -> Result<RawBufferIntoDeviceFuture> {
        if device == self.device {
            Ok(RawBufferIntoDeviceFuture::ready(Ok(self)))
        } else {
            self.slice.to_device(device)
        }
    }
    fn into_vec<T: Scalar>(mut self) -> Result<Vec<T>, BufferOnDeviceError> {
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
                Err(BufferOnDeviceError::new(index))
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
        fn try_into_raw_buffer(self) -> Result<RawBuffer, Self> {
            Err(self)
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
        fn into_device(self, device: Device) -> Result<RawBufferIntoDeviceFuture> {
            if self.as_raw_slice().device == device {
                Ok(RawBufferIntoDeviceFuture::ready(self.into_raw_buffer()))
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

impl<T: Scalar> BufferRepr<T> {
    unsafe fn from_raw_unchecked(raw: RawBuffer) -> Self {
        debug_assert_eq!(raw.scalar_type, T::scalar_type());
        Self {
            raw,
            _m: PhantomData::default(),
        }
    }
}

impl<T: Scalar> RawData for BufferRepr<T> {
    fn as_raw_slice(&self) -> &RawSlice {
        &self.raw
    }
    fn into_raw_buffer(self) -> Result<RawBuffer> {
        Ok(self.raw)
    }
    fn try_into_raw_buffer(self) -> Result<RawBuffer, Self> {
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
        assert_eq!(raw.scalar_type, T::scalar_type());
        unsafe { Self::from_raw_unchecked(raw) }
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

impl<T: Scalar> SliceRepr<'_, T> {
    /*unsafe fn from_raw_unchecked(raw: RawSlice) -> Self {
        debug_assert_eq!(raw.scalar_type, T::scalar_type());
        Self {
            raw,
            _m: PhantomData::default(),
        }
    }*/
    fn from_raw_ref(raw: &RawSlice) -> Option<Self> {
        if T::scalar_type() == raw.scalar_type {
            Some(Self {
                raw: raw.clone(),
                _m: PhantomData::default(),
            })
        } else {
            None
        }
    }
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

impl<T: Scalar> SliceMutRepr<'_, T> {
    unsafe fn from_raw_unchecked(raw: RawSlice) -> Self {
        debug_assert_eq!(raw.scalar_type, T::scalar_type());
        Self {
            raw,
            _m: PhantomData::default(),
        }
    }
    fn from_raw_mut(raw: &mut RawSlice) -> Option<Self> {
        if T::scalar_type() == raw.scalar_type {
            Some(Self {
                raw: raw.clone(),
                _m: PhantomData::default(),
            })
        } else {
            None
        }
    }
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
    fn try_into_raw_buffer(self) -> Result<RawBuffer, Self> {
        match Arc::try_unwrap(self.raw) {
            Ok(raw) => Ok(raw),
            Err(raw) => Err(Self {
                raw,
                _m: PhantomData::default(),
            }),
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
    fn try_into_raw_buffer(self) -> Result<RawBuffer, Self> {
        match self {
            Self::Buffer(buffer) => Ok(buffer.raw),
            Self::Slice(slice) => Err(Self::Slice(slice)),
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
    fn try_into_raw_buffer(self) -> Result<RawBuffer, Self> {
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
        Self { raw }
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

impl<'a> ScalarSliceRepr<'a> {
    fn from_raw_ref(raw: &'a RawSlice) -> Self {
        Self::from_raw(raw.clone())
    }
    fn from_raw(raw: RawSlice) -> Self {
        Self {
            raw,
            _m: PhantomData::default(),
        }
    }
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

impl<'a> ScalarSliceMutRepr<'a> {
    fn from_raw_mut(raw: &'a mut RawSlice) -> Self {
        Self::from_raw(raw.clone())
    }
    fn from_raw(raw: RawSlice) -> Self {
        Self {
            raw,
            _m: PhantomData::default(),
        }
    }
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
    fn try_into_raw_buffer(self) -> Result<RawBuffer, Self> {
        match Arc::try_unwrap(self.raw) {
            Ok(raw) => Ok(raw),
            Err(raw) => Err(Self { raw }),
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
        Self { raw: Arc::new(raw) }
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
    fn try_into_raw_buffer(self) -> Result<RawBuffer, Self> {
        match self {
            Self::Buffer(buffer) => Ok(buffer.raw),
            Self::Slice(slice) => Err(Self::Slice(slice)),
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
#[module]
#[krnl(crate = crate)]
mod buffer_fill {
    use krnl_core::kernel;

    #[kernel(
        vulkan(1, 2),
        threads(256),
        elementwise,
        capabilities("Int8", "StorageBuffer8BitAccess")
    )]
    pub fn fill_u8(y: &mut u8, x: u8) {
        *y = x;
    }
    #[kernel(
        vulkan(1, 1),
        threads(256),
        elementwise,
        capabilities("Int8", "Int16", "StorageBuffer16BitAccess")
    )]
    pub fn fill_u16(y: &mut u16, x: u16) {
        *y = x;
    }
    #[kernel(vulkan(1, 1), threads(256), elementwise)]
    pub fn fill_u32(y: &mut u32, x: u32) {
        *y = x;
    }
    #[kernel(vulkan(1, 1), threads(256), elementwise, capabilities("Int64"))]
    pub fn fill_u64(y: &mut u64, x: u64) {
        *y = x;
    }
}

#[cfg(feature = "device")]
#[module(
    dependency("krnl-core", path = "krnl-core" , features = ["half"]),
    dependency("paste", version = "1.0.7"),
)]
#[krnl(crate = crate)]
mod buffer_cast {
    #[allow(unused_imports)]
    use krnl_core::{kernel, scalar::Scalar};
    #[allow(unused_imports)]
    #[cfg(any(feature = "half", krnl_device_crate))]
    use krnl_core::half::{f16, bf16};
    #[allow(unused_imports)]
    use paste::paste;

    macro_rules! impl_cast {
        (@ForY features($($features:literal),*), vulkan(1, $minor:literal), capabilities($($caps:literal),*), $x:ty =>) => ();
        (@ForY features($($features:literal),*), vulkan(1, $minor:literal), capabilities($($caps:literal),*), $x:ty => $y:ty $(,$ys:ty)*) => (
            paste! {
                $(#[cfg(any(krnl_device_crate, feature = $features))])*
                #[kernel(vulkan(1, $minor), threads(256), elementwise, capabilities($($caps),*))]
                pub fn [<cast_ $x _ $y>](x: &$x, y: &mut $y) {
                    *y = x.cast();
                }
            }
            impl_cast!(@ForY features($($features),*), vulkan(1, $minor), capabilities($($caps),*), $x => $($ys),*);
        );
        (features($($features:literal),*), vulkan(1, $minor:literal), capabilities($($caps:literal),*), => $($ys:ty),*) => ();
        (features($($features:literal),*), vulkan(1, $minor:literal), capabilities($($caps:literal),*), $x:ty $(,$xs:ty)* => $($ys:ty),*) => (
            impl_cast!(@ForY features($($features),*), vulkan(1, $minor), capabilities($($caps),*), $x => $($ys),*);
            impl_cast!(features($($features),*), vulkan(1, $minor), capabilities($($caps),*), $($xs),* => $($ys),*);
        );
    }

    impl_cast!(features(), vulkan(1, 1), capabilities("Int8"), u8, i8 => u8, i8, u32, i32, f32);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int8"), u32, i32, f32 => u8, i8);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int8", "Int16"), u8, i8 => u16, i16);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int8", "Int64"), u8, i8 => u64, i64);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int8", "Int64", "Float64"), u8, i8 => f64);
    impl_cast!(features(), vulkan(1, 2), capabilities("Int16"), u16, i16 => u16, i16, u32, i32, f32);
    impl_cast!(features(), vulkan(1, 2), capabilities("Int16"), u32, i32, f32 => u16, i16);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int8", "Int16"), u16, i16 => u8, i8);
    impl_cast!(features(), vulkan(1, 2), capabilities("Int16", "Int64"), u16, i16 => u64, i64);
    impl_cast!(features(), vulkan(1, 2), capabilities("Int16", "Int64", "Float64"), u16, i16 => f64);
    impl_cast!(features("half"), vulkan(1, 2), capabilities("Int8", "Int16"), f16, bf16 => u8, i8, u16, i16, f16, bf16, u32, i32, f32);
    impl_cast!(features("half"), vulkan(1, 2), capabilities("Int8", "Int16"), u8, i8, u16, i16, u32, i32, f32 => f16, bf16);
    impl_cast!(features("half"), vulkan(1, 2), capabilities("Int8", "Int16", "Int64"), f16, bf16 => u64, i64);
    impl_cast!(features("half"), vulkan(1, 2), capabilities("Int8", "Int16", "Int64", "Float64"), f16, bf16 => f64);
    impl_cast!(features(), vulkan(1, 1), capabilities(), u32, i32, f32 => u32, i32, f32);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int8", "Int64"), u64, i64 => u8, i8);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int16", "Int64"), u64, i64 => u16, i16);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int64"), u64, i64 => u32, i32, f32, u64, i64);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int64"), u32, i32, f32 => u64, i64);
    impl_cast!(features("half"), vulkan(1, 1), capabilities("Int8", "Int16", "Int64"), u64, i64 => f16, bf16);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int8", "Int64", "Float64"), f64 => u8, i8);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int16", "Int64", "Float64"), f64 => u16, i16);
    impl_cast!(features("half"), vulkan(1, 1), capabilities("Int8", "Int16", "Int64", "Float64"), f64 => f16, bf16);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int64", "Float64"), f64 => u32, i32, f32, u64, i64, f64);
    impl_cast!(features(), vulkan(1, 1), capabilities("Int64", "Float64"), u32, i32, f32, u64, i64 => f64);
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
    pub fn device(&self) -> Device {
        self.data.as_raw_slice().device.clone()
    }
    pub fn scalar_type(&self) -> ScalarType {
        debug_assert_eq!(self.data.as_raw_slice().scalar_type, T::scalar_type());
        T::scalar_type()
    }
    pub fn len(&self) -> usize {
        self.data.as_raw_slice().len()
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
    pub fn as_scalar_slice(&self) -> ScalarSlice {
        self.as_slice().into()
    }
    pub fn try_as_slice<T2: Scalar>(&self) -> Option<Slice<T2>> {
        if let Some(data) = SliceRepr::from_raw_ref(self.data.as_raw_slice()) {
            Some(Slice { data })
        } else {
            None
        }
    }
    pub fn as_host_slice(&self) -> Result<&[T], BufferOnDeviceError> {
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
    pub fn cast<T2: Scalar>(&self) -> Result<CowBuffer<T2>> {
        if let Some(slice) = self.try_as_slice::<T2>() {
            Ok(CowBuffer {
                data: CowBufferRepr::Slice(slice.data),
            })
        } else if let Ok(slice) = self.as_host_slice() {
            let vec = slice.iter().map(|x| x.cast()).collect();
            Ok(CowBuffer::from_vec(vec))
        } else {
            Ok(Buffer::try_from(self.as_scalar_slice().cast(T2::scalar_type())?.into_scalar_buffer()?)
                .ok()
                .unwrap()
                .into())
        }
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
    pub fn try_as_slice_mut<T2: Scalar>(&mut self) -> Option<SliceMut<T2>> {
        if let Some(data) = SliceMutRepr::from_raw_mut(self.data.as_raw_slice_mut()) {
            Some(SliceMut { data })
        } else {
            None
        }
    }
    pub fn as_scalar_slice_mut(&mut self) -> ScalarSliceMut {
        ScalarSliceMut {
            data: ScalarSliceMutRepr::from_raw_mut(self.data.as_raw_slice_mut()),
        }
    }
    pub fn as_host_slice_mut(&mut self) -> Result<&mut [T], BufferOnDeviceError> {
        self.data.as_raw_slice_mut().as_host_slice_mut()
    }
    pub fn bitcast_mut<T2: Scalar>(&mut self) -> SliceMut<T2> {
        let raw_slice = self.data.as_raw_slice().clone().bitcast(T2::scalar_type());
        let data = unsafe { SliceMutRepr::from_raw_unchecked(raw_slice) };
        SliceMut { data }
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
                    self.as_scalar_slice_mut().fill(elem.into())?;
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
            DeviceKind::Host => Ok(Self::from_vec(vec![elem; len])),
            #[cfg(feature = "device")]
            DeviceKind::Device => {
                let mut buffer = unsafe { Buffer::uninit(device, len)? };
                buffer.fill(elem)?;
                Ok(Self {
                    data: S::from_raw_buffer(buffer.data.raw),
                })
            }
        }
    }
    pub fn zeros(device: Device, len: usize) -> Result<Self> {
        Self::from_elem(device, len, T::zero())
    }
    pub fn ones(device: Device, len: usize) -> Result<Self> {
        Self::from_elem(device, len, T::one())
    }
    pub fn to_device_mut(
        &mut self,
        device: Device,
    ) -> Result<impl BlockableFuture<Output = Result<()>> + '_> {
        self.data.to_device_mut(device)
    }
}

impl<T: Scalar> From<Buffer<T>> for CowBuffer<'_, T> {
    fn from(buffer: Buffer<T>) -> Self {
        Self {
            data: CowBufferRepr::Buffer(buffer.data),
        }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for CowBuffer<'a, T> {
    fn from(slice: Slice<'a, T>) -> Self {
        Self {
            data: CowBufferRepr::Slice(slice.data),
        }
    }
}

impl<T: Scalar> TryFrom<ScalarBuffer> for Buffer<T> {
    type Error = ScalarBuffer;
    fn try_from(buffer: ScalarBuffer) -> Result<Self, Self::Error> {
        if buffer.scalar_type() == T::scalar_type() {
            Ok(Self {
                data: unsafe {
                    BufferRepr::from_raw_unchecked(buffer.data.raw)
                },
            })
        } else {
            Err(buffer)
        }
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

impl<S: ScalarData> ScalarBufferBase<S> {
    pub fn device(&self) -> Device {
        self.data.as_raw_slice().device.clone()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.data.as_raw_slice().scalar_type
    }
    pub fn len(&self) -> usize {
        self.data.as_raw_slice().len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.as_raw_slice().is_empty()
    }
    pub fn as_scalar_slice(&self) -> ScalarSlice {
        ScalarSlice {
            data: ScalarSliceRepr::from_raw_ref(self.data.as_raw_slice()),
        }
    }
    pub fn try_as_slice<T2: Scalar>(&self) -> Option<Slice<T2>> {
        if let Some(data) = SliceRepr::from_raw_ref(self.data.as_raw_slice()) {
            Some(Slice { data })
        } else {
            None
        }
    }
    pub fn to_buffer(&self) -> Result<ScalarBuffer> {
        self.as_scalar_slice().into_scalar_buffer()
    }
    pub fn into_scalar_buffer(self) -> Result<ScalarBuffer> {
        Ok(ScalarBuffer {
            data: ScalarBufferRepr::from_raw_buffer(self.data.into_raw_buffer()?),
        })
    }
    pub fn to_scalar_arc_buffer(&self) -> Result<ScalarArcBuffer> {
        Ok(ScalarArcBuffer {
            data: ScalarArcBufferRepr {
                raw: self.data.to_arc_raw_buffer()?,
            },
        })
    }
    pub fn cast(&self, scalar_type: ScalarType) -> Result<ScalarCowBuffer> {
        use ScalarType::*;
        let x_ty = self.scalar_type();
        let y_ty = scalar_type;
        if x_ty == y_ty {
            return Ok(self.as_scalar_slice().into());
        }
        let device = self.device();
        macro_rules! cast {
            (=> $($y:ty),*) => ();
            ($x:ty $(,$xs:ty)* => $($y:ty),*) => (
                paste! {
                    $(
                        if x_ty == [<$x:upper>] && y_ty == [<$y:upper>] {
                            let x = self.try_as_slice::<$x>().unwrap();
                            if device.is_host() {
                                return Ok(ScalarBuffer::from(x.cast::<$y>()?.into_buffer()?).into());
                            }
                            #[cfg(feature = "device")] {
                                let mut y = unsafe {
                                    Buffer::<$y>::uninit(device.clone(), x.len())?
                                };
                                buffer_cast::[<cast_ $x _ $y>]::build(device)?.dispatch(x, y.as_slice_mut())?;
                                return Ok(ScalarBuffer::from(y).into());
                            }
                            unreachable!()
                        }
                    )*
                }
                cast!($($xs),* => $($y),*);
            );
        }
        cast!(u8, i8, u16, i16, u32, i32, f32, u64, i64, f64 => u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
        #[cfg(feature = "half")] {
            use half::{f16, bf16};
            cast!(f16, bf16 => f16, bf16, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
            cast!(u8, i8, u16, i16, u32, i32, f32, u64, i64, f64 => f16, bf16);
        }
        todo!()
    }
}

impl<S: ScalarDataOwned> ScalarBufferBase<S> {
    pub unsafe fn uninit(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Ok(Self {
            data: S::from_raw_buffer(unsafe { RawBuffer::uninit(device, scalar_type, len)? }),
        })
    }
    pub fn from_vec<T: Scalar>(vec: Vec<T>) -> Self {
        Self {
            data: S::from_raw_buffer(RawBuffer::from_vec(vec)),
        }
    }
    pub fn from_scalar_elem(device: Device, len: usize, elem: ScalarElem) -> Result<Self> {
        let mut buffer = unsafe { ScalarBuffer::uninit(device, len, elem.scalar_type())? };
        buffer.fill(elem)?;
        Ok(Self {
            data: S::from_raw_buffer(buffer.data.raw),
        })
    }
    pub fn zeros(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Self::from_scalar_elem(device, len, ScalarElem::zero(scalar_type))
    }
    pub fn ones(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Self::from_scalar_elem(device, len, ScalarElem::zero(scalar_type))
    }
}

impl<S: ScalarDataMut> ScalarBufferBase<S> {
    pub fn try_as_slice_mut<T2: Scalar>(&mut self) -> Option<SliceMut<T2>> {
        if let Some(data) = SliceMutRepr::from_raw_mut(self.data.as_raw_slice_mut()) {
            Some(SliceMut { data })
        } else {
            None
        }
    }
    pub fn bitcast_mut(&mut self, scalar_type: ScalarType) -> ScalarSliceMut {
        let raw_slice = self.data.as_raw_slice_mut().clone().bitcast(scalar_type);
        let data = ScalarSliceMutRepr::from_raw(raw_slice);
        ScalarSliceMut { data }
    }
    pub fn fill(&mut self, elem: ScalarElem) -> Result<()> {
        use bytemuck::cast;
        use ScalarElem as E;
        use ScalarType::*;
        if self.scalar_type() != elem.scalar_type() {
            bail!(
                "ScalarBuffer::fill: expected `elem` with scalar type {:?}, found {:?}!",
                self.scalar_type(),
                elem.scalar_type()
            );
        }
        if !self.is_empty() {
            let x = elem.to_scalar_bits();
            let y = self;
            match y.device().kind() {
                DeviceKind::Host => {
                    let mut y = y.bitcast_mut(x.scalar_type());
                    match x {
                        E::U8(x) => y.try_as_slice_mut().unwrap().fill(x),
                        E::U16(x) => y.try_as_slice_mut().unwrap().fill(x),
                        E::U32(x) => y.try_as_slice_mut().unwrap().fill(x),
                        E::U64(x) => y.try_as_slice_mut().unwrap().fill(x),
                        _ => unreachable!(),
                    }
                },
                #[cfg(feature = "device")]
                DeviceKind::Device => {
                    use spirv::Capability;
                    let device = y.device();
                    let device_base = device.inner.device().unwrap();
                    let n = y.len();
                    let x = if n % 8 == 0 && device_base.capability_enabled(Capability::Int64) {
                        match x {
                            E::U8(x) => E::U64(u64::from_ne_bytes([x; 8])),
                            E::U16(x) => {
                                let [x1, x2] = x.to_ne_bytes();
                                E::U64(u64::from_ne_bytes([x1, x2, x1, x2, x1, x2, x1, x2]))
                            }
                            E::U32(x) => {
                                let [x1, x2, x3, x4] = x.to_ne_bytes();
                                E::U64(u64::from_ne_bytes([x1, x2, x3, x4, x1, x2, x3, x4]))
                            }
                            x => x,
                        }
                    } else if n % 4 == 0 {
                        match x {
                            E::U8(x) => E::U32(u32::from_ne_bytes([x; 4])),
                            E::U16(x) => {
                                let [x1, x2] = x.to_ne_bytes();
                                E::U32(u32::from_ne_bytes([x1, x2, x1, x2]))
                            }
                            x => x,
                        }
                    } else if n % 2 == 0 && device_base.capability_enabled(Capability::Int16) && device_base.capability_enabled(Capability::StorageBuffer16BitAccess) {
                        match x {
                            E::U8(x) => E::U16(u16::from_ne_bytes([x; 2])),
                            x => x,
                        }
                    } else {
                        x
                    };
                    let mut y = y.bitcast_mut(x.scalar_type());
                    match x {
                        E::U8(x) => buffer_fill::fill_u8::build(device)?.dispatch(y.try_as_slice_mut().unwrap(), x),
                        E::U16(x) => buffer_fill::fill_u16::build(device)?.dispatch(y.try_as_slice_mut().unwrap(), x),
                        E::U32(x) => buffer_fill::fill_u32::build(device)?.dispatch(y.try_as_slice_mut().unwrap(), x),
                        E::U64(x) => buffer_fill::fill_u64::build(device)?.dispatch(y.try_as_slice_mut().unwrap(), x),
                        _ => unreachable!(),
                    }
                }
            }
        } else {
            Ok(())
        }
    }
}

impl ScalarSlice<'_> {
    // for kernel::__dispatch
    pub(crate) fn into_raw_slice(self) -> RawSlice {
        self.data.raw
    }
}

impl<T: Scalar> From<Buffer<T>> for ScalarBuffer {
    fn from(buffer: Buffer<T>) -> Self {
        Self {
            data: ScalarBufferRepr {
                raw: buffer.data.raw,
            }
        }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for ScalarSlice<'a> {
    fn from(slice: Slice<'a, T>) -> Self {
        Self {
            data: ScalarSliceRepr {
                raw: slice.data.raw,
                _m: PhantomData::default(),
            },
        }
    }
}

impl<'a, T: Scalar> From<SliceMut<'a, T>> for ScalarSliceMut<'a> {
    fn from(slice: SliceMut<'a, T>) -> Self {
        Self {
            data: ScalarSliceMutRepr {
                raw: slice.data.raw,
                _m: PhantomData::default(),
            },
        }
    }
}

impl From<ScalarBuffer> for ScalarCowBuffer<'_> {
    fn from(buffer: ScalarBuffer) -> Self {
        Self {
            data: ScalarCowBufferRepr::Buffer(buffer.data),
        }
    }
}

impl<'a> From<ScalarSlice<'a>> for ScalarCowBuffer<'a> {
    fn from(slice: ScalarSlice<'a>) -> Self {
        Self {
            data: ScalarCowBufferRepr::Slice(slice.data),
        }
    }
}

impl<'a, T: Scalar> From<CowBuffer<'a, T>> for ScalarCowBuffer<'a> {
    fn from(buffer: CowBuffer<'a, T>) -> Self {
        let data = match buffer.data {
            CowBufferRepr::Buffer(buffer) => ScalarCowBufferRepr::Buffer(ScalarBufferRepr { raw: buffer.raw }),
            CowBufferRepr::Slice(slice) => ScalarCowBufferRepr::Slice(ScalarSliceRepr::from_raw(slice.raw)),
        };
        Self {
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;
    #[cfg(feature = "half")]
    use half::{bf16, f16};
    use paste::paste;

    fn buffer_test_lengths() -> impl IntoIterator<Item = usize> {
        [0, 1, 3, 4, 16, 67, 300, 1011]
    }

    #[test]
    fn buffer_from_vec() -> Result<()> {
        for n in buffer_test_lengths() {
            let x_vec = (1..=n as u32).into_iter().collect::<Vec<_>>();
            let buffer = Buffer::from_vec(x_vec.clone());
            assert_eq!(buffer.len(), n);
            let y_vec = buffer.into_vec()?.block()?;
            assert_eq!(x_vec, y_vec);
        }
        Ok(())
    }

    #[cfg(feature = "device")]
    #[test]
    fn buffer_into_device() -> Result<()> {
        let device = Device::new(0)?;
        for n in buffer_test_lengths() {
            let x_vec = (1..=n as u32).into_iter().collect::<Vec<_>>();
            let buffer = Buffer::from_vec(x_vec.clone())
                .into_device(device.clone())?
                .block()?;
            let y_vec = buffer.into_vec()?.block()?;
            assert_eq!(x_vec, y_vec);
        }
        Ok(())
    }

    fn buffer_fill<T: Scalar>(device: Device) -> Result<()> {
        let elem = T::one();
        for n in buffer_test_lengths() {
            let x_vec = (10..20)
                .cycle()
                .map(|x| T::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<_>>();
            let y_vec_true = vec![elem; n];
            let mut buffer = Buffer::from_vec(x_vec.clone())
                .into_device(device.clone())?
                .block()?;

            buffer.fill(elem)?;
            let y_vec: Vec<T> = buffer.into_vec()?.block()?;
            assert_eq!(y_vec, y_vec_true);
        }
        Ok(())
    }

    fn scalar_buffer_fill<T: Scalar>(device: Device) -> Result<()> {
        let elem = T::one();
        for n in buffer_test_lengths() {
            let x_vec = (10..20)
                .cycle()
                .map(|x| T::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<_>>();
            let y_vec_true = vec![elem; n];
            let mut buffer = Buffer::from_vec(x_vec)
                .into_device(device.clone())?
                .block()?;
            buffer.as_scalar_slice_mut().fill(elem.into())?;
            let y_vec: Vec<T> = buffer.into_vec()?.block()?;
            assert_eq!(y_vec, y_vec_true);
        }
        Ok(())
    }

    macro_rules! impl_buffer_fill_tests {
        (@Host f16 => $func:ident $block:block) => (
            #[cfg(feature = "half")]
            impl_buffer_fill_tests!(@Host => $func $block);
        );
        (@Host bf16 => $func:ident $block:block) => (
            #[cfg(feature = "half")]
            impl_buffer_fill_tests!(@Host => $func $block);
        );
        (@Host $($t:ty)? => $func:ident $block:block) => (
            #[test]
            fn $func() -> Result<()> $block
        );
        (@Device f16 => $func:ident $block:block) => (
            #[cfg(feature = "half")]
            impl_buffer_fill_tests!(@Device => $func $block);
        );
        (@Device bf16 => $func:ident $block:block) => (
            #[cfg(feature = "half")]
            impl_buffer_fill_tests!(@Device => $func $block);
        );
        (@Device u32 => $func:ident $block:block) => (
            #[cfg(feature = "device")]
            #[test]
            fn $func() -> Result<()> $block
        );
        (@Device i32 => $func:ident $block:block) => (
            impl_buffer_fill_tests!(@Device u32 => $func $block);
        );
        (@Device f32 => $func:ident $block:block) => (
            impl_buffer_fill_tests!(@Device u32 => $func $block);
        );
        (@Device $($t:ty)? => $func:ident $block:block) => (
            #[cfg(feature = "device")]
            #[ignore]
            #[test]
            fn $func() -> Result<()> $block
        );
        ($($t:ty),*) => (
            paste! {
                $(
                    impl_buffer_fill_tests!(@Host $t => [<buffer_fill_host_ $t>] {
                        buffer_fill::<$t>(Device::host())
                    });
                    impl_buffer_fill_tests!(@Host $t => [<scalar_buffer_fill_host_ $t>] {
                        scalar_buffer_fill::<$t>(Device::host())
                    });
                    impl_buffer_fill_tests!(@Device $t => [<buffer_fill_device_ $t>] {
                        buffer_fill::<$t>(Device::new(0)?)
                    });
                    impl_buffer_fill_tests!(@Device $t => [<scalar_buffer_fill_device_ $t>] {
                        scalar_buffer_fill::<$t>(Device::new(0)?)
                    });
                )*
            }
        );
    }
    impl_buffer_fill_tests!(u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64);

    fn buffer_cast<X: Scalar, Y: Scalar>(device: Device) -> Result<()> {
        for n in buffer_test_lengths() {
            let x_vec = (10u32..20)
                .cycle()
                .map(|x| X::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<X>>();
            let y_vec_true = x_vec.iter().map(|x| x.cast()).collect::<Vec<Y>>();
            let buffer = Buffer::from_vec(x_vec)
                .into_device(device.clone())?
                .block()?;
            let buffer = buffer.cast()?;
            let y_vec: Vec<Y> = buffer.into_vec()?.block()?;
            assert_eq!(&y_vec, &y_vec_true);
        }
        Ok(())
    }

    fn scalar_buffer_cast<X: Scalar, Y: Scalar>(device: Device) -> Result<()> {
        for n in buffer_test_lengths() {
            let x_vec = (10u32..20)
                .cycle()
                .map(|x| X::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<X>>();
            let y_vec_true = x_vec.iter().map(|x| x.cast()).collect::<Vec<Y>>();
            let buffer = Buffer::from_vec(x_vec)
                .into_device(device.clone())?
                .block()?;
            let buffer = buffer.as_scalar_slice().cast(Y::scalar_type())?.into_scalar_buffer()?;
            let y_vec: Vec<Y> = buffer.try_as_slice().unwrap().into_vec()?.block()?;
            assert_eq!(&y_vec, &y_vec_true);
        }
        Ok(())
    }

    macro_rules! impl_buffer_cast_tests {
        (@ForY features($($features:literal),*), $(@ignore($ignore:tt),)? $x:ty =>) => ();
        (@ForY features($($features:literal),*), $(@ignore($ignore:tt),)? $x:ty => $y:ty $(,$ys:ty)*) => (
            paste! {
                $(#[cfg(feature = $features)])*
                #[test]
                fn [<buffer_cast_ $x _ $y _host>]() -> Result<()> {
                    buffer_cast::<$x, $y>(Device::host())
                }
                $(#[cfg(feature = $features)])*
                #[test]
                fn [<scalar_buffer_cast_ $x _ $y _host>]() -> Result<()> {
                    scalar_buffer_cast::<$x, $y>(Device::host())
                }
                $(#[cfg(feature = $features)])*
                #[cfg(feature = "device")]
                $(#[$ignore])?
                #[test]
                fn [<buffer_cast_ $x _ $y _device>]() -> Result<()> {
                    buffer_cast::<$x, $y>(Device::new(0)?)
                }
                $(#[cfg(feature = $features)])*
                #[cfg(feature = "device")]
                $(#[$ignore])?
                #[test]
                fn [<scalar_buffer_cast_ $x _ $y _device>]() -> Result<()> {
                    scalar_buffer_cast::<$x, $y>(Device::new(0)?)
                }
            }
            impl_buffer_cast_tests!(features($($features),*), $(@ignore($ignore),)? $x => $($ys),*);
        );
        (features($($features:literal),*), $(@ignore($ignore:tt),)? => $($ys:ty),*) => ();
        (features($($features:literal),*), $(@ignore($ignore:tt),)? $x:ty $(,$xs:ty)* => $($ys:ty),*) => (
            impl_buffer_cast_tests!(@ForY features($($features),*), $(@ignore($ignore),)? $x => $($ys),*);
            impl_buffer_cast_tests!(features($($features),*), $(@ignore($ignore),)? $($xs),* => $($ys),*);
        );
    }

    impl_buffer_cast_tests!(features(), u32, i32, f32 => u32, i32, f32);
    impl_buffer_cast_tests!(features(), @ignore(ignore), u8, i8, u16, i16, u32, i32, f32, u64, i64, f64 => u8, i8, u16, i16, u64, i64, f64);
    impl_buffer_cast_tests!(features(), @ignore(ignore), u8, i8, u16, u64, i64, f64 => u32, i32, f32);
    impl_buffer_cast_tests!(features("half"), @ignore(ignore), f16, bf16 => u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64);
    impl_buffer_cast_tests!(features("half"), @ignore(ignore), u8, i8, u16, i16, u32, i32, f32, u64, i64, f64 => f16, bf16);
}
