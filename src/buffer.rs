/*!

Buffers store data, and can be arguments to [kernels](crate::kernel). Buffers on the [host](crate::device::Device::host)
store data inline, essentially as a [`Vec`] or `[_]`.

[`BufferBase`](crate::buffer::BufferBase) is a generic buffer. [`ScalarBufferBase`](crate::buffer::ScalarBufferBase) is a dynamically typed buffer.

# Example
```
use krnl::{anyhow::Result, device::Device, buffer::{Buffer, Slice, SliceMut}};

fn saxpy(x: Slice<f32>, alpha: f32, mut y: SliceMut<f32>) -> Result<()> {
    if let Some((x, y)) = x.as_host_slice().zip(y.as_host_slice_mut()) {
        for (x, y) in x.iter().copied().zip(y) {
            *y += alpha * x;
        }
        return Ok(());
    }
    todo!()
}

fn main() -> Result<()> {
    let x = vec![1f32];
    let alpha = 2f32;
    let y = vec![0f32];
    # if false {
    let device = Device::builder().build().ok().unwrap_or(Device::host());
    # }
    # let device = Device::host();
    let x = Buffer::from(x).into_device(device.clone())?;
    let mut y = Buffer::from(y).into_device(device.clone())?;
    saxpy(x.as_slice(), alpha, y.as_slice_mut())?;
    let y = y.into_vec()?;
    println!("{y:?}");
    Ok(())
}
```
*/

#[cfg(doc)]
use crate::device::error::DeviceLost;
use crate::{
    device::{Device, DeviceInner},
    scalar::{Scalar, ScalarElem, ScalarType},
};
use anyhow::{bail, Result};
use bytemuck::PodCastError;
use dry::{macro_for, macro_wrap};
use half::{bf16, f16};
#[cfg(feature = "device")]
use krnl_macros::module;
use paste::paste;
use serde::{de::Deserializer, ser::Serializer, Deserialize, Serialize};
use std::{
    fmt::{self, Debug},
    marker::PhantomData,
    mem::{forget, size_of},
    ops::{Bound, RangeBounds},
    sync::Arc,
};

#[cfg(feature = "device")]
use crate::device::DeviceBuffer;

/// Errors.
pub mod error {
    #[cfg(feature = "device")]
    use crate::device::DeviceId;
    use std::fmt::{self, Debug, Display};

    /// No more memory on the device.
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    pub struct OutOfDeviceMemory(#[cfg(feature = "device")] pub(crate) DeviceId);

    impl Display for OutOfDeviceMemory {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(self, f)
        }
    }

    /// Device buffers are limited to [`i32::MAX`] (2147483647) bytes .
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    pub struct DeviceBufferTooLarge {
        #[cfg(feature = "device")]
        #[allow(unused)]
        pub(crate) bytes: usize,
    }

    impl Display for DeviceBufferTooLarge {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(self, f)
        }
    }
}
#[cfg(doc)]
use error::{DeviceBufferTooLarge, OutOfDeviceMemory};

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
struct RawSlice {
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
struct RawBuffer {
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
            let len = slice.len / self.width;
            let cap = self.cap / self.width;
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

macro_for!($S in [ScalarBufferRepr, ScalarArcBufferRepr] {
    impl Sealed for $S {}
    unsafe impl Send for $S {}
    unsafe impl Sync for $S {}
});

macro_for!($S in [ScalarSliceRepr,ScalarSliceMutRepr, ScalarCowBufferRepr] {
    impl Sealed for $S<'_> {}
    unsafe impl Send for $S<'_> {}
    unsafe impl Sync for $S<'_> {}
});

/** Marker trait for scalar buffers.

This trait can not be implemented outside the crate. */
pub trait ScalarData: Sealed {
    #[doc(hidden)]
    fn as_scalar_slice(&self) -> ScalarSliceRepr;
    #[doc(hidden)]
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr> {
        None
    }
    #[doc(hidden)]
    fn device(&self) -> Device {
        self.as_scalar_slice().raw.device()
    }
    #[doc(hidden)]
    fn scalar_type(&self) -> ScalarType {
        self.as_scalar_slice().scalar_type
    }
    #[doc(hidden)]
    fn len(&self) -> usize {
        let slice = self.as_scalar_slice();
        slice.raw.len() / slice.scalar_type().size()
    }
    #[doc(hidden)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[doc(hidden)]
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self>
    where
        Self: Sized,
    {
        Err(self)
    }
    #[doc(hidden)]
    fn try_into_scalar_arc_buffer(self) -> Result<ScalarArcBufferRepr, Self>
    where
        Self: Sized,
    {
        match self.try_into_scalar_buffer() {
            Ok(buffer) => Ok(ScalarArcBufferRepr::from_scalar_buffer(buffer)),
            Err(this) => Err(this),
        }
    }
    #[doc(hidden)]
    fn to_scalar_arc_buffer(&self) -> Result<ScalarArcBufferRepr> {
        Ok(ScalarArcBufferRepr::from_scalar_buffer(
            self.as_scalar_slice().to_scalar_buffer()?,
        ))
    }
}

/// Marker trait for mutable scalar buffers.
pub trait ScalarDataMut: ScalarData {
    #[doc(hidden)]
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr;
}

/// Marker trait for owned scalar buffers.
pub trait ScalarDataOwned: ScalarData {
    #[doc(hidden)]
    fn from_scalar_buffer(buffer: ScalarBufferRepr) -> Self
    where
        Self: Sized;
    #[doc(hidden)]
    fn make_scalar_slice_mut(&mut self) -> Result<ScalarSliceMutRepr>;
}

/// [`ScalarBuffer`] representation.
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
            _m: PhantomData,
        }
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr> {
        Some(self.as_scalar_slice_mut())
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
            _m: PhantomData,
        }
    }
}

impl ScalarDataOwned for ScalarBufferRepr {
    fn from_scalar_buffer(buffer: Self) -> Self
    where
        Self: Sized,
    {
        buffer
    }
    fn make_scalar_slice_mut(&mut self) -> Result<ScalarSliceMutRepr> {
        Ok(self.as_scalar_slice_mut())
    }
}

/// [`ScalarSlice`] representation.
#[derive(Clone)]
pub struct ScalarSliceRepr<'a> {
    raw: RawSlice,
    scalar_type: ScalarType,
    _m: PhantomData<&'a ()>,
}

impl<'a> ScalarSliceRepr<'a> {
    fn to_scalar_buffer(&self) -> Result<ScalarBufferRepr> {
        Ok(ScalarSlice { data: self.clone() }.to_owned()?.data)
    }
    fn to_device(&self, device: Device) -> Result<ScalarBufferRepr> {
        Ok(ScalarSlice { data: self.clone() }.to_device(device)?.data)
    }
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
    fn slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        Some(Self {
            raw: self.raw.slice(range, self.scalar_type)?,
            ..self
        })
    }
}

impl<'a, T: Scalar> From<SliceRepr<'a, T>> for ScalarSliceRepr<'a> {
    fn from(slice: SliceRepr<'a, T>) -> Self {
        Self {
            raw: slice.raw,
            scalar_type: T::scalar_type(),
            _m: PhantomData,
        }
    }
}

impl ScalarData for ScalarSliceRepr<'_> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
}

impl Serialize for ScalarSliceRepr<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::{Error, SerializeSeq, SerializeTupleStruct};

        let scalar_type = self.scalar_type;
        let slice: ScalarCowBufferRepr = if self.device().is_host() {
            self.clone().into()
        } else {
            self.to_device(Device::host())
                .map_err(S::Error::custom)?
                .into()
        };
        let slice = slice.as_scalar_slice();
        let slice = SliceRepr::<u8>::try_from(slice.bitcast(ScalarType::U8).unwrap())
            .ok()
            .unwrap();
        let data = BufferData {
            bytes: slice.as_host_slice().unwrap(),
        };
        let mut tuple_struct = serializer.serialize_tuple_struct("Buffer", 3)?;
        tuple_struct.serialize_field(&scalar_type)?;
        tuple_struct.serialize_field(&self.len())?;
        tuple_struct.serialize_field(&data)?;

        struct BufferData<'a> {
            bytes: &'a [u8],
        }

        impl Serialize for BufferData<'_> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let len = self.bytes.len();
                let items = len / 8 + (len % 8 != 0) as usize;
                let mut seq = serializer.serialize_seq(Some(items))?;
                for chunk in self.bytes.chunks(8) {
                    let mut item = 0u64;
                    bytemuck::bytes_of_mut(&mut item)[..chunk.len()].copy_from_slice(chunk);
                    seq.serialize_element(&item.to_be())?;
                }
                seq.end()
            }
        }

        tuple_struct.end()
    }
}

impl<'de> Deserialize<'de> for ScalarBufferRepr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{DeserializeSeed, Error, SeqAccess, Visitor};

        struct BufferVisitor;

        impl<'de> Visitor<'de> for BufferVisitor {
            type Value = ScalarBufferRepr;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "struct Buffer")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let scalar_type = if let Some(scalar_type) = seq.next_element::<ScalarType>()? {
                    scalar_type
                } else {
                    return Err(A::Error::custom("expected ScalarType"));
                };
                let len = if let Some(len) = seq.next_element::<usize>()? {
                    len
                } else {
                    return Err(A::Error::custom("expected usize"));
                };
                let visitor = BufferDataVisitor { scalar_type, len };
                if let Some(buffer) = seq.next_element_seed(visitor)? {
                    Ok(buffer)
                } else {
                    Err(A::Error::custom("expected sequence"))
                }
            }
        }

        struct BufferDataVisitor {
            scalar_type: ScalarType,
            len: usize,
        }

        impl BufferDataVisitor {
            fn parse<'de, T: Scalar, A>(self, mut seq: A) -> Result<ScalarBufferRepr, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let Self { scalar_type, len } = self;
                let mut data = Vec::<T>::with_capacity(len);
                while data.len() < len {
                    if let Some(item) = seq.next_element::<u64>()? {
                        let item = u64::from_be(item);
                        data.extend(bytemuck::cast_slice(&[item]));
                    } else {
                        break;
                    }
                }
                if data.len() != len {
                    return Err(A::Error::invalid_length(
                        data.len(),
                        &len.to_string().as_str(),
                    ));
                }
                assert_eq!(size_of::<T>(), scalar_type.size());
                let raw = BufferRepr::from_vec(data).raw;
                Ok(ScalarBufferRepr { raw, scalar_type })
            }
        }

        impl<'de> Visitor<'de> for BufferDataVisitor {
            type Value = ScalarBufferRepr;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "buffer data")
            }
            fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                match self.scalar_type.size() {
                    4 => self.parse::<u32, _>(seq),
                    _ => todo!(),
                }
            }
        }

        impl<'de> DeserializeSeed<'de> for BufferDataVisitor {
            type Value = ScalarBufferRepr;
            fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserializer.deserialize_seq(self)
            }
        }

        deserializer.deserialize_tuple_struct("Buffer", 3, BufferVisitor)
    }
}

/// [`ScalarBuffer`] representation.
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
    fn slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        Some(Self {
            raw: self.raw.slice(range, self.scalar_type)?,
            ..self
        })
    }
    fn copy_from_scalar_slice(&mut self, src: &ScalarSliceRepr) -> Result<()> {
        if self.scalar_type() != src.scalar_type() {
            bail!(
                "Can not copy slice from {:?} to {:?}",
                src.scalar_type(),
                self.scalar_type()
            );
        }
        macro_wrap!(paste! {
            match self.scalar_type() {
                macro_for!($T in  [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    ScalarType::[<$T:upper>] => {
                        SliceMutRepr::<$T>::try_from(self.as_scalar_slice_mut())
                            .ok()
                            .unwrap()
                            .copy_from_slice(&src.clone().try_into().ok().unwrap())
                    }
                })
                _ => unreachable!(),
            }
        })
    }
}

impl<'a, T: Scalar> From<SliceMutRepr<'a, T>> for ScalarSliceMutRepr<'a> {
    fn from(slice: SliceMutRepr<'a, T>) -> Self {
        Self {
            raw: slice.raw,
            scalar_type: T::scalar_type(),
            _m: PhantomData,
        }
    }
}

impl ScalarData for ScalarSliceMutRepr<'_> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
}

impl ScalarDataMut for ScalarSliceMutRepr<'_> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr {
        ScalarSliceMutRepr {
            raw: self.raw.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
}

/// [`ScalarArcBuffer`] representation.
#[derive(Clone)]
pub struct ScalarArcBufferRepr {
    raw: Arc<RawBuffer>,
    scalar_type: ScalarType,
}

impl From<ScalarBufferRepr> for ScalarArcBufferRepr {
    fn from(buffer: ScalarBufferRepr) -> Self {
        Self::from_scalar_buffer(buffer)
    }
}

impl<T: Scalar> From<ArcBufferRepr<T>> for ScalarArcBufferRepr {
    fn from(buffer: ArcBufferRepr<T>) -> Self {
        Self {
            raw: buffer.raw,
            scalar_type: T::scalar_type(),
        }
    }
}

impl ScalarData for ScalarArcBufferRepr {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr> {
        let raw = Arc::get_mut(&mut self.raw)?;
        Some(ScalarSliceMutRepr {
            raw: raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        })
    }
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self> {
        match Arc::try_unwrap(self.raw) {
            Ok(raw) => Ok(ScalarBufferRepr {
                raw,
                scalar_type: self.scalar_type,
            }),
            Err(raw) => Err(Self {
                raw,
                scalar_type: self.scalar_type,
            }),
        }
    }
    fn try_into_scalar_arc_buffer(self) -> Result<Self, Self> {
        Ok(self)
    }
    fn to_scalar_arc_buffer(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

impl ScalarDataOwned for ScalarArcBufferRepr {
    fn from_scalar_buffer(buffer: ScalarBufferRepr) -> Self {
        Self {
            raw: Arc::new(buffer.raw),
            scalar_type: buffer.scalar_type,
        }
    }
    fn make_scalar_slice_mut(&mut self) -> Result<ScalarSliceMutRepr> {
        if let Some(raw) = Arc::get_mut(&mut self.raw) {
            return Ok(ScalarSliceMutRepr {
                raw: raw.slice.clone(),
                scalar_type: self.scalar_type,
                _m: PhantomData,
            });
        }
        self.raw = Arc::new(self.as_scalar_slice().to_scalar_buffer()?.raw);
        Ok(ScalarSliceMutRepr {
            raw: self.raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        })
    }
}

/// [`ScalarCowBuffer`] representation.
pub enum ScalarCowBufferRepr<'a> {
    #[doc(hidden)]
    Borrowed(ScalarSliceRepr<'a>),
    #[doc(hidden)]
    Owned(ScalarBufferRepr),
}

impl<'a> From<ScalarSliceRepr<'a>> for ScalarCowBufferRepr<'a> {
    fn from(slice: ScalarSliceRepr<'a>) -> Self {
        Self::Borrowed(slice)
    }
}

impl<'a> From<ScalarBufferRepr> for ScalarCowBufferRepr<'a> {
    fn from(buffer: ScalarBufferRepr) -> Self {
        Self::Owned(buffer)
    }
}

impl<'a, T: Scalar> From<CowBufferRepr<'a, T>> for ScalarCowBufferRepr<'a> {
    fn from(buffer: CowBufferRepr<'a, T>) -> Self {
        match buffer {
            CowBufferRepr::Borrowed(slice) => Self::Borrowed(slice.into()),
            CowBufferRepr::Owned(buffer) => Self::Owned(buffer.into()),
        }
    }
}

impl<'a> ScalarData for ScalarCowBufferRepr<'a> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        match self {
            Self::Borrowed(slice) => slice.clone(),
            Self::Owned(buffer) => buffer.as_scalar_slice(),
        }
    }
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self>
    where
        Self: Sized,
    {
        match self {
            Self::Borrowed(_) => Err(self),
            Self::Owned(buffer) => Ok(buffer),
        }
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr> {
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(buffer) => buffer.get_scalar_slice_mut(),
        }
    }
}

impl<'a> ScalarDataOwned for ScalarCowBufferRepr<'a> {
    fn from_scalar_buffer(buffer: ScalarBufferRepr) -> Self
    where
        Self: Sized,
    {
        Self::Owned(buffer)
    }
    fn make_scalar_slice_mut(&mut self) -> Result<ScalarSliceMutRepr> {
        match self {
            Self::Borrowed(slice) => {
                *self = Self::Owned(slice.to_scalar_buffer()?);
                Ok(self.get_scalar_slice_mut().unwrap())
            }
            Self::Owned(buffer) => buffer.make_scalar_slice_mut(),
        }
    }
}

/** Dynamically typed buffer.

Use [`From`] to convert from [`BufferBase`].
*/
#[derive(Clone)]
pub struct ScalarBufferBase<S: ScalarData> {
    data: S,
}

/** ScalarBuffer

See [`ScalarBufferBase`]. */
pub type ScalarBuffer = ScalarBufferBase<ScalarBufferRepr>;
/** ScalarSlice

See [`ScalarBufferBase`]. */
pub type ScalarSlice<'a> = ScalarBufferBase<ScalarSliceRepr<'a>>;
/** ScalarSliceMut

See [`ScalarBufferBase`]. */
pub type ScalarSliceMut<'a> = ScalarBufferBase<ScalarSliceMutRepr<'a>>;
/** ScalarArcBuffer

Like [`Arc`], a clonable shared scalar buffer.

See [`ScalarBufferBase`]. */
pub type ScalarArcBuffer = ScalarBufferBase<ScalarArcBufferRepr>;
/** CowBuffer

Like [`Cow`](::std::borrow::Cow), can be created from a [`ScalarSlice`] or [`ScalarBuffer`].

See [`BufferBase`]. */
pub type ScalarCowBuffer<'a> = ScalarBufferBase<ScalarCowBufferRepr<'a>>;

impl<S: ScalarDataOwned> ScalarBufferBase<S> {
    /// Allocate a scalar buffer.
    ///
    /// # Safety
    /// The buffer will not be initialized.
    ///
    /// **errors**
    /// - [`DeviceLost`]
    /// - [`DeviceBufferTooLarge`]
    /// - [`OutOfDeviceMemory`]
    ///
    /// See [`ScalarBufferBase::zeros()`] for a safe alternative.
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
    /** Create a scalar buffer filled with `elem'.

    **errors**
    - [`DeviceLost`]
    - [`DeviceBufferTooLarge`]
    - [`OutOfDeviceMemory`]
    - Could not dispatch the kernel.
        - This may require [`Features`](crate::device::Features) for the type.

    See [.fill()](ScalarBufferBase::fill).
     */
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
    /** Create a scalar buffer filled with 0's.

    See [`from_elem`](ScalarBufferBase::from_elem). */
    pub fn zeros(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Self::from_elem(device, len, ScalarElem::zero(scalar_type))
    }
    /** Create a scalar buffer filled with 1's.

    See [`from_elem`](ScalarBufferBase::from_elem). */
    pub fn ones(device: Device, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Self::from_elem(device, len, ScalarElem::one(scalar_type))
    }
    /// Create a scalar buffer from a [`ScalarBuffer`].
    pub fn from_scalar_buffer(buffer: ScalarBuffer) -> Self {
        let data = S::from_scalar_buffer(buffer.data);
        Self { data }
    }
}

impl<S: ScalarData> ScalarBufferBase<S> {
    /// The device.
    pub fn device(&self) -> Device {
        self.data.device()
    }
    /// The scalar_type.
    pub fn scalar_type(&self) -> ScalarType {
        self.data.scalar_type()
    }
    /// The length.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Is the buffer empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Borrow as a scalar slice.
    pub fn as_scalar_slice(&self) -> ScalarSlice {
        let data = self.data.as_scalar_slice();
        ScalarSlice { data }
    }
    /// Borrow as a mutable scalar slice.
    pub fn as_scalar_slice_mut(&mut self) -> ScalarSliceMut
    where
        S: ScalarDataMut,
    {
        let data = self.data.as_scalar_slice_mut();
        ScalarSliceMut { data }
    }
    /// Borrows as a mutable scalar slice if possible.
    pub fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMut> {
        Some(ScalarSliceMut {
            data: self.data.get_scalar_slice_mut()?,
        })
    }
    /// Borrows as a mutable scalar slice, cloning if necessary.
    pub fn make_scalar_slice_mut(&mut self) -> Result<ScalarSliceMut>
    where
        S: ScalarDataOwned,
    {
        Ok(ScalarSliceMut {
            data: self.data.make_scalar_slice_mut()?,
        })
    }
    /** Move into an owned scalar buffer.

    Avoids copying if possible.

    See [`.to_owned()`](ScalarBufferBase::to_owned). */
    pub fn into_owned(self) -> Result<ScalarBuffer> {
        match self.data.try_into_scalar_buffer() {
            Ok(data) => Ok(ScalarBuffer { data }),
            Err(data) => Self { data }.to_owned(),
        }
    }
    /** Copies to an owned scalar buffer.

    **errors**

    - [`DeviceLost`]
    - [`OutOfDeviceMemory`]
    - Could not dispatch the kernel. */
    pub fn to_owned(&self) -> Result<ScalarBuffer> {
        self.cast(self.scalar_type())
    }
    /** Moves into a scalar arc buffer.

    Avoids copying if possible.

    See [`.to_shared()`](ScalarBufferBase::to_shared). */
    pub fn into_shared(self) -> Result<ScalarArcBuffer> {
        let data = match self.data.try_into_scalar_arc_buffer() {
            Ok(data) => data,
            Err(data) => data.to_scalar_arc_buffer()?,
        };
        Ok(ScalarArcBuffer { data })
    }
    /** Copies or clones to a scalar arc buffer.

    ScalarArcBuffer can be cheaply cloned.

    See [`.to_owned`](ScalarBufferBase::to_owned). */
    pub fn to_shared(&self) -> Result<ScalarArcBuffer> {
        let data = self.data.to_scalar_arc_buffer()?;
        Ok(ScalarArcBuffer { data })
    }
    /** Moves the buffer to the device.

    Avoids copying if possible.

    See [`.to_device()`](ScalarBufferBase::to_device). */
    pub fn into_device(self, device: Device) -> Result<ScalarBuffer> {
        if device == self.device() {
            self.into_owned()
        } else {
            self.to_device(device)
        }
    }
    /** Moves the buffer to the device as a scalar arc buffer

    ScalarArcBuffer can be cheeply cloned.

    See [`.into_device()`](ScalarBufferBase::into_device). */
    pub fn into_device_shared(self, device: Device) -> Result<ScalarArcBuffer> {
        if device == self.device() {
            self.into_shared()
        } else {
            self.to_device(device).map(Into::into)
        }
    }
    /** Copies to the device.

    See [`BufferBase::to_device`]. */
    pub fn to_device(&self, device: Device) -> Result<ScalarBuffer> {
        let slice = self.as_scalar_slice();
        macro_wrap!(paste! { match slice.scalar_type() {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                ScalarType::[<$T:upper>] => Ok(Slice::<$T>::try_from(slice).ok().unwrap().to_device(device)?.into()),
            })
            _ => unreachable!(),
        }})
    }
    /** Copies to the device in place.

    **errors**

    See [`.to_device()`](BufferBase::to_device). */
    pub fn to_device_mut(&mut self, device: Device) -> Result<()>
    where
        S: ScalarDataOwned,
    {
        if self.device() != device {
            self.data = S::from_scalar_buffer(self.data.as_scalar_slice().to_device(device)?);
        }
        Ok(())
    }
    /** Copies to the device as a scalar arc buffer.

    See [`.to_device()`](ScalarBufferBase::to_device). */
    pub fn to_device_shared(&self, device: Device) -> Result<ScalarArcBuffer> {
        if device == self.device() {
            self.to_shared()
        } else {
            self.to_device(device).map(Into::into)
        }
    }
    /** Fills the buffer with `elem`.

    See [`BufferBase::fill`]. */
    pub fn fill(&mut self, elem: ScalarElem) -> Result<()>
    where
        S: ScalarDataMut,
    {
        let slice = self.as_scalar_slice_mut();
        macro_wrap!(paste! { match elem {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                ScalarElem::[<$T:upper>](elem) => {
                    SliceMut::<$T>::try_from(slice).ok().unwrap().fill(elem)?;
                }
            })
            _ => unreachable!(),
        }});
        Ok(())
    }
    /** Casts to `scalar_type`.

    See [`BufferBase::cast`]. */
    pub fn cast(&self, scalar_type: ScalarType) -> Result<ScalarBuffer> {
        macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            if let Ok(x) = Slice::<$X>::try_from(self.as_scalar_slice()) {
                macro_wrap!(paste! {
                    match scalar_type {
                        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                            ScalarType::[<$Y:upper>] => {
                                return x.cast::<$Y>().map(Into::into);
                            }
                        })
                        _ => (),
                    }
                });
            }
        });
        unreachable!()
    }
    /** Casts into `scalar_type`.

    See [`.cast()`](ScalarBufferBase::scalar_cast). */
    pub fn cast_into(self, scalar_type: ScalarType) -> Result<ScalarBuffer> {
        if self.scalar_type() == scalar_type {
            self.into_owned()
        } else {
            self.cast(scalar_type)
        }
    }
    /** Casts into `scalar_type` as a scalar arc buffer.

    See [`.cast()`](ScalarBufferBase::cast). */
    pub fn cast_shared(&self, scalar_type: ScalarType) -> Result<ScalarArcBuffer> {
        if self.scalar_type() == scalar_type {
            self.to_shared()
        } else {
            self.cast(scalar_type).map(Into::into)
        }
    }
    /** Reinterpret as a slice with `scalar_type`.

    See [`bytemuck::cast_slice`]. */
    pub fn bitcast(&self, scalar_type: ScalarType) -> Result<ScalarSlice, PodCastError> {
        let data = self.data.as_scalar_slice().bitcast(scalar_type)?;
        Ok(ScalarSlice { data })
    }
    /** Reinterpret as a mutable slice with `scalar_type`.

    See [`bytemuck::cast_slice_mut`]. */
    pub fn bitcast_mut(&mut self, scalar_type: ScalarType) -> Result<ScalarSliceMut, PodCastError>
    where
        S: DataMut,
    {
        let data = self.data.as_scalar_slice_mut().bitcast_mut(scalar_type)?;
        Ok(ScalarSliceMut { data })
    }
    /** Copies from src.

    See [`BufferBase::copy_from_slice`]. */
    pub fn copy_from_scalar_slice(&mut self, src: &ScalarSlice) -> Result<()>
    where
        S: ScalarDataMut,
    {
        self.data
            .as_scalar_slice_mut()
            .copy_from_scalar_slice(&src.data)
    }
    /** A subslice with `range`.

    See [`BufferBase::slice`]. */
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<ScalarSlice> {
        let data = self.data.as_scalar_slice().slice(range)?;
        Some(ScalarSlice { data })
    }
    /** A mutable subslice with `range`.

    See [`BufferBase::slice_mut`]. */
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
        let data = S::from_scalar_buffer(ScalarBufferRepr::from(buffer.data));
        Self { data }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for ScalarSlice<'a> {
    fn from(slice: Slice<'a, T>) -> Self {
        let data = ScalarSliceRepr {
            raw: slice.data.raw,
            scalar_type: T::scalar_type(),
            _m: PhantomData,
        };
        Self { data }
    }
}

impl<'a, T: Scalar> From<SliceMut<'a, T>> for ScalarSliceMut<'a> {
    fn from(slice: SliceMut<'a, T>) -> Self {
        let data = ScalarSliceMutRepr {
            raw: slice.data.raw,
            scalar_type: T::scalar_type(),
            _m: PhantomData,
        };
        Self { data }
    }
}

impl From<ScalarBuffer> for ScalarArcBuffer {
    fn from(buffer: ScalarBuffer) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<T: Scalar> From<ArcBuffer<T>> for ScalarArcBuffer {
    fn from(buffer: ArcBuffer<T>) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl From<ScalarBuffer> for ScalarCowBuffer<'_> {
    fn from(buffer: ScalarBuffer) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<'a> From<ScalarSlice<'a>> for ScalarCowBuffer<'a> {
    fn from(slice: ScalarSlice<'a>) -> Self {
        Self {
            data: slice.data.into(),
        }
    }
}

impl<'a, T: Scalar> From<CowBuffer<'a, T>> for ScalarCowBuffer<'a> {
    fn from(buffer: CowBuffer<'a, T>) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<S: ScalarData> Debug for ScalarBufferBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ScalarBufferBase")
            .field("device", &self.device())
            .field("scalar_type", &self.scalar_type())
            .field("len", &self.len())
            .finish()
    }
}

impl<S1: ScalarData> Serialize for ScalarBufferBase<S1> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.as_scalar_slice().serialize(serializer)
    }
}

impl<'de, S: ScalarDataOwned> Deserialize<'de> for ScalarBufferBase<S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self {
            data: S::from_scalar_buffer(ScalarBufferRepr::deserialize(deserializer)?),
        })
    }
}

macro_for!($S in [BufferRepr, ArcBufferRepr] {
    impl<T: Scalar> Sealed for $S<T> {}
    unsafe impl<T: Scalar> Send for $S<T> {}
    unsafe impl<T: Scalar> Sync for $S<T> {}
});

macro_for!($S in [SliceRepr, SliceMutRepr, CowBufferRepr] {
    impl<T: Scalar> Sealed for $S<'_, T> {}
    unsafe impl<T: Scalar> Send for $S<'_, T> {}
    unsafe impl<T: Scalar> Sync for $S<'_, T> {}
});

/// Marker trait for buffers.
pub trait Data: ScalarData {
    /// The type of the buffer.
    type Elem: Scalar;
    #[doc(hidden)]
    fn as_slice(&self) -> SliceRepr<Self::Elem>;
    #[doc(hidden)]
    fn get_slice_mut(&mut self) -> Option<SliceMutRepr<Self::Elem>> {
        None
    }
    #[doc(hidden)]
    fn as_host_slice(&self) -> Option<&[Self::Elem]> {
        self.as_slice().into_host_slice()
    }
    #[doc(hidden)]
    fn try_into_buffer(self) -> Result<BufferRepr<Self::Elem>, Self>
    where
        Self: Sized,
    {
        Err(self)
    }
    #[doc(hidden)]
    fn try_into_arc_buffer(self) -> Result<ArcBufferRepr<Self::Elem>, Self>
    where
        Self: Sized,
    {
        match self.try_into_buffer() {
            Ok(buffer) => Ok(ArcBufferRepr::from_buffer(buffer)),
            Err(this) => Err(this),
        }
    }
    #[doc(hidden)]
    fn to_arc_buffer(&self) -> Result<ArcBufferRepr<Self::Elem>>
    where
        Self: Sized,
    {
        Ok(ArcBufferRepr::from_buffer(self.as_slice().to_buffer()?))
    }
}

/// Marker trait for mutable buffers.
pub trait DataMut: Data + ScalarDataMut {
    #[doc(hidden)]
    fn as_slice_mut(&mut self) -> SliceMutRepr<Self::Elem>;
    #[doc(hidden)]
    fn as_host_slice_mut(&mut self) -> Option<&mut [Self::Elem]> {
        self.as_slice_mut().into_host_slice_mut()
    }
}

/// Marker trait for owned buffers.
pub trait DataOwned: Data {
    #[doc(hidden)]
    fn from_buffer(buffer: BufferRepr<Self::Elem>) -> Self;
    #[doc(hidden)]
    fn make_slice_mut(&mut self) -> Result<SliceMutRepr<Self::Elem>>;
}

/// [`Buffer`] representation.
pub struct BufferRepr<T> {
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
            _m: PhantomData,
        }
    }
    unsafe fn uninit(device: Device, len: usize) -> Result<Self> {
        match device.inner() {
            DeviceInner::Host => {
                let mut vec = Vec::with_capacity(len);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    vec.set_len(len);
                }
                Ok(Self::from_vec(vec))
            }
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
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> ScalarDataMut for BufferRepr<T> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr {
        ScalarSliceMutRepr {
            raw: self.raw.clone(),
            scalar_type: T::scalar_type(),
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> Data for BufferRepr<T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<Self::Elem> {
        SliceRepr {
            raw: self.raw.slice.clone(),
            _m: PhantomData,
        }
    }
    fn get_slice_mut(&mut self) -> Option<SliceMutRepr<T>> {
        Some(self.as_slice_mut())
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
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> DataOwned for BufferRepr<T> {
    fn from_buffer(buffer: Self) -> Self {
        buffer
    }
    fn make_slice_mut(&mut self) -> Result<SliceMutRepr<T>> {
        Ok(self.as_slice_mut())
    }
}

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

/// [`Slice`] representation.
#[derive(Clone)]
pub struct SliceRepr<'a, T> {
    raw: RawSlice,
    _m: PhantomData<&'a T>,
}

impl<'a, T: Scalar> SliceRepr<'a, T> {
    fn from_host_slice(host_slice: &'a [T]) -> Self {
        let ptr = host_slice.as_ptr() as *mut u8;
        let len = std::mem::size_of_val(host_slice);
        let raw = RawSlice {
            inner: RawSliceInner::Host(RawHostSlice { ptr, len }),
        };
        Self {
            raw,
            _m: PhantomData,
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
    fn to_buffer(&self) -> Result<BufferRepr<T>> {
        self.to_device(self.device())
    }
    fn to_device(&self, device: Device) -> Result<BufferRepr<T>> {
        let mut output = unsafe { BufferRepr::uninit(device, self.len())? };
        output.as_slice_mut().copy_from_slice(self)?;
        Ok(output)
    }
    fn bitcast<Y: Scalar>(self) -> Result<SliceRepr<'a, Y>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(Y::scalar_type())?;
        Ok(SliceRepr {
            raw,
            _m: PhantomData,
        })
    }
    fn slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        Some(Self {
            raw: self.raw.slice(range, T::scalar_type())?,
            ..self
        })
    }
}

impl<'a, T: Scalar> ScalarData for SliceRepr<'a, T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: T::scalar_type(),
            _m: PhantomData,
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

/// [`SliceMut`] representation.
pub struct SliceMutRepr<'a, T> {
    raw: RawSlice,
    _m: PhantomData<&'a T>,
}

impl<'a, T: Scalar> SliceMutRepr<'a, T> {
    fn from_host_slice_mut(host_slice: &'a mut [T]) -> Self {
        let ptr = host_slice.as_ptr() as *mut u8;
        let len = std::mem::size_of_val(host_slice);
        let raw = RawSlice {
            inner: RawSliceInner::Host(RawHostSlice { ptr, len }),
        };
        Self {
            raw,
            _m: PhantomData,
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
        if self.len() != src.len() {
            bail!(
                "source slice length ({}) does not match destination slice length ({})",
                src.len(),
                self.len()
            );
        }
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
            _m: PhantomData,
        })
    }
    fn slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        Some(Self {
            raw: self.raw.slice(range, T::scalar_type())?,
            ..self
        })
    }
}

impl<T: Scalar> ScalarData for SliceMutRepr<'_, T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: T::scalar_type(),
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> ScalarDataMut for SliceMutRepr<'_, T> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr {
        ScalarSliceMutRepr {
            raw: self.raw.clone(),
            scalar_type: T::scalar_type(),
            _m: PhantomData,
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

/// [`ArcBuffer`] representation.
#[derive(Clone)]
pub struct ArcBufferRepr<T> {
    raw: Arc<RawBuffer>,
    _m: PhantomData<T>,
}

impl<T: Scalar> From<BufferRepr<T>> for ArcBufferRepr<T> {
    fn from(buffer: BufferRepr<T>) -> Self {
        Self {
            raw: Arc::new(buffer.raw),
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> TryFrom<ScalarArcBufferRepr> for ArcBufferRepr<T> {
    type Error = ScalarArcBufferRepr;
    fn try_from(buffer: ScalarArcBufferRepr) -> Result<Self, Self::Error> {
        if T::scalar_type() == buffer.scalar_type {
            Ok(Self {
                raw: buffer.raw,
                _m: PhantomData,
            })
        } else {
            Err(buffer)
        }
    }
}

impl<T: Scalar> ScalarData for ArcBufferRepr<T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        self.as_slice().into()
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr> {
        self.get_slice_mut().map(Into::into)
    }
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self> {
        self.try_into_buffer().map(Into::into)
    }
    fn try_into_scalar_arc_buffer(self) -> Result<ScalarArcBufferRepr, Self> {
        self.try_into_arc_buffer().map(Into::into)
    }
    fn to_scalar_arc_buffer(&self) -> Result<ScalarArcBufferRepr> {
        Ok(self.clone().into())
    }
}

impl<T: Scalar> Data for ArcBufferRepr<T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<T> {
        SliceRepr {
            raw: self.raw.slice.clone(),
            _m: PhantomData,
        }
    }
    fn get_slice_mut(&mut self) -> Option<SliceMutRepr<T>> {
        let raw = Arc::get_mut(&mut self.raw)?;
        Some(SliceMutRepr {
            raw: raw.slice.clone(),
            _m: PhantomData,
        })
    }
    fn try_into_buffer(self) -> Result<BufferRepr<T>, Self> {
        match Arc::try_unwrap(self.raw) {
            Ok(raw) => Ok(BufferRepr {
                raw,
                _m: PhantomData,
            }),
            Err(raw) => Err(Self {
                raw,
                _m: PhantomData,
            }),
        }
    }
    fn try_into_arc_buffer(self) -> Result<Self, Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
    fn to_arc_buffer(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

impl<T: Scalar> DataOwned for ArcBufferRepr<T> {
    fn from_buffer(buffer: BufferRepr<T>) -> Self {
        Self {
            raw: Arc::new(buffer.raw),
            _m: PhantomData,
        }
    }
    fn make_slice_mut(&mut self) -> Result<SliceMutRepr<T>> {
        if let Some(raw) = Arc::get_mut(&mut self.raw) {
            return Ok(SliceMutRepr {
                raw: raw.slice.clone(),
                _m: PhantomData,
            });
        }
        self.raw = Arc::new(self.as_slice().to_buffer()?.raw);
        Ok(SliceMutRepr {
            raw: self.raw.slice.clone(),
            _m: PhantomData,
        })
    }
}

/// [`CowBuffer`] representation.
pub enum CowBufferRepr<'a, T> {
    #[doc(hidden)]
    Borrowed(SliceRepr<'a, T>),
    #[doc(hidden)]
    Owned(BufferRepr<T>),
}

impl<'a, T> From<SliceRepr<'a, T>> for CowBufferRepr<'a, T> {
    fn from(slice: SliceRepr<'a, T>) -> Self {
        Self::Borrowed(slice)
    }
}

impl<'a, T> From<BufferRepr<T>> for CowBufferRepr<'a, T> {
    fn from(buffer: BufferRepr<T>) -> Self {
        Self::Owned(buffer)
    }
}

impl<'a, T: Scalar> TryFrom<ScalarCowBufferRepr<'a>> for CowBufferRepr<'a, T> {
    type Error = ScalarCowBufferRepr<'a>;
    fn try_from(buffer: ScalarCowBufferRepr<'a>) -> Result<Self, Self::Error> {
        match buffer {
            ScalarCowBufferRepr::Borrowed(slice) => SliceRepr::try_from(slice)
                .map(Into::into)
                .map_err(Into::into),
            ScalarCowBufferRepr::Owned(buffer) => BufferRepr::try_from(buffer)
                .map(Into::into)
                .map_err(Into::into),
        }
    }
}

impl<'a, T: Scalar> ScalarData for CowBufferRepr<'a, T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr {
        self.as_slice().into()
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr> {
        self.get_slice_mut().map(Into::into)
    }
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self> {
        self.try_into_buffer().map(Into::into)
    }
}

impl<'a, T: Scalar> Data for CowBufferRepr<'a, T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<T> {
        match self {
            Self::Borrowed(slice) => slice.clone(),
            Self::Owned(buffer) => buffer.as_slice(),
        }
    }
    fn get_slice_mut(&mut self) -> Option<SliceMutRepr<T>> {
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(buffer) => buffer.get_slice_mut(),
        }
    }
    fn try_into_buffer(self) -> Result<BufferRepr<T>, Self> {
        match self {
            Self::Borrowed(_) => Err(self),
            Self::Owned(buffer) => Ok(buffer),
        }
    }
}

impl<'a, T: Scalar> DataOwned for CowBufferRepr<'a, T> {
    fn from_buffer(buffer: BufferRepr<T>) -> Self {
        Self::Owned(buffer)
    }
    fn make_slice_mut(&mut self) -> Result<SliceMutRepr<T>> {
        match self {
            Self::Borrowed(slice) => {
                *self = Self::Owned(slice.to_buffer()?);
                Ok(self.get_slice_mut().unwrap())
            }
            Self::Owned(buffer) => buffer.make_slice_mut(),
        }
    }
}

/** A buffer.

Use [`TryFrom`](core::convert::TryFrom) to convert from [`ScalarBufferBase`].
*/
#[derive(Clone)]
pub struct BufferBase<S: Data> {
    data: S,
}

/** Buffer

See [`BufferBase`]. */
pub type Buffer<T> = BufferBase<BufferRepr<T>>;
/** Slice

See [`BufferBase`]. */
pub type Slice<'a, T> = BufferBase<SliceRepr<'a, T>>;
/** SliceMut

See [`BufferBase`]. */
pub type SliceMut<'a, T> = BufferBase<SliceMutRepr<'a, T>>;
/** ArcBuffer

Like [`Arc`], a clonable shared buffer.

See [`BufferBase`]. */
pub type ArcBuffer<T> = BufferBase<ArcBufferRepr<T>>;
/** CowBuffer

Like [`Cow`](::std::borrow::Cow), can be created from a [`Slice`] or [`Buffer`].

See [`BufferBase`]. */
pub type CowBuffer<'a, T> = BufferBase<CowBufferRepr<'a, T>>;

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

impl<T: Scalar> TryFrom<ScalarArcBuffer> for ArcBuffer<T> {
    type Error = ScalarArcBuffer;
    fn try_from(buffer: ScalarArcBuffer) -> Result<Self, Self::Error> {
        match buffer.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarArcBuffer { data }),
        }
    }
}

impl<T: Scalar> From<Buffer<T>> for ArcBuffer<T> {
    fn from(buffer: Buffer<T>) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for CowBuffer<'a, T> {
    fn from(slice: Slice<'a, T>) -> Self {
        Self {
            data: slice.data.into(),
        }
    }
}

impl<T: Scalar> From<Buffer<T>> for CowBuffer<'_, T> {
    fn from(buffer: Buffer<T>) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarCowBuffer<'a>> for CowBuffer<'a, T> {
    type Error = ScalarCowBuffer<'a>;
    fn try_from(buffer: ScalarCowBuffer<'a>) -> Result<Self, Self::Error> {
        match buffer.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarCowBuffer { data }),
        }
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> Default for BufferBase<S> {
    fn default() -> Self {
        Self::zeros(Device::host(), 0).unwrap()
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> BufferBase<S> {
    /// Allocate a buffer.
    ///
    /// # Safety
    /// The buffer will not be initialized.
    ///
    /// **errors**
    ///
    /// - [`DeviceLost`]
    /// - [`DeviceBufferTooLarge`]
    /// - [`OutOfDeviceMemory`]
    ///
    /// See [`zeros()`](Buffer::zeros) for a safe alternative.
    pub unsafe fn uninit(device: Device, len: usize) -> Result<Self> {
        let data = S::from_buffer(unsafe { BufferRepr::uninit(device, len)? });
        Ok(Self { data })
    }
    /** Create a buffer filled with `elem'

    **errors**
    - [`DeviceLost`]
    - [`DeviceBufferTooLarge`]
    - [`OutOfDeviceMemory`]

    See [`.fill()`](BufferBase::fill).
    */
    pub fn from_elem(device: Device, len: usize, elem: T) -> Result<Self> {
        let mut output = unsafe { Buffer::uninit(device, len)? };
        output.fill(elem)?;
        Ok(Self {
            data: S::from_buffer(output.data),
        })
    }
    /** Create a buffer filled with 0's.

    See [`from_elem`](BufferBase::from_elem). */
    pub fn zeros(device: Device, len: usize) -> Result<Self> {
        Self::from_elem(device, len, T::zero())
    }
    /** Create a buffer filled with 1's.

    See [`from_elem`](BufferBase::from_elem). */
    pub fn ones(device: Device, len: usize) -> Result<Self> {
        Self::from_elem(device, len, T::one())
    }
    /// Create a buffer from a [`Vec`].
    pub fn from_vec(vec: Vec<T>) -> Self {
        let data = S::from_buffer(BufferRepr::from_vec(vec));
        Self { data }
    }
    /// Create a buffer from a [`Buffer`].
    pub fn from_buffer(buffer: Buffer<T>) -> Self {
        let data = S::from_buffer(buffer.data);
        Self { data }
    }
}

impl<'a, T: Scalar> Slice<'a, T> {
    /// Create a slice from a `&[T]`.
    pub fn from_host_slice(host_slice: &'a [T]) -> Self {
        let data = SliceRepr::from_host_slice(host_slice);
        Self { data }
    }
}

impl<'a, T: Scalar> SliceMut<'a, T> {
    /// Create a mutable slice from a `&mut [T]`.
    pub fn from_host_slice_mut(host_slice: &'a mut [T]) -> Self {
        let data = SliceMutRepr::from_host_slice_mut(host_slice);
        Self { data }
    }
}

impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
    /// The device.
    pub fn device(&self) -> Device {
        self.data.device()
    }
    /// The scalar type.
    pub fn scalar_type(&self) -> ScalarType {
        self.data.scalar_type()
    }
    /// The length.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Is the buffer empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Borrow as a slice.
    pub fn as_slice(&self) -> Slice<T> {
        let data = self.data.as_slice();
        Slice { data }
    }
    /// Borrow as a mutable slice.
    pub fn as_slice_mut(&mut self) -> SliceMut<T>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut();
        SliceMut { data }
    }
    /// Borrows as a mutable slice if possible.
    pub fn get_slice_mut(&mut self) -> Option<SliceMut<T>> {
        Some(SliceMut {
            data: self.data.get_slice_mut()?,
        })
    }
    /// Borrows as a mutable slice, cloning if necessary.
    pub fn make_slice_mut(&mut self) -> Result<SliceMut<T>>
    where
        S: DataOwned,
    {
        Ok(SliceMut {
            data: self.data.make_slice_mut()?,
        })
    }
    /** Borrow as a host slice.

    Returns [`None`] if not on the host. */
    pub fn as_host_slice(&self) -> Option<&[T]> {
        self.data.as_host_slice()
    }
    /** Borrow as a mutable host slice.

    Returns [`None`] if not on the host. */
    pub fn as_host_slice_mut(&mut self) -> Option<&mut [T]>
    where
        S: DataMut,
    {
        self.data.as_host_slice_mut()
    }
    /// Borrow as a scalar slice.
    pub fn as_scalar_slice(&self) -> ScalarSlice {
        let data = self.data.as_scalar_slice();
        ScalarSlice { data }
    }
    /// Borrow as a mutable scalar slice.
    pub fn as_scalar_slice_mut(&mut self) -> ScalarSliceMut
    where
        S: DataMut,
    {
        let data = self.data.as_scalar_slice_mut();
        ScalarSliceMut { data }
    }
    /** Moves into a buffer.

    See [`.to_owned()`](BufferBase::to_owned). */
    pub fn into_owned(self) -> Result<Buffer<T>> {
        match self.data.try_into_buffer() {
            Ok(data) => Ok(Buffer { data }),
            Err(data) => Self { data }.to_owned(),
        }
    }
    /** Copies to a buffer.

    **errors**

    - [`DeviceLost`]
    - [`OutOfDeviceMemory`]
    - Could not dispatch the kernel. */
    pub fn to_owned(&self) -> Result<Buffer<T>> {
        self.cast()
    }
    /** Moves into an arc buffer.

    See [`.to_shared()`] */
    pub fn into_shared(self) -> Result<ArcBuffer<T>> {
        let data = match self.data.try_into_arc_buffer() {
            Ok(data) => data,
            Err(data) => data.to_arc_buffer()?,
        };
        Ok(ArcBuffer { data })
    }
    /** Copies or clones to an arc buffer.

    See [`.to_owned`](BufferBase::to_owned). */
    pub fn to_shared(&self) -> Result<ArcBuffer<T>> {
        let data = self.data.to_arc_buffer()?;
        Ok(ArcBuffer { data })
    }
    /** Moves into the device.

    See [`.into_device()`](BufferBase::into_device). */
    pub fn into_device(self, device: Device) -> Result<Buffer<T>> {
        if device == self.device() {
            self.into_owned()
        } else {
            self.to_device(device)
        }
    }
    /** Moves into the device as an arc buffer.

    See [`.into_device()`](BufferBase::into_device). */
    pub fn into_device_shared(self, device: Device) -> Result<ArcBuffer<T>> {
        if device == self.device() {
            self.into_shared()
        } else {
            self.to_device(device).map(Into::into)
        }
    }
    /** Copies to the device.

    **errors**

    - [`DeviceLost`]
    - [`OutOfDeviceMemory`]
    - Could not dispatch the kernel. */
    pub fn to_device(&self, device: Device) -> Result<Buffer<T>> {
        let data = self.data.as_slice().to_device(device)?;
        Ok(Buffer { data })
    }
    /** Copies to the device in place.

    **errors**

    See [`.to_device()`](BufferBase::to_device). */
    pub fn to_device_mut(&mut self, device: Device) -> Result<()>
    where
        S: DataOwned,
    {
        if self.device() != device {
            self.data = S::from_buffer(self.data.as_slice().to_device(device)?);
        }
        Ok(())
    }
    /** Copies to the device as an arc buffer.

    See [`.to_device()`](BufferBase::to_device). */
    pub fn to_device_shared(&self, device: Device) -> Result<ArcBuffer<T>> {
        if device == self.device() {
            self.to_shared()
        } else {
            self.to_device(device).map(Into::into)
        }
    }
    /** Moves into a [`Vec`].

    See [`.into_device()`](BufferBase::into_device) */
    pub fn into_vec(self) -> Result<Vec<T>> {
        match self.data.try_into_buffer() {
            Ok(data) => data.into_vec(),
            Err(data) => data.as_slice().to_vec(),
        }
    }
    /** Copies to a [`Vec`].

    See [`.to_device()`](BufferBase::to_device). */
    pub fn to_vec(&self) -> Result<Vec<T>> {
        self.data.as_slice().to_vec()
    }
    /** Fills with `elem`.

    **errors**
    - [`DeviceLost`]
    - The kernel could not be dispatched.
        - This may require [`Features`](crate::device::Features) for the type. */
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
                bail!("Device {device:?} does not support 8 bit operations!");
            }
        }
    }
    /** Casts to `Y`.

    **errors**
    - DeviceLost
    - The kernel could not be dispatched.
        - This may require [`Features`](crate::device::Features) for the type. */
    pub fn cast<Y: Scalar>(&self) -> Result<Buffer<Y>> {
        let mut output = unsafe { Buffer::uninit(self.device(), self.len())? };
        self.as_slice().cast_impl(&mut output.as_slice_mut())?;
        Ok(output)
    }
    /** Casts into `Y`.

    See [`.cast()`](BufferBase::cast). */
    pub fn cast_into<Y: Scalar>(self) -> Result<Buffer<Y>> {
        if T::scalar_type() == Y::scalar_type() {
            let buffer = self.into_owned()?;
            Ok(ScalarBuffer::from(buffer).try_into().ok().unwrap())
        } else {
            self.cast()
        }
    }
    /** Casts to `Y` as an arc buffer.

    See [`.cast()`](BufferBase::cast). */
    pub fn cast_shared<Y: Scalar>(&self) -> Result<ArcBuffer<Y>> {
        if T::scalar_type() == Y::scalar_type() {
            let buffer = self.to_shared()?;
            Ok(ScalarArcBuffer::from(buffer).try_into().ok().unwrap())
        } else {
            self.cast().map(Into::into)
        }
    }
    /** Reinterpret as a slice with type `Y`.

    See [`bytemuck::cast_slice`]. */
    pub fn bitcast<Y: Scalar>(&self) -> Result<Slice<Y>, bytemuck::PodCastError> {
        let data = self.data.as_slice().bitcast()?;
        Ok(Slice { data })
    }
    /** Reinterpret as a mutable slice with type `Y`.

    See [`bytemuck::cast_slice_mut`]. */
    pub fn bitcast_mut<Y: Scalar>(&mut self) -> Result<SliceMut<Y>, bytemuck::PodCastError>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut().bitcast()?;
        Ok(SliceMut { data })
    }
    /** Copies from src.

    **errors**
    - `src` is not the same length.
    - [`DeviceLost`]
    - The kernel could not be dispatched.
    */
    pub fn copy_from_slice(&mut self, src: &Slice<T>) -> Result<()>
    where
        S: DataMut,
    {
        self.data.as_slice_mut().copy_from_slice(&src.data)
    }
    /** A subslice with `range`.

    Returns None if range is out of bounds.

    See [`<[_]>::get()`](https://doc.rust-lang.org/std/primitive.slice.html#method.get). */
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<Slice<T>> {
        let data = self.data.as_slice().slice(range)?;
        Some(Slice { data })
    }
    /** A mutable subslice with `range`.

    Returns None if range is out of bounds.

    See [`<[_]>::get_mut()`](https://doc.rust-lang.org/std/primitive.slice.html#method.get_mut). */
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<SliceMut<T>>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut().slice(range)?;
        Some(SliceMut { data })
    }
}

impl<T: Scalar> Slice<'_, T> {
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
            device_scalar_buffer_cast_impl(self.as_scalar_slice(), output.as_scalar_slice_mut())
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
    }
}

#[cfg(feature = "device")]
fn device_scalar_buffer_cast_impl(x: ScalarSlice, y: ScalarSliceMut) -> Result<()> {
    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        let x = match Slice::<$X>::try_from(x) {
            Ok(x) => {
                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    let y = match SliceMut::<$Y>::try_from(y) {
                        Ok(y) => {
                            let builder = paste! {
                                kernels::[<cast_ $X _ $Y>]::builder()?
                            };
                            return builder.build(y.device())?.dispatch(x, y);
                        }
                        Err(y) => y,
                    };
                });
                let _ = y;
                unreachable!()
            }
            Err(x) => x,
        };
    });
    let _ = x;
    unreachable!()
}

impl<S: Data> Debug for BufferBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BufferBase")
            .field("device", &self.device())
            .field("scalar_type", &self.scalar_type())
            .field("len", &self.len())
            .finish()
    }
}

impl<S1: Data> Serialize for BufferBase<S1> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.as_scalar_slice().serialize(serializer)
    }
}

impl<'de, T: Scalar, S: DataOwned<Elem = T>> Deserialize<'de> for BufferBase<S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let scalar_buffer = ScalarBufferRepr::deserialize(deserializer)?;
        let buffer = BufferRepr::try_from(scalar_buffer).map_err(|e| {
            D::Error::custom(format!(
                "Expected {:?}, found {:?}!",
                T::scalar_type(),
                e.scalar_type()
            ))
        })?;
        Ok(Self {
            data: S::from_buffer(buffer),
        })
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
           #[kernel]
           pub fn [<fill_ $T>](x: $T, #[item] y: &mut $T) {
               *y = x;
           }
       }
    });

    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                #[kernel]
                pub fn [<cast_ $X _ $Y>](#[item] x: $X, #[item] y: &mut $Y) {
                    *y = Default::default();
                    *y = x.cast();
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_test::{assert_tokens, Token};

    #[test]
    fn buffer_serde_tokens() {
        let input = vec![1u32, 2, 3, 4];
        let items: Vec<u64> = input
            .chunks(2)
            .map(|chunk| {
                let mut item = 0u64;
                bytemuck::bytes_of_mut(&mut item).copy_from_slice(bytemuck::cast_slice(chunk));
                item.to_be()
            })
            .collect();
        let buffer = Buffer::from_vec(input.clone());
        let tokens = [
            Token::TupleStruct {
                name: "Buffer",
                len: 3,
            },
            Token::Str("U32"),
            Token::U64(4),
            Token::Seq { len: Some(2) },
            Token::U64(items[0]),
            Token::U64(items[1]),
            Token::SeqEnd,
            Token::TupleStructEnd,
        ];

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(transparent)]
        struct BufferWrap(Buffer<u32>);

        impl PartialEq for BufferWrap {
            fn eq(&self, other: &Self) -> bool {
                self.0.as_host_slice().unwrap() == other.0.as_host_slice().unwrap()
            }
        }

        impl Eq for BufferWrap {}

        assert_tokens(&BufferWrap(buffer), &tokens);
    }

    #[test]
    fn buffer_serde_json() {
        let x_vec = vec![1u32, 2, 3, 4];
        let string = serde_json::to_string(&Buffer::from_vec(x_vec.clone())).unwrap();
        let y_vec = serde_json::from_str::<Buffer<u32>>(&string)
            .unwrap()
            .into_vec()
            .unwrap();
        assert_eq!(x_vec, y_vec);
    }

    #[test]
    fn buffer_serde_bincode2() {
        let x_vec = vec![1u32, 2, 3, 4];
        let bytes = bincode2::serialize(&Buffer::from_vec(x_vec.clone())).unwrap();
        let y_vec = bincode2::deserialize::<Buffer<u32>>(&bytes)
            .unwrap()
            .into_vec()
            .unwrap();
        assert_eq!(x_vec, y_vec);
    }
}
