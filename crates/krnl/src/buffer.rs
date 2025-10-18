use crate::{
    Result,
    context::{Buffer as RawBuffer, Context, Slice as RawSlice, SliceMut as RawSliceMut},
};
use bytemuck::Pod;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::sync::Arc;

pub trait Data: Sized {
    type Elem;
    fn __as_slice(&self) -> SliceRepr<'_, Self::Elem>;
    fn __into_buffer(self) -> Result<BufferRepr<Self::Elem>>
    where
        Self::Elem: Pod;
}

pub trait DataOwned: Data {
    fn __from_buffer(buffer: BufferRepr<Self::Elem>) -> Self;
    /*
    fn __try_into_buffer(self) -> Option<BufferRepr<Self::Elem>>
    where
        Self::Elem: Pod;
    */
}

pub trait DataMut: DataTryMut {
    fn __as_slice_mut(&mut self) -> SliceMutRepr<'_, Self::Elem> {
        self.__get_slice_mut().unwrap()
    }
}

pub trait DataTryMut: Data {
    fn __get_slice_mut(&mut self) -> Option<SliceMutRepr<'_, Self::Elem>>;
    fn __make_slice_mut(&mut self) -> Result<SliceMutRepr<'_, Self::Elem>>
    where
        Self::Elem: Pod;
}

pub struct BufferRepr<T> {
    raw: RawBuffer<T>,
}

impl<T> Data for BufferRepr<T> {
    type Elem = T;
    fn __as_slice(&self) -> SliceRepr<'_, Self::Elem> {
        SliceRepr {
            raw: self.raw.as_slice(),
        }
    }
    fn __into_buffer(self) -> Result<BufferRepr<Self::Elem>>
    where
        Self::Elem: Pod,
    {
        Ok(self)
    }
}

impl<T> DataOwned for BufferRepr<T> {
    fn __from_buffer(buffer: BufferRepr<Self::Elem>) -> Self {
        buffer
    }
    /*
    fn __into_buffer(self) -> BufferRepr<Self::Elem> {
        self
    }
    */
}

impl<T> DataMut for BufferRepr<T> {
    fn __as_slice_mut(&mut self) -> SliceMutRepr<'_, Self::Elem> {
        SliceMutRepr {
            raw: self.raw.as_slice_mut(),
        }
    }
}

impl<T> DataTryMut for BufferRepr<T> {
    fn __get_slice_mut(&mut self) -> Option<SliceMutRepr<'_, Self::Elem>> {
        Some(self.__as_slice_mut())
    }
    fn __make_slice_mut(&mut self) -> Result<SliceMutRepr<'_, Self::Elem>>
    where
        Self::Elem: Pod,
    {
        Ok(self.__as_slice_mut())
    }
}

pub struct SliceRepr<'a, T> {
    raw: RawSlice<'a, T>,
}

impl<T> Data for SliceRepr<'_, T> {
    type Elem = T;
    fn __as_slice(&self) -> SliceRepr<'_, Self::Elem> {
        SliceRepr {
            raw: self.raw.clone(),
        }
    }
    fn __into_buffer(self) -> Result<BufferRepr<Self::Elem>>
    where
        Self::Elem: Pod,
    {
        Ok(BufferRepr {
            raw: self.raw.to_buffer()?,
        })
    }
}

pub struct SliceMutRepr<'a, T> {
    raw: RawSliceMut<'a, T>,
}

impl<T> Data for SliceMutRepr<'_, T> {
    type Elem = T;
    fn __as_slice(&self) -> SliceRepr<'_, Self::Elem> {
        SliceRepr {
            raw: self.raw.as_slice(),
        }
    }
    fn __into_buffer(self) -> Result<BufferRepr<Self::Elem>>
    where
        Self::Elem: Pod,
    {
        Ok(BufferRepr {
            raw: self.raw.as_slice().to_buffer()?,
        })
    }
}

impl<T> DataMut for SliceMutRepr<'_, T> {
    fn __as_slice_mut(&mut self) -> SliceMutRepr<'_, Self::Elem> {
        SliceMutRepr {
            raw: self.raw.as_slice_mut(),
        }
    }
}

impl<T> DataTryMut for SliceMutRepr<'_, T> {
    fn __get_slice_mut(&mut self) -> Option<SliceMutRepr<'_, Self::Elem>> {
        Some(self.__as_slice_mut())
    }
    fn __make_slice_mut(&mut self) -> Result<SliceMutRepr<'_, Self::Elem>>
    where
        Self::Elem: Pod,
    {
        Ok(self.__as_slice_mut())
    }
}

pub struct ArcBufferRepr<T> {
    raw: Arc<RawBuffer<T>>,
}

pub struct BufferBase<S> {
    data: S,
}

pub type Buffer<T> = BufferBase<BufferRepr<T>>;
pub type Slice<'a, T> = BufferBase<SliceRepr<'a, T>>;
pub type SliceMut<'a, T> = BufferBase<SliceMutRepr<'a, T>>;
pub type ArcBuffer<T> = BufferBase<ArcBufferRepr<T>>;

impl<T, S: DataOwned<Elem = T>> From<Vec<T>> for BufferBase<S> {
    fn from(vec: Vec<T>) -> Self {
        Self {
            data: S::__from_buffer(BufferRepr {
                raw: RawBuffer::Host(vec),
            }),
        }
    }
}

impl<'a, T> From<&'a [T]> for Slice<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self {
            data: SliceRepr {
                raw: RawSlice::Host(slice),
            },
        }
    }
}

impl<'a, T> Slice<'a, T> {
    pub fn into_host_slice(self) -> Option<&'a [T]> {
        #[allow(irrefutable_let_patterns)]
        if let RawSlice::Host(slice) = self.data.raw {
            Some(slice)
        } else {
            None
        }
    }
}

impl<'a, T> SliceMut<'a, T> {
    pub fn into_host_slice_mut(self) -> Option<&'a mut [T]> {
        #[allow(irrefutable_let_patterns)]
        if let RawSliceMut::Host(slice) = self.data.raw {
            Some(slice)
        } else {
            None
        }
    }
}

impl<T: Pod, S: DataOwned<Elem = T>> BufferBase<S> {
    pub unsafe fn uninit(context: Context, len: usize) -> Result<Self> {
        let raw = unsafe { RawBuffer::uninit(context, len)? };
        Ok(Self {
            data: S::__from_buffer(BufferRepr { raw }),
        })
    }
    pub fn zeros(context: Context, len: usize) -> Result<Self> {
        if context.is_host() {
            return Ok(Self::from(vec![T::zeroed(); len]));
        }
        let mut buffer = unsafe { Buffer::uninit(context.clone(), len)? };
        if len % 8 == 0 {
            buffer
                .as_slice_mut()
                .try_bitcast_mut()
                .unwrap()
                .fill([0u32; 2])?;
        } else if len % 4 == 0 {
            buffer
                .as_slice_mut()
                .try_bitcast_mut()
                .unwrap()
                .fill(0u32)?;
        } else if len % 2 == 0 {
            buffer
                .as_slice_mut()
                .try_bitcast_mut()
                .unwrap()
                .fill(0u16)?;
        } else {
            buffer.as_slice_mut().try_bitcast_mut().unwrap().fill(0u8)?;
        }
        Ok(Self {
            data: S::__from_buffer(buffer.data),
        })
    }
}

impl<S: Data> BufferBase<S> {
    pub fn context(&self) -> Context {
        self.data.__as_slice().raw.context()
    }
    pub fn len(&self) -> usize {
        self.data.__as_slice().raw.len()
    }
    pub fn as_slice(&self) -> Slice<'_, S::Elem> {
        Slice {
            data: self.data.__as_slice(),
        }
    }
    pub fn as_slice_mut(&mut self) -> SliceMut<'_, S::Elem>
    where
        S: DataMut,
    {
        SliceMut {
            data: self.data.__as_slice_mut(),
        }
    }
}

impl<'a, T> Slice<'a, T> {
    pub(crate) fn as_context_slice(&self) -> &crate::context::Slice<'a, T> {
        &self.data.raw
    }
}

impl<'a, T> SliceMut<'a, T> {
    pub(crate) fn as_context_slice_mut(&mut self) -> &mut crate::context::SliceMut<'a, T> {
        &mut self.data.raw
    }
}

impl<'a, T: Pod> SliceMut<'a, T> {
    pub fn try_bitcast_mut<Y: Pod>(self) -> Option<SliceMut<'a, Y>> {
        self.data.raw.try_bitcast_mut().map(|raw| SliceMut {
            data: SliceMutRepr { raw },
        })
    }
}

impl<T: Pod, S: Data<Elem = T>> BufferBase<S> {
    pub fn into_context(self, context: Context) -> Result<Buffer<T>> {
        if self.context() == context {
            Ok(Buffer {
                data: self.data.__into_buffer()?,
            })
        } else {
            Ok(Buffer {
                data: BufferRepr {
                    raw: self.data.__as_slice().raw.to_context(context)?,
                },
            })
        }
    }
    pub fn into_vec(self) -> Result<Vec<T>> {
        let buffer = self.into_context(Context::Host)?;
        match buffer.data.raw {
            RawBuffer::Host(vec) => Ok(vec),
            #[cfg(feature = "device")]
            RawBuffer::Device(_) => unreachable!(),
        }
    }
    #[cfg(target_family = "wasm")]
    pub async fn into_vec_async(self) -> Result<Vec<T>> {
        if let Context::Host = self.context() {
            self.into_vec()
        } else {
            self.data.__as_slice().raw.to_vec_async().await
        }
    }
}

/*
impl Slice<'_, u8> {
    pub fn bytes_of<T: Pod>(slice: Slice<T>) -> Self {
        todo!()
    }
}

impl SliceMut<'_, u8> {
    pub fn bytes_of_mut<T: Pod>(slice: SliceMut<T>) -> Self {
        todo!()
    }
}
*/

pub struct Zip<T = ()>(pub T);

impl Default for Zip {
    fn default() -> Self {
        Self(())
    }
}

impl<T> Zip<(T,)> {
    pub fn new(x: T) -> Self {
        Self((x,))
    }
}

impl<A> Zip<(A,)> {
    pub fn and<T>(self, x: T) -> Zip<(A, T)> {
        let Self((a,)) = self;
        Zip((a, x))
    }
}

impl<A, B> Zip<(A, B)> {
    pub fn and<T>(self, x: T) -> Zip<(A, B, T)> {
        let Self((a, b)) = self;
        Zip((a, b, x))
    }
}

impl<A, B> Zip<(A, B)> {
    fn try_from<A1, B1>(input: Zip<(A1, B1)>) -> Result<Self, A::Error>
    where
        A: TryFrom<A1>,
        B: TryFrom<B1, Error = A::Error>,
    {
        let Zip((a1, b1)) = input;
        Ok(Self((A::try_from(a1)?, B::try_from(b1)?)))
    }
}

impl<'a, A: Copy, B> Zip<(&'a [A], &'a mut [B])> {
    pub fn for_each<F: FnMut(A, &mut B)>(self, mut f: F) {
        let Self((a, b)) = self;
        a.iter()
            .copied()
            .zip(b.iter_mut())
            .for_each(move |(a, b)| f(a, b));
    }
    pub fn par_for_each<F: Fn(A, &mut B) + Send + Sync>(self, threads: usize, f: F)
    where
        A: Send + Sync,
        B: Send + Sync,
    {
        if threads == 1 {
            self.for_each(f);
            return;
        }
        let Self((a, b)) = self;
        a.par_iter()
            .zip(b.par_iter_mut())
            .for_each(|(a, b)| f(*a, b));
    }
}
