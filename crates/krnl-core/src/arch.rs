#[cfg(not(target_arch = "spirv"))]
use core::marker::PhantomData;
#[cfg(target_arch = "spirv")]
use core::ops::Index;
#[cfg(target_arch = "spirv")]
use spirv_std::arch::IndexUnchecked;

pub trait Length {
    fn len(&self) -> usize;
}

impl<T> Length for [T] {
    fn len(&self) -> usize {
        Self::len(self)
    }
}

pub trait UnsafeIndexMut<I> {
    type Output;
    unsafe fn unsafe_index_mut(&self, index: I) -> &mut Self::Output;
}

#[cfg(target_arch = "spirv")]
fn debug_index_out_of_bounds(index: usize, len: usize) {
    unsafe {
        spirv_std::macros::debug_printfln!(
            "index out of bounds: the len is %x but the index is %x",
            index as u32,
            len as u32,
        );
    }
}

#[cfg(target_arch = "spirv")]
trait IndexUncheckedMutExt<T> {
    unsafe fn index_unchecked_mut_ext(&self, index: usize) -> &mut T;
}

#[cfg(target_arch = "spirv")]
impl<T> IndexUncheckedMutExt<T> for [T] {
    unsafe fn index_unchecked_mut_ext(&self, index: usize) -> &mut T {
        // https://docs.rs/spirv-std/0.5.0/src/spirv_std/arch.rs.html#237-248
        unsafe {
            ::core::arch::asm!(
                "%slice_ptr = OpLoad _ {slice_ptr_ptr}",
                "%data_ptr = OpCompositeExtract _ %slice_ptr 0",
                "%val_ptr = OpAccessChain _ %data_ptr {index}",
                "OpReturnValue %val_ptr",
                slice_ptr_ptr = in(reg) &self,
                index = in(reg) index,
                options(noreturn)
            )
        }
    }
}

#[doc(hidden)]
#[cfg(target_arch = "spirv")]
pub struct Slice<'a, T> {
    inner: &'a [T],
    offset: usize,
    len: usize,
}

#[cfg(target_arch = "spirv")]
impl<'a, T> Slice<'a, T> {
    pub unsafe fn from_raw_parts(inner: &'a [T], offset: usize, len: usize) -> Self {
        Self { inner, offset, len }
    }
}

#[cfg(target_arch = "spirv")]
impl<T> Length for Slice<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(target_arch = "spirv")]
impl<T> Index<usize> for Slice<'_, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        if index < self.len {
            unsafe { self.inner.index_unchecked(self.offset + index) }
        } else {
            debug_index_out_of_bounds(index, self.len);
            panic!();
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
#[derive(Copy, Clone)]
pub struct UnsafeSliceMut<'a, T> {
    ptr: *mut T,
    len: usize,
    _m: PhantomData<&'a ()>,
}

#[cfg(target_arch = "spirv")]
pub struct UnsafeSliceMut<'a, T> {
    inner: &'a mut [T],
    offset: usize,
    len: usize,
}

#[cfg(not(target_ach = "spirv"))]
unsafe impl<T: Send> Send for UnsafeSliceMut<'_, T> {}
#[cfg(not(target_ach = "spirv"))]
unsafe impl<T: Sync> Sync for UnsafeSliceMut<'_, T> {}

#[cfg(not(target_arch = "spirv"))]
impl<'a, T> From<&'a mut [T]> for UnsafeSliceMut<'a, T> {
    fn from(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _m: PhantomData::default(),
        }
    }
}

#[cfg(target_arch = "spirv")]
impl<'a, T> UnsafeSliceMut<'a, T> {
    pub unsafe fn from_raw_parts(inner: &'a mut [T], offset: usize, len: usize) -> Self {
        Self { inner, offset, len }
    }
}

impl<T> Length for UnsafeSliceMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> UnsafeIndexMut<usize> for UnsafeSliceMut<'_, T> {
    type Output = T;
    #[cfg(not(target_arch = "spirv"))]
    unsafe fn unsafe_index_mut(&self, index: usize) -> &mut Self::Output {
        if index < self.len {
            unsafe { &mut *self.ptr.add(index) }
        } else {
            panic!(
                "index out of bounds: the len is {index} but the index is {len}",
                len = self.len
            );
        }
    }
    #[cfg(target_arch = "spirv")]
    unsafe fn unsafe_index_mut(&self, index: usize) -> &mut Self::Output {
        if index < self.len {
            unsafe { self.inner.index_unchecked_mut_ext(self.offset + index) }
        } else {
            debug_index_out_of_bounds(index, self.len);
            panic!();
        }
    }
}
