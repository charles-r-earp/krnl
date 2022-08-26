use crate::ops::{IndexUnchecked, IndexUncheckedMut};
use core::ops::{Index, IndexMut};

#[derive(Clone, Copy)]
pub struct Slice<'a, T> {
    inner: &'a [T],
    offset: usize,
    len: usize,
}

impl<'a, T> Slice<'a, T> {
    #[doc(hidden)]
    pub unsafe fn __from_raw_parts(inner: &'a [T], offset: usize, len: usize) -> Self {
        Self { inner, offset, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Index<usize> for Slice<'_, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        assert!(index < self.len);
        unsafe { self.inner.index_unchecked(self.offset + index) }
    }
}

impl<T> IndexUnchecked<usize> for Slice<'_, T> {
    unsafe fn index_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        unsafe { self.inner.index_unchecked(self.offset + index) }
    }
}

pub struct SliceMut<'a, T> {
    inner: &'a mut [T],
    offset: usize,
    len: usize,
}

impl<'a, T> SliceMut<'a, T> {
    #[doc(hidden)]
    pub fn __from_raw_parts_mut(inner: &'a mut [T], offset: usize, len: usize) -> Self {
        Self { inner, offset, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Index<usize> for SliceMut<'_, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        assert!(index < self.len);
        unsafe { self.inner.index_unchecked(self.offset + index) }
    }
}

impl<T> IndexMut<usize> for SliceMut<'_, T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.len);
        unsafe { self.inner.index_unchecked_mut(self.offset + index) }
    }
}

impl<T> IndexUnchecked<usize> for SliceMut<'_, T> {
    unsafe fn index_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        unsafe { self.inner.index_unchecked(self.offset + index) }
    }
}

impl<T> IndexUncheckedMut<usize> for SliceMut<'_, T> {
    unsafe fn index_unchecked_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.len);
        unsafe { self.inner.index_unchecked_mut(self.offset + index) }
    }
}
