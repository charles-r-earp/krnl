use spirv_std::arch::IndexUnchecked;
use std::ops::Index;

pub trait IndexOr<I>: Index<I> {
    fn index_or<'a>(&'a self, index: I, default: &'a Self::Output) -> &'a Self::Output;
}

impl<T> IndexOr<usize> for [T] {
    fn index_or<'a>(&'a self, index: usize, default: &'a Self::Output) -> &'a T {
        if index < self.len() {
            unsafe { self.index_unchecked(index) }
        } else {
            default
        }
    }
}

pub trait IndexMutOr<I>: IndexOr<I> {
    fn index_mut_or<'a>(
        &'a mut self,
        index: I,
        default: &'a mut Self::Output,
    ) -> &'a mut Self::Output;
}

impl<T> IndexMutOr<usize> for [T] {
    fn index_mut_or<'a>(&'a mut self, index: usize, default: &'a mut Self::Output) -> &'a mut T {
        if index < self.len() {
            unsafe { self.index_unchecked_mut(index) }
        } else {
            default
        }
    }
}

pub struct UnsafeMut<T>(T);

impl<'a, T> UnsafeMut<&'_ mut [T]> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub unsafe fn unsafe_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T, const N: usize> UnsafeMut<&'_ mut [T; N]> {
    pub fn len(&self) -> usize {
        N
    }
    pub unsafe fn unsafe_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<T> From<T> for UnsafeMut<T> {
    fn from(inner: T) -> Self {
        Self(inner)
    }
}

#[test]
fn foo() {
    let mut y = vec![1, 2, 3];
    let mut y = UnsafeMut::from(y.as_mut_slice());
    unsafe {
        assert_eq!(y.unsafe_mut()[0], 1);
    }
}
