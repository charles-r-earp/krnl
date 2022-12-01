pub trait ArrayLength {
    fn length(&self) -> usize;
}

impl<T> ArrayLength for [T] {
    fn length(&self) -> usize {
        self.len()
    }
}

impl<T, const N: usize> ArrayLength for [T; N] {
    fn length(&self) -> usize {
        self.len()
    }
}

#[repr(transparent)]
pub struct UnsafeMut<'a, T: ?Sized> {
    inner: &'a mut T,
}

impl<'a, T: ?Sized> UnsafeMut<'a, T> {
    pub fn from_mut(inner: &'a mut T) -> Self {
        Self { inner }
    }
    pub fn len(&self) -> usize
    where
        T: ArrayLength,
    {
        self.inner.length()
    }
    pub unsafe fn unsafe_ref(&self) -> &T {
        &self.inner
    }
    pub unsafe fn unsafe_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

#[repr(transparent)]
pub struct UninitUnsafeMut<'a, T: ?Sized> {
    inner: UnsafeMut<'a, T>,
}

impl<'a, T: ?Sized> UninitUnsafeMut<'a, T> {
    pub fn from_mut(inner: &'a mut T) -> Self {
        Self {
            inner: UnsafeMut::from_mut(inner),
        }
    }
    pub fn len(&self) -> usize
    where
        T: ArrayLength,
    {
        self.inner.len()
    }
    pub unsafe fn uninit_unsafe_mut(&mut self) -> &mut T {
        unsafe { self.inner.unsafe_mut() }
    }
    pub unsafe fn assume_init(&mut self) -> &mut UnsafeMut<'a, T> {
        &mut self.inner
    }
}
