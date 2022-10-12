pub struct GlobalMut<'a, T: ?Sized>(&'a mut T);

#[doc(hidden)]
pub fn __global_mut(x: &mut T) -> GlobalMut<T> {
    GlobalMut(x)
}

impl<'a, T: ?Sized> GlobalMut<'a, T> {
    pub unsafe fn global_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> GlobalMut<'_, [T]> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub struct GroupUninitMut<'a, T: ?Sized>(&'a mut T);

#[doc(hidden)]
pub fn __group_uninit_mut(x: &mut T) -> GroupUninitMut<T> {
    GroupUninitMut(x)
}

impl<'a, T: ?Sized> GroupUninitMut<'a, T> {
    pub unsafe fn group_uninit_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<'a, T, A: AsRef<[T]>> GroupUninitMut<A> {
    pub fn len(&self) -> usize {
        self.0.as_ref().len()
    }
}
