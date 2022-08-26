pub struct GlobalMut<'a, T: ?Sized>(&'a mut T);

impl<'a, T> GlobalMut<'a, T> {
    #[doc(hidden)]
    pub fn __new(x: &'a mut T) -> Self {
        Self(x)
    }
}

impl<'a, T> GlobalMut<'a, T> {
    pub unsafe fn global_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

pub struct GroupUninitMut<'a, T>(&'a mut T);

impl<'a, T> GroupUninitMut<'a, T> {
    #[doc(hidden)]
    pub fn __new(x: &'a mut T) -> Self {
        Self(x)
    }
    pub unsafe fn group_uninit_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
