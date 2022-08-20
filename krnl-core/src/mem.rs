
#[repr(transparent)]
pub struct Uninit<T>(T);

impl<T> Uninit<T> {
    pub unsafe fn uninit_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
