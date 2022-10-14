use core::future::Future;

pub trait BlockableFuture: Future {
    fn block(self) -> Self::Output;
}

impl<T> BlockableFuture for T
where
    T: Future,
{
    fn block(self) -> Self::Output {
        blocker::block(self)
    }
}
