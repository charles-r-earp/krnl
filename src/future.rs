use core::future::Future;
use std::sync::Arc;

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
