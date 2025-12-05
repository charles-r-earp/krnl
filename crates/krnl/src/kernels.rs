use crate::{
    self as krnl,
    macros::{host_only, kernel},
    scalar::Scalar,
};
use num_traits::{AsPrimitive, FromPrimitive};
host_only! {
    use bytemuck::Pod;
    use crate::{Result, buffer::{BufferBase, Buffer, Slice, Data, DataMut}, kernel::KernelDef};
}

#[kernel]
pub(crate) fn fill_u8(x: u8, #[kernel(item)] y: &mut u8) {
    *y = x;
}

#[kernel]
pub(crate) fn fill_u16(x: u16, #[kernel(item)] y: &mut u16) {
    *y = x;
}

#[kernel]
pub(crate) fn fill_u32(x: u32, #[kernel(item)] y: &mut u32) {
    *y = x;
}

#[kernel]
pub(crate) fn fill_u32x2(x1: u32, x2: u32, #[kernel(item)] y: &mut [u32; 2]) {
    *y = [x1, x2];
}

#[kernel]
pub fn cast<X: Scalar + AsPrimitive<Y> + AsPrimitive<u32>, Y: Scalar + FromPrimitive>(
    #[kernel(item)] x: X,
    #[kernel(item)] y: &mut Y,
) {
    use krnl::scalar::ScalarType;

    if const { ScalarType::of::<X>() as u32 == ScalarType::of::<u8>() as u32 } {
        let x: u32 = x.as_();
        if const { ScalarType::of::<Y>() as u32 == ScalarType::of::<i16>() as u32 } {
            *y = Y::from_i16(x as i16).unwrap();
            return;
        }
        if const { ScalarType::of::<Y>() as u32 == ScalarType::of::<i32>() as u32 } {
            *y = Y::from_i32(x as i32).unwrap();
            return;
        }
        if const { ScalarType::of::<Y>() as u32 == ScalarType::of::<i64>() as u32 } {
            *y = Y::from_i64(x as i64).unwrap();
            return;
        }
    }

    *y = x.as_();
}

#[kernel]
pub(crate) fn copy_u8(#[kernel(item)] x: u8, #[kernel(item)] y: &mut u8) {
    *y = x;
}

#[kernel]
pub(crate) fn copy_u16(#[kernel(item)] x: u16, #[kernel(item)] y: &mut u16) {
    *y = x;
}

#[kernel]
pub(crate) fn copy_u32(#[kernel(item)] x: u32, #[kernel(item)] y: &mut u32) {
    *y = x;
}

#[kernel]
pub(crate) fn copy_u32x2(#[kernel(item)] x: [u32; 2], #[kernel(item)] y: &mut [u32; 2]) {
    *y = x;
}

host_only! {
    impl<T: Pod, S: DataMut<Elem = T>> BufferBase<S> {
        pub fn fill(&mut self, x: T) -> Result<()> {
            if let Some(y) = self.as_slice_mut().into_host_slice_mut() {
                for y in y {
                    *y = x;
                }
                return Ok(());
            }
            if let Some(y) = self.as_slice_mut().try_bitcast_mut::<[u32; 2]>() {
                let [x1, x2] = if const { size_of::<T>() == 8 } {
                    bytemuck::cast(x)
                } else if const { size_of::<T>() == 4 } {
                    let x: u32 = bytemuck::cast(x);
                    [x; 2]
                } else if const { size_of::<T>() == 2 } {
                    let x: u16 = bytemuck::cast(x);
                    bytemuck::cast([x; 4])
                } else {
                    let x: u8 = bytemuck::cast(x);
                    bytemuck::cast([x; 8])
                };
                fill_u32x2::builder(()).build(y.context())?.exec((x1, x2, y))?;
                Ok(())
            } else if let Some(y) = self.as_slice_mut().try_bitcast_mut::<u32>() {
                let x = if const { size_of::<T>() == 4 } {
                    bytemuck::cast(x)
                } else if const { size_of::<T>() == 2 } {
                    let x: u16 = bytemuck::cast(x);
                    bytemuck::cast([x; 2])
                } else {
                    let x: u8 = bytemuck::cast(x);
                    bytemuck::cast([x; 4])
                };
                fill_u32::builder(()).build(y.context())?.exec((x, y))?;
                Ok(())
            } else if let Some(y) = self.as_slice_mut().try_bitcast_mut::<u16>() {
                let x: u16 = if const { size_of::<T>() == 2 } {
                    bytemuck::cast(x)
                } else {
                    let x: u8 = bytemuck::cast(x);
                    bytemuck::cast([x; 2])
                };
                fill_u16::builder(()).build(y.context())?.exec((x, y))?;
                Ok(())
            } else {
                let y = self.as_slice_mut().try_bitcast_mut::<u8>().unwrap();
                let x: u8 = bytemuck::cast(x);
                fill_u8::builder(()).build(y.context())?.exec((x, y))?;
                Ok(())
            }
        }
        pub fn copy_from_slice(&mut self, slice: Slice<T>) -> Result<()> {
            if self.context() != slice.context() || self.context().is_host() {
                return self.as_slice_mut().as_context_slice_mut().copy_from_slice(slice.as_context_slice().clone());
            }
            if let Some((x, y)) = slice.clone().try_bitcast::<[u32; 2]>().zip(self.as_slice_mut().try_bitcast_mut::<[u32; 2]>()) {
                copy_u32x2::builder(()).build(y.context())?.exec((x, y))
            } else if let Some((x, y)) = slice.clone().try_bitcast::<u32>().zip(self.as_slice_mut().try_bitcast_mut::<u32>()) {
                copy_u32::builder(()).build(y.context())?.exec((x, y))
            } else if let Some((x, y)) = slice.clone().try_bitcast::<u16>().zip(self.as_slice_mut().try_bitcast_mut::<u16>()) {
                copy_u16::builder(()).build(y.context())?.exec((x, y))
            } else {
                let x = slice.try_bitcast::<u8>().unwrap();
                let y = self.as_slice_mut().try_bitcast_mut::<u8>().unwrap();
                copy_u8::builder(()).build(y.context())?.exec((x, y))
            }
        }
    }

    impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
        pub fn cast<Y: Scalar + FromPrimitive>(self) -> Result<Buffer<Y>>
        where
            T: AsPrimitive<Y> + AsPrimitive<u32>,
        {
            let x = self.as_slice();
            if let Some(x) = x.as_slice().into_host_slice() {
                let y: Vec<Y> = x.iter().copied().map(|x| x.as_()).collect();
                return Ok(y.into());
            }
            let mut y = unsafe { Buffer::uninit(self.context(), self.len())? };
            cast::builder(())
                .build(y.context())?
                .exec((x, y.as_slice_mut()))?;
            Ok(y)
        }
    }
}
