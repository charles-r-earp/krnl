use crate::{
    self as krnl,
    macros::{host_only, kernel},
    scalar::Scalar,
};
use num_traits::AsPrimitive;
host_only! {
    use crate::{Result, scalar::Element, buffer::{BufferBase, Buffer, Data, DataMut}, kernel::KernelDef};
}

#[kernel]
pub fn fill_u8(x: u8, #[kernel(item)] y: &mut u8) {
    *y = x;
}

#[kernel]
pub fn fill_u16(x: u16, #[kernel(item)] y: &mut u16) {
    *y = x;
}

#[kernel]
pub fn fill_u32(x: u32, #[kernel(item)] y: &mut u32) {
    *y = x;
}

#[kernel]
pub fn fill_u32x2(x: [u32; 2], #[kernel(item)] y: &mut [u32; 2]) {
    *y = x;
}

#[kernel]
pub fn cast<X: Scalar + AsPrimitive<Y>, Y: Scalar>(
    #[kernel(item)] x: X,
    #[kernel(item)] y: &mut Y,
) {
    *y = x.as_();
}

host_only! {
    impl<T: Element, S: DataMut<Elem = T>> BufferBase<S> {
        pub fn fill(&mut self, x: T) -> Result<()> {
            if let Some(y) = self.as_slice_mut().into_host_slice_mut() {
                for y in y {
                    *y = x;
                }
                return Ok(());
            }
            if let Some(y) = self.as_slice_mut().try_bitcast_mut::<[u32; 2]>() {
                let x = if const { size_of::<T>() == 8 } {
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
                fill_u32x2::builder(()).build(y.context())?.exec((x, y))?;
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
    }

    impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
        pub fn cast<Y: Scalar>(self) -> Result<Buffer<Y>>
        where
            T: AsPrimitive<Y>,
        {
            let x = self.as_slice();
            let mut y = unsafe { Buffer::uninit(self.context(), self.len())? };
            cast::builder(())
                .build(y.context())?
                .exec((x, y.as_slice_mut()))?;
            Ok(y)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        buffer::{Buffer, Slice},
        context::{Context, Device},
    };

    #[cfg(feature = "device")]
    #[test]
    fn fill_device() {
        let context = Context::Device(Device::builder().build().unwrap());
        let n = 10;
        let mut y = unsafe { Buffer::uninit(context, n).unwrap() };
        y.fill(1).unwrap();
        let y = y.into_vec().unwrap();
        assert_eq!(y, vec![1u32; n]);
    }

    #[cfg(feature = "device")]
    #[test]
    fn zeros_device() {
        let context = Context::Device(Device::builder().build().unwrap());
        let n = 10;
        let y = Buffer::<f32>::zeros(context, n).unwrap();
        let y = y.into_vec().unwrap();
        assert_eq!(y, vec![0f32; n]);
    }

    #[cfg(feature = "device")]
    #[test]
    fn cast_u32_f32_device() {
        let context = Context::Device(Device::builder().build().unwrap());
        let n = 10;
        let x_vec: Vec<u32> = (1..=n).map(|x| x as u32).collect();
        let y_vec: Vec<f32> = x_vec.iter().copied().map(|x| x as f32).collect();
        let x = Slice::from(x_vec.as_slice())
            .into_context(context.clone())
            .unwrap();
        let y = x.cast::<f32>().unwrap();
        let y = y.into_vec().unwrap();
        assert_eq!(y, y_vec);
    }
}
