

```rust
#[module(
    vulkan(1, 1),
    dependency("krnl-core", version = "0.0.1"),
)]
mod axpy {
    use krnl_core::scalar::Scalar;

    pub fn axpy<T: Scalar>(x: &T, alpha: T, y: &mut T) {
        *y += alpha * *x;
    }

    #[kernel(elementwise, threads(256))] pub fn axpy_f32(x: &f32, alpha: f32, y: &mut f32) {
        axpy(x, alpha, y);
    }
}

fn axpy<T: Scalar>(x: &Slice<T>, alpha: T, y: &mut SliceMut<T>) -> Result<()> {
    if y.device() == Device::host() {
        x.as_host_slice()?;
            .iter()
            .zip(y.as_host_slice_mut()?)
            .for_each(|(x, alpha, y)| axpy::axpy(x, alpha, y));
    } else {
        let scalar_type = T::scalar_type();
        axpy::module()?
            .kernel(format!("axpy_{}", scalar_type.name()))?
            .compile_builder()
            .compile(x.device())?
            .dispatch_builder()
            .slice("x", x)
            .push("alpha", alpha)
            .slice_mut("y", y)
            .build()?
            .dispatch()?;
    }
    Ok(())
}
```
