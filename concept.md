

```rust
#[module(
    dependency("krnl-core", version = "0.0.1"),
)]
mod axpy {
    use krnl_core::scalar::Scalar;

    pub fn axpy_impl<T: Scalar>(x: &T, alpha: T, y: &mut T) {
        *y += alpha * *x;
    }
    #[kernel(elementwise, threads(256))]
    pub fn axpy_f32(x: &f32, alpha: f32, y: &mut f32) {
        axpy_impl(x, alpha, y);
    }
    #[kernel(elementwise, threads(256), capabilities(Int64, Float64))]
    pub fn axpy_f64(x: &f64, alpha: f64, y: &mut f64) {
        axpy_impl(x, alpha, y);
    }
}

// generated

mod axpy {
    pub mod axpy_f32 {
        pub struct Kernel {
            base: KernelBase,
        }
        impl Kernel {
            fn build(device: Device) -> Result<Self> {
                todo!()
            }
            pub fn dispatch(&self, x: Slice<f32>, alpha: f32, y: SliceMut<f32>) -> Result<()> {
                todo!()
            }
        }
        pub fn build(device: Device) -> Result<Kernel> {
            Kernel::build(device)
        }
    }
    macro_rules! __krnl_spirv {
        _ => ( // build == false
            unimplemented!("#[module] has attribute #[krnl(build=false)]")
        );
        (axpy_f32) => ( // for each kernel
            &[..]
        );
    }
}

```

# Steps

Parse module args
Get module body tokens

## krnl_build

Initialize krnl dir
Generate device crate
cargo check on device crate
kernel macro parses kernel, writes out kernel info
Compile for each compile options to get spirvs
Save module to cache

## then

Add spirv functions to module tokens
