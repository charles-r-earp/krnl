

```rust
#[module(
    dependency(
        krnl_core(version = "0.1.0", git = "", default-features = false, features = ""),
        num-traits(..),
    ),
)]
mod axpy {
    #[cfg(target_arch = "spirv")]
    use krnl_core::{scalar::Scalar, glam::UVec3, mem::Uninit};

    #[cfg(target_arch = "spirv")]
    fn axpy<T: Scalar>(
        global_threads: UVec3,
        global_id: UVec3,
        x: &[T],
        alpha: T,
        y: &mut [T],
    ) {
        let mut idx = global_id.x;
        while idx as usize < y.len() {
            y[idx as usize] += alpha * x[idx as usize];
            idx += global_threads.x;
        }
    }

    #[kernel(threads(256), capability("storage16_bit_access", "int16"))] fn axpy_bf16(
        x: &[f32],
        alpha: f32,
        y: &mut [f32],
    ) {
        axpy(global_threads, global_id, x, alpha, y);
    }

    #[kernel(threads(256))] fn axpy_f32(
        x: &[f32],
        alpha: f32,
        y: &mut [f32],
    ) {
        axpy(global_threads, global_id, x, alpha, y);
    }

    #[kernel(threads(256))] fn axpy_f64(
        x: &[f64],
        alpha: f64,
        y: &mut [f64],
    ) {
        axpy(global_threads, global_id, x, alpha, y);
    }

    #[kernel(threads(256)] group_ex(
        x: &[f32],
        #[group] x_group: &mut Uninit([f32; 10]),
        y: &mut [f32],
    ) {
        let x_group = unsafe {
            x_group.uninit_mut()
        };
    }

    #[kernel(threads(256)] group_ex(
        #[builtin] global_id: UVec3,
        #[builtin] global_threads: UVec3,
        x: &[f32],
        #[group] x_group: &mut Uninit([f32; 10]),
        y: &mut [f32],
    ) {
        let x_group = unsafe {
            x_group.uninit_mut()
        };
    }
}

fn axpy<T: Scalar>(x: &Slice<T>, alpha: T, y: &mut SliceMut<T>) -> Result<()> {
    let scalar_type = T::scalar_type();
    axpy::module()?
        .kernel(format!("axpy_{}", scalar_type.name()))?
        .build(x.device())?
        .global_threads([y.len()])
        .slice("x", x)
        .push("alpha", alpha)
        .slice_mut("y", y)
        .build()?
        .dispatch()
}
```

```rust
#[shared_module(dependencies(
    krnl-core = "0.0.1",
    glam = "0.22.1",
))]
mod shared {
    use krnl_core::scalar::Scalar;
    use glam::UVec3;

    fn axpy<T: Scalar>(
        global_threads: UVec3,
        global_id: UVec3,
        x: &[T],
        alpha: T,
        y: &mut [T],
    ) {
        let mut idx = global_id.x;
        while idx as usize < y.len() {
            y[idx as usize] += alpha * x[idx as usize];
            idx += global_threads.x;
        }
    }

    #[kernel_module(vulkan("1.1", capabilities(""), extensions(""))]
    pub mod x32 {
        use super::*;

        #[kernel(threads(256))] axpy_f32(
            #[builtin] global_threads: UVec3,
            #[builtin] global_id: UVec3,
            x: &[f32],
            alpha: f32,
            y: &mut [f32],
        ) {
            axpy(global_threads, global_id, x, alpha, y);
        }
    }
}
```
