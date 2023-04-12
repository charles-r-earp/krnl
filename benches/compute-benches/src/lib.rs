#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "autograph")]
pub mod autograph_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "device")]
pub mod krnl_backend;
#[cfg(feature = "ocl")]
pub mod ocl_backend;

#[cfg(all(
    any(
        feature = "device",
        feature = "autograph",
        feature = "cuda",
        feature = "ocl"
    ),
    debug_assertions
))]
fn saxpy_host(x: &[f32], alpha: f32, y: &mut [f32]) {
    x.iter().zip(y.iter_mut()).for_each(|(x, y)| {
        *y += alpha * *x;
    });
}
