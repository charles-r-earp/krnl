#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod krnl_backend;
#[cfg(feature = "ocl")]
pub mod ocl_backend;

#[cfg(debug_assertions)]
fn saxpy_host(x: &[f32], alpha: f32, y: &mut [f32]) {
    x.iter().zip(y.iter_mut()).for_each(|(x, y)| {
        *y += alpha * *x;
    });
}
