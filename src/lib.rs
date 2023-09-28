/*!
# **krnl**
Safe, portable, high performance compute (GPGPU) kernels.

Developed for [**autograph**](https://docs.rs/autograph).
- Similar functionality to CUDA and OpenCL.
- Supports GPU's and other Vulkan 1.2 capable devices.
- MacOS / iOS supported via [MoltenVK](https://github.com/KhronosGroup/MoltenVK).
- Kernels are written inline, entirely in Rust.
- Simple iterator patterns can be implemented without unsafe.
- Buffers on the host can be accessed natively as Vecs and slices.

# **krnlc**
Kernel compiler for **krnl**.
- Built on [RustGPU](https://github.com/EmbarkStudios/rust-gpu)'s spirv-builder.
- Supports dependencies defined in Cargo.toml.
- Uses [spirv-tools](https://github.com/EmbarkStudios/spirv-tools-rs) to validate and optimize.
- Compiles to "krnl-cache.rs", so the crate will build on stable Rust.

# Installing
For device functionality (kernels), install Vulkan for your platform.
- For development, it's recomended to install the [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/), which includes additional tools:
    - vulkaninfo
    - Validation layers
    - spirv-tools
        - This is used by **krnlc** for spirv validation and optimization.
            - **krnlc** builds by default without needing spirv-tools to be installed.

## Test
- Check that `vulkaninfo --summary` shows your devices.
    - Instance version should be >= 1.2.
- Alternatively, clone [**krnl**](https://github.com/charles-r-earp/krnl/releases/tag/v0.0.1).
    - Check that `cargo test --test integration_tests -- --exact none` shows your devices.
    - You can run all the tests with `cargo test`.

# Getting Started
- See [device](device) for creating devices.
- See [buffer](buffer) for creating buffers.
- See [kernel](kernel) for compute kernels.

# Example
```
 use krnl::{
    anyhow::Result,
    buffer::{Buffer, Slice, SliceMut},
    device::Device,
    macros::module,
};

#[module]
# #[krnl(no_build)]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    pub fn saxpy_impl(x: f32, alpha: f32, y: &mut f32) {
        *y += alpha * x;
    }

    // Item kernels for iterator patterns.
    #[kernel]
    pub fn saxpy(#[item] x: f32, alpha: f32, #[item] y: &mut f32) {
        saxpy_impl(x, alpha, y);
    }

    // General purpose kernels like CUDA / OpenCL.
    #[kernel]
    pub fn saxpy_global(#[global] x: Slice<f32>, alpha: f32, #[global] y: UnsafeSlice<f32>) {
        use krnl_core::buffer::UnsafeIndex;
        let mut index = kernel.global_index() as usize;
        while index < x.len().min(y.len()) {
            saxpy_impl(x[index], alpha, unsafe { y.unsafe_index_mut(index) });
            index += kernel.global_threads() as usize;
        }
    }
}

fn saxpy(x: Slice<f32>, alpha: f32, mut y: SliceMut<f32>) -> Result<()> {
    if let Some((x, y)) = x.as_host_slice().zip(y.as_host_slice_mut()) {
        for (x, y) in x.iter().copied().zip(y) {
            kernels::saxpy_impl(x, alpha, y);
        }
        return Ok(());
    }
    kernels::saxpy::builder()?
        .build(y.device())?
        .dispatch(x, alpha, y)
}

fn main() -> Result<()> {
    let x = vec![1f32];
    let alpha = 2f32;
    let y = vec![0f32];
    # if false {
    let device = Device::builder().build().ok().unwrap_or(Device::host());
    # }
    # let device = Device::host();
    let x = Buffer::from(x).into_device(device.clone())?;
    let mut y = Buffer::from(y).into_device(device.clone())?;
    saxpy(x.as_slice(), alpha, y.as_slice_mut())?;
    let y = y.into_vec()?;
    println!("{y:?}");
    Ok(())
}
```
*/

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// anyhow
pub extern crate anyhow;
/// krnl-core
pub extern crate krnl_core;
/// krnl-macros
pub extern crate krnl_macros as macros;
#[doc(hidden)]
pub extern crate once_cell;

/// half
pub use krnl_core::half;

#[doc(inline)]
pub use krnl_core::scalar;

/// Buffers.
pub mod buffer;
/// Devices.
pub mod device;
/// Kernels.
pub mod kernel;
