[![LicenseBadge]][License]
[![DocsBadge]][Docs]
[![Build Status](https://github.com/charles-r-earp/autograph/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/charles-r-earp/krnl/actions)

[License]: https://github.com/charles-r-earp/krnl/blob/main/LICENSE-APACHE
[LicenseBadge]: https://img.shields.io/badge/license-MIT/Apache_2.0-blue.svg

[Docs]: https://docs.rs/krnl
[DocsBadge]: https://docs.rs/krnl/badge.svg

# **krnl**
Accelerated compute in Rust. 

## Components
- **krnl**: Compute library
    - Can be used like CUDA / OpenCL 
    - High level code can be written abstractly and run on multiple targets. 
    - Host
        - Operations can be implemented natively with Vecs and slices. 
    - Device
        - Enabled with "device" feature.
        - Targets Vulkan 1.2
        - MacOS / iOS supported via [MoltenVK](https://github.com/KhronosGroup/MoltenVK).
            - This is untested.
        - Kernels 
            - Similar programming model to CUDA / OpenCL.
            - Up to 3 dimensional groups / threads.
            - Strongly typed.
            - Can be dispatched without unsafe.
            - Item kernels for iterator patterns.
            - SpecConstants that can be set at rutime. 
            - Group (shared) memory (can be specialized).
            - SPIR-V assembly for additional functions not implemented in [spirv-std](https://github.com/EmbarkStudios/rust-gpu). 
        - Queues
            - Creation, submission, and synchronization is handled automatically.
            - Creates 1 or more compute queues based on device.
            - Uses a dedicated transfer queue if available.
    - Buffers 
        - If possible, can map memory directly to copy data without a queue. 
        - Slices, Arc, and Cow storage similar to [ndarray](https://github.com/rust-ndarray/ndarray)'s ArrayBase
        - ScalarBuffers for dynamic typing. 
        - Support all basic numerical types u8 to f64, as well as [half](https://github.com/starkat99/half-rs)'s f16 and bf16.  
    - Supports WebAssembly (wasm32)
        - Device functionality not implemented
- **krnl-macros**: Macros for inline kernels.
- **krnl-core**: Shared core library for host / device.
- **krnlc**: Extracts and compiles kernel modules.
    - Built on [RustGPU](https://github.com/EmbarkStudios/rust-gpu)'s spirv-builder.
    - Supports dependencies defined in Cargo.toml. 
    - Uses [spirv-tools](https://github.com/EmbarkStudios/spirv-tools-rs) to validate and optimize. 
    - Run prior to compiling host crate, caches the result in "krnl-cache.rs".
    - Can check that the cache is up to date. 
    - Downstream dependencies do not depend on **krnlc**. 
        - This means that **krnl** and any crates using it can run on stable Rust. 

# Example
Here is a basic example for computing saxpy `y[i] += alpha * x[i]`:  
```
use krnl::{
    anyhow::Result,
    buffer::{Buffer, Slice, SliceMut},
    device::Device,
    macros::module,
};
use std::env::var;

fn main() -> Result<()> {
    // inputs
    let x = vec![1f32];
    let alpha = 2f32;
    let y = vec![0f32];
    // Device can be created with an index, defaults to 0
    let device = Device::builder().build().ok().unwrap_or(Device::host());
    // Buffer: From<Vec> is zero cost
    // Buffer::into_device is zero cost for host
    let x = Buffer::from(x).into_device(device.clone())?;
    let mut y = Buffer::from(y).into_device(device.clone())?;
    saxpy(x.as_slice(), alpha, y.as_slice_mut())?;
    // Buffer::into_vec is zero cost if not on the device
    let y = y.into_vec()?;
    println!("{y:?}");
    Ok(())
}

// Saxpy implemented for both host and device
fn saxpy(x: Slice<f32>, alpha: f32, mut y: SliceMut<f32>) -> Result<()> {
    if let Some((x, y)) = x.as_host_slice().zip(y.as_host_slice_mut()) {
        // Iterators are idiomatic for this kind of operation
        for (x, y) in x.iter().copied().zip(y) {
            kernels::saxpy_impl(x, alpha, y);
        }
        Ok(())
    } else if var("global").is_err() {
        // Item kernels are the device equivalent
        // Builder is cached via once_cell
        kernels::saxpy::builder()?
            // Kernels are cached in the device
            .build(y.device())?
            // if not provided global threads are inferred
            // Dispatch is unsafe if declaration is unsafe
            .dispatch(x, alpha, y)
    } else {
        kernels::saxpy_global::builder()?
            .build(y.device())?
            // Can also specify thread groups
            .with_global_threads(x.len().min(y.len()).try_into().unwrap())
            .dispatch(x, alpha, y)
    }
}

// Shared scope
// Code inside the module will be extracted by krnlc
// and compiled in one crate for the spirv arch
// Changes to the module will trigger a compile error
#[module]
mod kernels {
    // use cfg to filter for host / device code
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    // This can be shared with host code
    pub fn saxpy_impl(x: f32, alpha: f32, y: &mut f32) {
        *y += alpha * x;
    }

    // Item kernels do not require unsafe
    #[kernel(threads(256))]
    pub fn saxpy(#[item] x: f32, alpha: f32, #[item] y: &mut f32) {
        saxpy_impl(x, alpha, y);
    }

    // Saxpy can be implemented directly
    #[kernel(threads(256))]
    pub fn saxpy_global(#[global] x: Slice<f32>, alpha: f32, #[global] y: UnsafeSlice<f32>) {
        // Checked indexing, but from multiple threads
        use krnl_core::buffer::UnsafeIndex;
        // kernel is a hidden argument with methods like global_id, groups, threads etc
        let mut index = kernel.global_index() as usize;
        while index < x.len().min(y.len()) {
            // Access to y is shared mutation and is unsafe
            saxpy_impl(x[index], alpha, unsafe { y.unsafe_index_mut(index) });
            index += kernel.global_threads() as usize;
        }
    }
}
```

# Recent Changes 
See [Releases.md](https://github.com/charles-r-earp/krnl/blob/main/Releases.md)

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.

# Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
