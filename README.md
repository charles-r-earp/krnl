[![DocsBadge]][Docs]
[![build](https://github.com/charles-r-earp/krnl/actions/workflows/ci.yaml/badge.svg)](https://github.com/charles-r-earp/krnl/actions/workflows/ci.yaml)

[Docs]: https://docs.rs/krnl
[DocsBadge]: https://docs.rs/krnl/badge.svg

# krnl

Safe, portable, high performance compute (GPGPU) kernels.

Developed for [autograph](https://github.com/charles-r-earp/autograph).

- Similar functionality to CUDA and OpenCL.
- Supports GPU's and other Vulkan 1.2 capable devices.
- MacOS / iOS supported via [MoltenVK](https://github.com/KhronosGroup/MoltenVK).
- Kernels are written inline, entirely in Rust.
  - Simple iterator patterns can be implemented without unsafe.
  - Supports inline [SPIR-V](https://www.khronos.org/spir) assembly.
  - DebugPrintf integration, generates backtraces for panics.
- Buffers on the host can be accessed natively as Vecs and slices.

# krnlc

Kernel compiler for krnl.

- Built on [spirv-builder](https://github.com/EmbarkStudios/rust-gpu/tree/main/crates/spirv-builder).
- Supports dependencies defined in Cargo.toml.
- Uses [spirv-tools](https://github.com/EmbarkStudios/spirv-tools-rs) to validate and optimize.
- Compiles to "krnl-cache.rs", so the crate will build on stable Rust.

See the docs for installation and usage instructions.

# Installing

For device functionality (kernels), install [Vulkan](https://www.vulkan.org) for your platform.

- For development, it's recomended to install the [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/), which includes additional tools:
  - vulkaninfo
  - Validation layers
    - DebugPrintf
  - spirv-tools
    - This is used by krnlc for spirv validation and optimization.
      - krnlc builds by default without needing spirv-tools to be installed.

## Test

- Check that `vulkaninfo --summary` shows your devices.
  - Instance version should be >= 1.2.
- Alternatively, check that `cargo test --test integration_tests -- --exact none` shows your devices.
  - You can run all the tests with `cargo test --all-features`.

# Getting Started

See the [docs](https://docs.rs/krnl) or build them locally with `cargo doc --all-features`.

# Example

```rust
use krnl::{
    macros::module,
    anyhow::Result,
    device::Device,
    buffer::{Buffer, Slice, SliceMut},
};

#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    pub fn saxpy_impl(alpha: f32, x: f32, y: &mut f32) {
        *y += alpha * x;
    }

    // Item kernels for iterator patterns.
    #[kernel]
    pub fn saxpy(alpha: f32, #[item] x: f32, #[item] y: &mut f32) {
        saxpy_impl(alpha, x, y);
    }

    // General purpose kernels like CUDA / OpenCL.
    #[kernel]
    pub fn saxpy_global(alpha: f32, #[global] x: Slice<f32>, #[global] y: UnsafeSlice<f32>) {
        use krnl_core::buffer::UnsafeIndex;

        let global_id = kernel.global_id();
        if global_id < x.len().min(y.len()) {
            saxpy_impl(alpha, x[global_id], unsafe { y.unsafe_index_mut(global_id) });
        }
    }
}

fn saxpy(alpha: f32, x: Slice<f32>, mut y: SliceMut<f32>) -> Result<()> {
    if let Some((x, y)) = x.as_host_slice().zip(y.as_host_slice_mut()) {
        x.iter()
            .copied()
            .zip(y.iter_mut())
            .for_each(|(x, y)| kernels::saxpy_impl(alpha, x, y));
        return Ok(());
    }
    if true {
        kernels::saxpy::builder()?
            .build(y.device())?
            .dispatch(alpha, x, y)
    } else {
        // or
        kernels::saxpy_global::builder()?
            .build(y.device())?
            .with_global_threads(y.len() as u32)
            .dispatch(alpha, x, y)
    }
}

fn main() -> Result<()> {
    let x = vec![1f32];
    let alpha = 2f32;
    let y = vec![0f32];
    let device = Device::builder().build().ok().unwrap_or(Device::host());
    let x = Buffer::from(x).into_device(device.clone())?;
    let mut y = Buffer::from(y).into_device(device.clone())?;
    saxpy(alpha, x.as_slice(), y.as_slice_mut())?;
    let y = y.into_vec()?;
    println!("{y:?}");
    Ok(())
}
```

# Performance

_NVIDIA GeForce GTX 1060 with Max-Q Design_

[benches/compute-benches](benches/compute-benches)

## alloc

|                  | `krnl`                     | `cuda`                            | `ocl`                           |
| :--------------- | :------------------------- | :-------------------------------- | :------------------------------ |
| **`1,000,000`**  | `316.90 ns` (✅ **1.00x**) | `112.84 us` (❌ _356.06x slower_) | `495.45 ns` (❌ _1.56x slower_) |
| **`10,000,000`** | `318.15 ns` (✅ **1.00x**) | `1.10 ms` (❌ _3454.98x slower_)  | `506.82 ns` (❌ _1.59x slower_) |
| **`64,000,000`** | `317.56 ns` (✅ **1.00x**) | `6.31 ms` (❌ _19854.77x slower_) | `506.15 ns` (❌ _1.59x slower_) |

## upload

|                  | `krnl`                     | `cuda`                            | `ocl`                           |
| :--------------- | :------------------------- | :-------------------------------- | :------------------------------ |
| **`1,000,000`**  | `332.66 us` (✅ **1.00x**) | `359.18 us` (✅ **1.08x slower**) | `773.51 us` (❌ _2.33x slower_) |
| **`10,000,000`** | `4.83 ms` (✅ **1.00x**)   | `3.69 ms` (✅ **1.31x faster**)   | `8.76 ms` (❌ _1.81x slower_)   |
| **`64,000,000`** | `25.24 ms` (✅ **1.00x**)  | `24.34 ms` (✅ **1.04x faster**)  | `57.02 ms` (❌ _2.26x slower_)  |

## download

|                  | `krnl`                     | `cuda`                            | `ocl`                           |
| :--------------- | :------------------------- | :-------------------------------- | :------------------------------ |
| **`1,000,000`**  | `584.39 us` (✅ **1.00x**) | `447.38 us` (✅ **1.31x faster**) | `20.17 ms` (❌ _34.52x slower_) |
| **`10,000,000`** | `5.67 ms` (✅ **1.00x**)   | `4.03 ms` (✅ **1.41x faster**)   | `20.15 ms` (❌ _3.55x slower_)  |
| **`64,000,000`** | `28.82 ms` (✅ **1.00x**)  | `25.57 ms` (✅ **1.13x faster**)  | `37.01 ms` (❌ _1.28x slower_)  |

## zero

|                  | `krnl`                     | `cuda`                            | `ocl`                             |
| :--------------- | :------------------------- | :-------------------------------- | :-------------------------------- |
| **`1,000,000`**  | `38.15 us` (✅ **1.00x**)  | `25.28 us` (✅ **1.51x faster**)  | `34.12 us` (✅ **1.12x faster**)  |
| **`10,000,000`** | `250.90 us` (✅ **1.00x**) | `242.95 us` (✅ **1.03x faster**) | `251.86 us` (✅ **1.00x slower**) |
| **`64,000,000`** | `1.53 ms` (✅ **1.00x**)   | `1.55 ms` (✅ **1.01x slower**)   | `1.56 ms` (✅ **1.02x slower**)   |

## saxpy

|                  | `krnl`                     | `cuda`                            | `ocl`                             |
| :--------------- | :------------------------- | :-------------------------------- | :-------------------------------- |
| **`1,000,000`**  | `90.76 us` (✅ **1.00x**)  | `81.16 us` (✅ **1.12x faster**)  | `88.94 us` (✅ **1.02x faster**)  |
| **`10,000,000`** | `746.92 us` (✅ **1.00x**) | `770.03 us` (✅ **1.03x slower**) | `779.90 us` (✅ **1.04x slower**) |
| **`64,000,000`** | `4.71 ms` (✅ **1.00x**)   | `4.90 ms` (✅ **1.04x slower**)   | `4.91 ms` (✅ **1.04x slower**)   |

# Recent Changes

See [Releases.md](https://github.com/charles-r-earp/krnl/blob/main/Releases.md)

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
