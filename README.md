[![DocsBadge]][Docs]
[![build](https://github.com/charles-r-earp/krnl/actions/workflows/ci.yaml/badge.svg)](https://github.com/charles-r-earp/krnl/actions/workflows/ci.yaml)

[Docs]: https://docs.rs/krnl
[DocsBadge]: https://docs.rs/krnl/badge.svg

# krnl

Safe, portable, high performance compute (GPGPU) kernels.

Developed for [autograph](https://github.com/charles-r-earp/autograph).

- Similar functionality to CUDA and OpenCL.
- Native GPU acceleration via Vulkan 1.3.
- MacOS / iOS supported via [MoltenVK](https://github.com/KhronosGroup/MoltenVK).
- Web GPU acceleration via [WebGPU](https://gpuweb.github.io/gpuweb/).
- Kernels are written inline, entirely in Rust.
  - Simple iterator patterns can be implemented without unsafe.
  - Supports inline [SPIR-V](https://www.khronos.org/spir) assembly.
  - DebugPrintf integration, generates backtraces for panics.
- Buffers on the host can be accessed natively as Vecs and slices.

# krnlc

Kernel compiler for krnl.

- Invokes [cargo-gpu](https://github.com/Rust-GPU/cargo-gpu) for Rust toolchain management.
- Compiles Rust to SPIR-V via [spirv-builder](https://github.com/Rust-GPU/rust-gpu/tree/main/crates/spirv-builder).
- Uses [spirv-tools](https://github.com/Rust-GPU/spirv-tools-rs) to validate and optimize.
- Compiles to a SPIR-V module ahead of compile time, so that the crate can compile on stable Rust.
- Also functions as a library, supporting lazy compilation via a build script.
- At compile time, generates strongly typed bindings.

See the docs for installation and usage instructions. **under construction**

# Installing

For device functionality (kernels), install [Vulkan](https://www.vulkan.org) for your platform.

- For development, it's recomended to install the [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/), which includes additional tools:
  - vulkaninfo
  - Validation layers
    - DebugPrintf
  - spirv-tools
    - This is used by krnlc for spirv validation and optimization.
      - krnlc builds by default without needing spirv-tools to be installed.
- Check that `vulkaninfo --summary` shows your devices.
  - Instance version should be >= 1.3.

# Test

- You can run all the tests with `cargo test`.

## Web

Tests can be run with wasm-pack.

Install wasm-pack `cargo install wasm-pack`.

wasm-pack must be run in the test crate directory `cd tests/integration`.

WebGPU requires a [supported browser](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status).

Tests can be run with either `--firefox` or `--chrome` flags:
```
wasm-pack test --firefox
```

You should get an output like this:
```
Interactive browsers tests are now available at http://127.0.0.1:8000

Note that interactive mode is enabled because `NO_HEADLESS`
is specified in the environment of this process. Once you're
done with testing you'll need to kill this server with
Ctrl-C.
```

# Getting Started

See the [docs](https://docs.rs/krnl) or build them locally with `cargo doc --all-features`.

# Example

```rust
#![cfg_attr(target_arch = "spirv", no_std)]
#[cfg(not(target_arch = "spirv"))]
use krnl::{
    buffer::Buffer,
    context::{Context, Device},
    kernel::KernelDef,
};
use krnl::{macros::kernel, scalar::Scalar};
use num_traits::{Num, NumAssign};

fn axpy_impl<T: Scalar + Num + NumAssign>(alpha: T, x: T, y: &mut T) {
    *y += alpha * x;
}

#[kernel]
fn axpy<T: Scalar + Num + NumAssign>(alpha: T, #[kernel(item)] x: T, #[kernel(item)] y: &mut T) {
    axpy_impl(alpha, x, y);
}

#[cfg(not(target_arch = "spirv"))]
pub fn main() {
    let context = Context::Device(Device::builder().build().unwrap());
    let alpha = 2f32;
    let x = Buffer::from(vec![1f32])
        .into_context(context.clone())
        .unwrap();
    let mut y = Buffer::zeros(context.clone(), 1).unwrap();

    if let Some((x, y)) = x
        .as_slice()
        .into_host_slice()
        .zip(y.as_slice_mut().into_host_slice_mut())
    {
        for (x, y) in x.iter().copied().zip(y) {
            axpy_impl(alpha, x, y);
        }
    } else {
        axpy::builder(())
            .build(context)
            .unwrap()
            .exec((alpha, x.as_slice(), y.as_slice_mut()))
            .unwrap();
    }

    let y = y.into_vec().unwrap();
    dbg!(y);
}
```

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
