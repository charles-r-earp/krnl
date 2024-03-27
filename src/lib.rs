/*!
Safe, portable, high performance compute (GPGPU) kernels.

Developed for [autograph](https://docs.rs/autograph).
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
- Built on [spirv-builder](https://docs.rs/spirv-builder).
- Supports dependencies defined in Cargo.toml.
- Uses [spirv-tools](https://docs.rs/spirv-tools-rs) to validate and optimize.
- Compiles to "krnl-cache.rs", so the crate will build on stable Rust.

See [kernel](kernel#krnlc) for installation and usage instructions.

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
    - You can run all the tests with `cargo test`.
*/
#![cfg_attr(doc_cfg, feature(doc_auto_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// anyhow
pub extern crate anyhow;
/// krnl-core
pub extern crate krnl_core;
/// krnl-macros
pub extern crate krnl_macros as macros;
/// half
pub use krnl_core::half;
/// Numerical types.
#[doc(no_inline)]
pub use krnl_core::scalar;

/// Buffers.
pub mod buffer;
/// Devices.
pub mod device;
/// Kernels.
pub mod kernel;
