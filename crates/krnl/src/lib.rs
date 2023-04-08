/*!
# **krnl**
Safe, portable compute.

# Installing
For device functionality (kernels), install Vulkan for your platform.
- For development, it's recomended to install the [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/), which includes additional tools:
    - vulkaninfo
     - Validation layers
    - vkconfig
        - This can be used for valiation and with debug_printf.
    - spirv-tools
        - This is used by **krnlc** for validation and optimization.

## Test
- Check that `vulkaninfo --summary` shows your devices.
    - Instance version should be >= 1.2.
- Clone [**krnl**](https://github.com/charles-r-earp/krnl)
    - Check that `cargo test --test integration_tests -- --exact none` shows your devices.
    - Alternatively run all the tests with `cargo test`.

## **krnlc**
Kernels are compiled with **krnlc**.

**krnlc** requires a specific nightly toolchain (it will be installed automatically).
With spirv-tools installed as above:
```text
    cargo install krnlc --git https://github.com/charles-r-earp/krnl
        --no-default-features --features use-installed-tools
```
Otherwise:
```text
    cargo install krnlc --git https://github.com/charles-r-earp/krnl
```
# Kernels
Kernels are functions that can be executed repeatedly and or concurrently. For example:
```
fn fill_ones_impl(index: usize, y: &mut [u32]) {
    y[index] = 1;
}

fn ones(len: usize) -> Vec<u32> {
    let mut y = vec![0; len];
    for index in 0 .. y.len() {
        fill_ones_impl(index, y.as_mut_slice());
    }
    y
}
```
This can be executed concurrently with [`rayon`](https://docs.rs/rayon/latest/rayon/).
```
use krnl_core::buffer::{UnsafeSlice, UnsafeIndex};

unsafe fn fill_ones_impl(index: usize, y: UnsafeSlice<u32>) {
    unsafe {
        *y.unsafe_index_mut(index);
    }
}

fn ones(len: usize) -> Vec<u32> {
    let mut y = vec![0; len];
    {
        let y = UnsafeSlice::from(y.as_mut_slice());
        rayon::broadcast(|context| {
            unsafe {
                fill_ones_impl(context.index(), y);
            }
        });
    }
    y
}
```
To execute on a device, implement a kernel like this:
```no_run
use krnl::macros::module;

#[module]
# #[krnl(no_build)]
mod kernels {
    // The device crate will be linked to krnl-core.
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    use krnl_core::buffer::{UnsafeSlice, UnsafeIndex};

    pub unsafe fn fill_ones_impl(index: usize, y: UnsafeSlice<u32>) {
        unsafe {
            *y.unsafe_index_mut(index);
        }
    }

    #[kernel(threads(256) /* per group. More on that later. */)]
    pub fn fill_ones(#[global] y: UnsafeSlice<u32>) {
        let index = kernel.global_index() as usize;
        unsafe {
            fill_ones_impl(index, y);
        }
    }
}

use krnl::{anyhow::Result, device::Device, buffer::Buffer, krnl_core::buffer::UnsafeSlice};

pub fn fill_ones(device: Device, len: usize) -> Result<Buffer<u32>> {
    let mut y = Buffer::zeros(device.clone(), len)?;
    if let Some(y) = y.as_host_slice_mut() {
        let y = UnsafeSlice::from(y);
        rayon::broadcast(|context| unsafe {
            kernels::fill_ones_impl(context.index(), y);
        });
    } else {
        kernels::fill_ones::builder()?
            .build(y.device())?
            .with_global_threads(y.len().try_into().unwrap())
            .dispatch(y.as_slice_mut())?;
    }
    Ok(y)
}
```
Each kernel is called with a hidden kernel: [`Kernel`](krnl_core::kernel::Kernel) argument. 

### But wait! This is Rust, we can use iterators!

Item kernels are a safe, zero cost abstraction for iterator patterns:
```
use krnl::macros::module;

#[module]
# #[krnl(no_build)]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    pub fn fill_ones_impl(y: &mut u32) {
        *y = 1;
    }

    #[kernel(threads(256))]
    pub fn fill_ones(#[item] y: &mut u32) {
        fill_ones_impl(y);
    }
}

use krnl::{anyhow::Result, device::Device, buffer::Buffer, krnl_core::buffer::UnsafeSlice};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

pub fn fill_ones(device: Device, len: usize) -> Result<Buffer<u32>> {
    let mut y = Buffer::zeros(device.clone(), len)?;
    if let Some(y) = y.as_host_slice_mut() {
        y.par_iter_mut().for_each(kernels::fill_ones_impl);
    } else {
        kernels::fill_ones::builder()?
            .build(y.device())?
            /* global_threads is inferred if not provided */
            .dispatch(y.as_slice_mut())?;
    }
    Ok(y)
}
```
Item kernels are called with a hidden kernel: [`ItemKernel`](krnl_core::kernel::ItemKernel) argument. 

## Documentation 
The generated code from the kernel macro is fully documented. You can view it with `cargo doc --open`.
Also use `--document-private-items` if the item is private. 

## Groups 
Kernels are dispatched in groups of threads (CUDA thread blocks). The threads provided to `#[kernel(threads(..))]` 
sets the number of threads per group. This can be 1, 2, or 3 dimensional, corresponding to [`u32`], [`UVec2`](krnl_core::glam::UVec2),
and [`UVec3`](krnl_core::glam::UVec3). These are x, y, and z dimensions, where z is the outer dimension, and x
is the fastest changing dimension. Thus 1 dimensional kernels have y and z equal to 1.  

For simple kernels, threads can be arbitrary, typically 128, 256, or 512. This can be tuned for optimal performance. 
Note that the device may impose limits on the maximum thread dimensions. This is at least (1024, 1024, 64). 

Threads can be [specialized](#specialization).

Item kernels can infer the global_threads by the sizes of the item arguments. This is functionally equivalent
to `iter().zip()`. 
 
Note that `global_threads = groups * threads`. When provided to `.with_global_threads()` prior to dispatch, global_threads
are rounded up to the next multiple of threads. Because of this, it is typical to check that the global_index is 
in bounds as implemented above. 

Kernels can declare group shared memory:
```no_run
# use krnl::macros::module;
# #[module]
# #[krnl(no_build)]
# mod kernels {
# use krnl::krnl_core;
# use krnl_core::macros::kernel;
#[kernel(threads(64))] 
fn group_sum(
    #[global] x: Slice<f32>,
    #[group] x_group: UnsafeSlice<f32, 64>,
    #[global] y: UnsafeSlice<f32>,
) {
    use krnl_core::{buffer::UnsafeIndex, spirv_std::arch::workgroup_barrier};
        
    let global_id = kernel.global_id() as usize;
    let group_id = kernel.group_id() as usize;
    let thread_id = kernel.thread_id() as usize;
    unsafe {
        x_group.unsafe_index_mut(thread_id) = x[global_id];
        // Barriers are used to synchronize access to group memory.
        // This call must be reached by all threads in the group!
        workgroup_barrier();            
    }
    if thread_id == 0 {
        for i in 0 .. kernel.threads() as usize {
            unsafe {
                y.unsafe_index_mut(group_index) += x_group.unsafe_index(i);
            }
        }   
    }
}  
}
```
Group memory is initialized with T::default().
 
## Subgroups 
Thread groups are composed of subgroups of threads (CUDA warps). Typical values are:
- 32: NVIDIA, Intel  
- 64: AMD

Note that it must be at least 1 and typically is not greater than 128.

## Specialization 
SpecConstants are constants that are set when the kernel is compiled. Threads and group
slice lengths can be specialized. Group slice lengths can also be expressions.
```no_run
# use krnl::{macros::module, anyhow::Result, device::Device};
# #[module]
# #[krnl(no_build)]
# mod kernels {
# use krnl::krnl_core;
# use krnl_core::macros::kernel;
#[kernel(threads(N))] 
pub fn group_sum<const N: u32>(
    #[global] x: Slice<f32>,
    #[group] x_group: UnsafeSlice<f32, { 2 * N as usize }>,
    #[global] y: UnsafeSlice<f32>,
) {
    /* N is available here, but isn't const. */
}
# }
# fn foo(device: Device) -> Result<()> {    
    # use kernels::group_sum;
let kernel = group_sum::builder()?.specialize(128)?.build(device)?;
    # todo!()
# }
```

## Compiling
**krnlc** can read metadata from Cargo.toml:
```toml
[package.metadata.krnlc]
# features to enable when locating modules
features = ["zoom", "zap"] 

[package.metadata.krnlc.dependencies]
# keys are inherited from resolved values for the host target
foo = { version = "0.1.0", features = ["foo"] }
bar = { default-features = false } 

Compile with `krnlc` or `krnlc -p my-crate`:
1. Runs the equivalent of `cargo expand` to locate all modules.
2. Creates a new crate under \<target-dir\>/krnlc/crates/\<my-crate\>.
3. Compiles the device crate with [spirv-builder](https://docs.rs/crate/spirv-builder/latest).
4. Processes the output, validates and optimizes with spirv-tools.
5. Writes out to "krnl-cache.rs".

Note: Can also run with `--check` which will check that the cache is up to date without writing to it.

## debug_printf
The [debug_printf](krnl_core::spirv_std::macros::debug_printfln) and debug_printfln(krnl_core::spirv_std::macros::debug_printfln)
macros can be used to write to stdout. This requires the SPV_KHR_non_semantic_info extension. Use the `--non-semantic-info`
option to **krnlc** to enable this extension. You can use `#[cfg(target_feature = "ext:SPV_KHR_non_semantic_info")]` for 
conditional compilation. 

Non semantic info is not enabled by default because it will significantly increase binary size. 

[`Slice`](krnl_core::buffer::Slice) and [`UnsafeSlice`](krnl_core::buffer::UnsafeSlice) will write out the panic message in
addition to aborting the thread if an index is out of bounds. 

See <https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/debug_printf.md> for usage. 

## Panics 
Panics will abort the thread, but this will not be caught from the host. You can use [debug_printf](#debug_printf) to ensure a 
certain code path is not reached. 

*/

#![forbid(unsafe_op_in_unsafe_fn)]

pub extern crate anyhow;
pub extern crate krnl_core;
pub extern crate krnl_macros as macros;
#[doc(hidden)]
pub extern crate once_cell;

#[doc(inline)]
pub use krnl_core::scalar;

pub mod buffer;
pub mod device;
