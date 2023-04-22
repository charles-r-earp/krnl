/*!

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
            *y.unsafe_index_mut(index) = 1;
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

## *But wait! This is Rust, we can use iterators!*
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

# Kernels
The kernel macro generates a module like this:
```text

pub mod fill_ones {
    /// Builder for creating a [`Kernel`].
    ///
    /// See [`builder()`](builder).
    pub struct KernelBuilder { /* .. */ }

    /// Creates a builder.
    ///
    /// The builder is lazily created on first call.
    ///
    /// **errors**
    /// - The kernel wasn't compiled (with `#[krnl(no_build)]` applied to `#[module]`).
    /// - The kernel could not be deserialized. For stable releases, this is a bug, as `#[module]` should produce a compile error.
    pub fn builder() -> Result<KernelBuilder>;

    impl KernelBuilder {
        pub fn build(&self, device: Device) -> Result<Kernel>;
    }

    /// Kernel.
    pub struct Kernel { /* .. */ }

    impl Kernel {
        /// Global threads to dispatch.
        ///
        /// Implicitly declares groups by rounding up to the next multiple of threads.
        pub fn with_global_threads(self, global_threads: u32) -> Self;
        /// Groups to dispatch.
        ///
        /// For item kernels, if not provided, is inferred based on item arguments.
        pub fn with_groups(self, groups: u32) -> Self;
        /// Dispatches the kernel.
        ///
        /// - Waits for immutable access to slice arguments.
        /// - Waits for mutable access to mutable slice arguments.
        /// - Blocks until the kernel is queued.
        ///
        /// A device has 1 or more compute queues. One kernel can be queued while another is
        /// executing on that queue.
        ///
        /// **errors**
        /// - DeviceLost: The device was lost.
        /// - The kernel could not be queued.
        pub fn dispatch(&self, y: SliceMut<u32>) -> Result<()>;
    }
}
```

View the generated code and documentation with `cargo doc --open`.
Also use `--document-private-items` if the item is private.

# Modules
The module macro supports additional arguments within `krnl(..)`:
```no_run
# use krnl::macros::module;
# mod path { pub(super) mod to { pub(crate) use krnl; } }
#[module]
#[krnl(crate=path::to::krnl)]
#[krnl(no_build)] // Will skip compiling, used for krnl's docs.
pub mod foo {
    use super::*;
}
```

Modules mut be within a module hierarchy, not within fn's or impl blocks.

# Macros
Kernels can be generated via macro_rules! and procedural macros. For example, [dry](https://docs.rs/dry/latest/dry/) and [paste](https://docs.rs/paste/latest/paste/)
can be very useful:
```
# use krnl::macros::module;
# #[module] #[krnl(no_build)] mod kernels {
# use krnl::macros::kernel;
use dry::macro_for;
use paste::paste;
macro_for!($T in [i32, u32, f32] {
    paste! {
        #[kernel(threads(128))]
        fn [<add_ $T>](
            #[item] a: $T,
            #[item] b: $T,
            #[item] c: &mut $T,
        ) {
            *c = a + b;
        }
    }
});
# }

```



# Groups
Kernels are dispatched in groups of threads (CUDA thread blocks). The threads provided to `#[kernel(threads(..))]`
sets the number of threads per group. This can be 1, 2, or 3 dimensional, corresponding to [`u32`], [`UVec2`](krnl_core::glam::UVec2),
and [`UVec3`](krnl_core::glam::UVec3). These are x, y, and z dimensions, where z is the outer dimension, and x
is the fastest changing dimension. Thus 1 dimensional kernels have y and z equal to 1.

For simple kernels, threads can be arbitrary, typically 128, 256, or 512. This can be tuned for optimal performance.
Note that the device may impose limits on the maximum thread dimensions.

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
    use krnl_core::{buffer::UnsafeIndex, spirv_std::arch::workgroup_memory_barrier};

    let global_id = kernel.global_id() as usize;
    let group_id = kernel.group_id() as usize;
    let thread_id = kernel.thread_id() as usize;
    unsafe {
        x_group.unsafe_index_mut(thread_id) = x[global_id];
        // Barriers are used to synchronize access to group memory.
        // This call must be reached by all threads in the group!
        workgroup_memory_barrier();
    }
    if thread_id == 0 {
        for i in 0 .. kernel.threads() as usize {
            unsafe {
                y.unsafe_index_mut(group_index) += x_group.unsafe_index(i);
            }
        }
    }
}
# }
```
Group memory is zeroed.

# Subgroups
Thread groups are composed of subgroups of threads (CUDA warps). Typical [`.subgroup_threads()`](krnl_core::kernel::Kernel::subgroup_threads) are:
- 32: NVIDIA, Intel
- 64: AMD

Note that it must be at least 1 and typically is not greater than 128.

# Features
Kernels implicitly declare [`Features`](device::Features) based on types and or operations used.
If the [device](device::Device) does not support these features, `.build()` will return an
error.

See [`DeviceInfo::features`](device::DeviceInfo::features).

# Specialization
SpecConstants are constants that are set when the kernel is compiled. Threads and the length
of group buffers can be specialized:
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
Specialization will return an [`Err`] if a thread dimension is 0.

# Panics
Panics will abort the thread, but this will not be caught from the host. You can use [debug_printf](#debug_printf) to ensure a
certain code path is not reached.

# debug_printf
The [debug_printf](krnl_core::spirv_std::macros::debug_printfln) and [debug_printfln](krnl_core::spirv_std::macros::debug_printfln)
macros can be used to write to stdout. This requires the SPV_KHR_non_semantic_info extension. Use the `--non-semantic-info`
option to [**krnlc**](#compiling) to enable this extension. You can use `#[cfg(target_feature = "ext:SPV_KHR_non_semantic_info")]` for
conditional compilation.

Non semantic info is not enabled by default because it will significantly increase binary size.

[`Slice`](krnl_core::buffer::Slice) and [`UnsafeSlice`](krnl_core::buffer::UnsafeSlice) will write out the panic message in
addition to aborting the thread if an index is out of bounds.

See <https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/debug_printf.md> for usage.

# Compiling
Kernels are compiled with **krnlc**.

## Toolchains
To locate modules, **krnlc** will use the nightly toolchain. Install it with:
```text
rustup toolchain install nightly
```
To compile kernels with [spirv-builder](https://docs.rs/crate/spirv-builder), a specific nightly is required:
```text
rustup component add --toolchain nightly-2023-01-21 rust-src rustc-dev llvm-tools-preview
```

## Installing
With spirv-tools installed (will save significant compile time):
```text
cargo +nightly-2023-01-21 install krnlc --no-default-features --features use-installed-tools
```
Otherwise:
```text
cargo +nightly-2023-01-21 install krnlc
```

## Metadata
**krnlc** can read metadata from Cargo.toml:
```toml
[package.metadata.krnlc]
# enable default features when locating modules
default-features = false
# features to enable when locating modules
features = ["zoom", "zap"]

[package.metadata.krnlc.dependencies]
# keys are inherited from resolved values for the host target
foo = { version = "0.1.0", features = ["foo"] }
bar = { default-features = false }
baz = {}
```

## Compiling your kernels!
Compile with `krnlc` or `krnlc -p my-crate`:
1. Runs the equivalent of [`cargo expand`](https://github.com/dtolnay/cargo-expand) to locate all modules.
2. Generates a device crate under \<target-dir\>/krnlc/crates/\<my-crate\>.
3. Compiles the device crate with [spirv-builder](https://docs.rs/crate/spirv-builder).
4. Processes the output, validates and optimizes with [spirv-tools](https://docs.rs/spirv-tools).
5. Writes out to "krnl-cache.rs".

Note: Can also run with `--check` which will check that the cache is up to date without writing to it.
*/

use crate::{
    device::{Device, DeviceInner, Features},
    scalar::{ScalarElem, ScalarType},
};
use anyhow::{bail, Result};
#[cfg(feature = "device")]
use rspirv::{binary::Assemble, dr::Operand};
use serde::Deserialize;
use std::sync::Arc;
#[cfg(feature = "device")]
use std::{collections::HashMap, hash::Hash};

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Deserialize, Debug)]
pub(crate) struct KernelDesc {
    pub(crate) name: String,
    hash: u64,
    pub(crate) spirv: Vec<u32>,
    features: Features,
    pub(crate) threads: Vec<u32>,
    safe: bool,
    spec_descs: Vec<SpecDesc>,
    pub(crate) slice_descs: Vec<SliceDesc>,
    push_descs: Vec<PushDesc>,
}

#[cfg(feature = "device")]
impl KernelDesc {
    pub(crate) fn push_consts_range(&self) -> u32 {
        let mut size: usize = self.push_descs.iter().map(|x| x.scalar_type.size()).sum();
        while size % 4 != 0 {
            size += 1;
        }
        size += self.slice_descs.len() * 2 * 4;
        size.try_into().unwrap()
    }
    fn specialize(&self, threads: Vec<u32>, spec_consts: &[ScalarElem]) -> Result<Self> {
        use rspirv::spirv::{Decoration, Op};
        let mut module = rspirv::dr::load_words(&self.spirv).unwrap();
        let mut spec_ids = HashMap::<u32, u32>::with_capacity(spec_consts.len());
        for inst in module.annotations.iter() {
            if inst.class.opcode == Op::Decorate {
                if let [Operand::IdRef(id), Operand::Decoration(Decoration::SpecId), Operand::LiteralInt32(spec_id)] =
                    inst.operands.as_slice()
                {
                    spec_ids.insert(*id, *spec_id);
                }
            }
        }
        for inst in module.types_global_values.iter_mut() {
            if inst.class.opcode == Op::SpecConstant {
                if let Some(result_id) = inst.result_id {
                    if let Some(spec_id) = spec_ids.get(&result_id) {
                        if let Some(value) = spec_consts.get(*spec_id as usize) {
                            match inst.operands.as_mut_slice() {
                                [Operand::LiteralInt32(a)] => {
                                    bytemuck::bytes_of_mut(a).copy_from_slice(value.as_bytes());
                                }
                                [Operand::LiteralInt32(a), Operand::LiteralInt32(b)] => {
                                    bytemuck::bytes_of_mut(a)
                                        .copy_from_slice(&value.as_bytes()[..8]);
                                    bytemuck::bytes_of_mut(b)
                                        .copy_from_slice(&value.as_bytes()[9..]);
                                }
                                _ => unreachable!("{:?}", inst.operands),
                            }
                        }
                    }
                }
            }
        }
        {
            //use rspirv::binary::Disassemble;
            //eprintln!("{}", module.disassemble())
        }
        let spirv = module.assemble();
        Ok(Self {
            spirv,
            spec_descs: Vec::new(),
            threads,
            ..self.clone()
        })
    }
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Deserialize, Debug)]
struct SpecDesc {
    #[allow(unused)]
    name: String,
    scalar_type: ScalarType,
    thread_dim: Option<usize>,
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Deserialize, Debug)]
pub(crate) struct SliceDesc {
    name: String,
    pub(crate) scalar_type: ScalarType,
    pub(crate) mutable: bool,
    item: bool,
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Deserialize, Debug)]
struct PushDesc {
    #[allow(unused)]
    name: String,
    scalar_type: ScalarType,
}

#[cfg(feature = "device")]
#[derive(PartialEq, Eq, Hash, Debug)]
pub(crate) struct KernelKey {
    id: usize,
    spec_bytes: Vec<u8>,
}

#[doc(hidden)]
pub mod __private {
    use super::*;
    #[cfg(feature = "device")]
    use crate::device::{DeviceBuffer, RawKernel};
    use crate::{
        buffer::{ScalarSlice, ScalarSliceMut, Slice, SliceMut},
        scalar::Scalar,
    };

    #[doc(hidden)]
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    #[derive(Clone)]
    pub struct KernelBuilder {
        id: usize,
        desc: Arc<KernelDesc>,
        spec_consts: Vec<ScalarElem>,
        threads: [u32; 3],
    }

    impl KernelBuilder {
        pub fn from_bytes(bytes: &'static [u8]) -> Result<Self> {
            let desc: Arc<KernelDesc> = Arc::new(bincode2::deserialize(bytes)?);
            let mut threads = [1, 1, 1];
            threads[..desc.threads.len()].copy_from_slice(&desc.threads);
            Ok(Self {
                id: bytes.as_ptr() as usize,
                desc,
                spec_consts: Vec::new(),
                threads,
            })
        }
        pub fn specialize(mut self, spec_consts: &[ScalarElem]) -> Result<Self> {
            assert_eq!(spec_consts.len(), self.desc.spec_descs.len());
            for (spec_const, spec_desc) in
                spec_consts.iter().copied().zip(self.desc.spec_descs.iter())
            {
                assert_eq!(spec_const.scalar_type(), spec_desc.scalar_type);
                if let Some(dim) = spec_desc.thread_dim {
                    if let ScalarElem::U32(value) = spec_const {
                        if value == 0 {
                            bail!("threads.{} cannot be zero!", ["x", "y", "z"][dim],);
                        }
                        self.threads[dim] = value;
                    } else {
                        unreachable!()
                    }
                }
            }
            self.spec_consts.clear();
            self.spec_consts.extend_from_slice(spec_consts);
            Ok(self)
        }
        pub fn features(&self) -> Features {
            self.desc.features
        }
        pub fn hash(&self) -> u64 {
            self.desc.hash
        }
        pub fn safe(&self) -> bool {
            self.desc.safe
        }
        pub fn build(&self, device: Device) -> Result<Kernel> {
            match device.inner() {
                DeviceInner::Host => {
                    bail!("Kernel `{}` expected device, found host!", self.desc.name);
                }
                #[cfg(feature = "device")]
                DeviceInner::Device(device) => {
                    let desc = &self.desc;
                    let name = &desc.name;
                    let features = desc.features;
                    let device_features = device.info().features();
                    if !device_features.contains(&features) {
                        bail!("Kernel {name} requires {features:?}, {device:?} has {device_features:?}!");
                    }
                    let spec_bytes = if !self.desc.spec_descs.is_empty() {
                        if self.spec_consts.is_empty() {
                            bail!("Kernel `{name}` must be specialized!");
                        }
                        self.spec_consts
                            .iter()
                            .flat_map(|x| x.as_bytes())
                            .copied()
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let key = KernelKey {
                        id: self.id,
                        spec_bytes,
                    };
                    let inner = if !desc.spec_descs.is_empty() {
                        RawKernel::cached(device.clone(), key, || {
                            desc.specialize(
                                self.threads[..self.desc.threads.len()].to_vec(),
                                &self.spec_consts,
                            )
                            .map(Arc::new)
                        })?
                    } else {
                        RawKernel::cached(device.clone(), key, || Ok(desc.clone()))?
                    };
                    Ok(Kernel {
                        inner,
                        groups: None,
                    })
                }
            }
        }
    }

    #[doc(hidden)]
    #[derive(Clone)]
    pub struct Kernel {
        #[cfg(feature = "device")]
        inner: RawKernel,
        #[cfg(feature = "device")]
        groups: Option<[u32; 3]>,
    }

    #[cfg(feature = "device")]
    fn global_threads_to_groups(global_threads: &[u32], threads: &[u32]) -> [u32; 3] {
        debug_assert_eq!(global_threads.len(), threads.len());
        let mut groups = [1; 3];
        for (gt, (g, t)) in global_threads
            .iter()
            .copied()
            .zip(groups.iter_mut().zip(threads.iter().copied()))
        {
            *g = gt / t + u32::from(gt % t != 0);
        }
        groups
    }

    impl Kernel {
        pub fn with_global_threads(
            #[cfg_attr(not(feature = "device"), allow(unused_mut))] mut self,
            global_threads: &[u32],
        ) -> Self {
            #[cfg(feature = "device")]
            {
                let desc = &self.inner.desc();
                let groups = global_threads_to_groups(global_threads, &desc.threads);
                self.groups.replace(groups);
                self
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = global_threads;
                unreachable!()
            }
        }
        pub fn with_groups(
            #[cfg_attr(not(feature = "device"), allow(unused_mut))] mut self,
            groups: &[u32],
        ) -> Self {
            #[cfg(feature = "device")]
            {
                debug_assert_eq!(groups.len(), self.inner.desc().threads.len());
                let mut new_groups = [1; 3];
                new_groups[..groups.len()].copy_from_slice(groups);
                self.groups.replace(new_groups);
                self
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = groups;
                unreachable!()
            }
        }
        pub unsafe fn dispatch(
            &self,
            slices: &[KernelSliceArg],
            push_consts: &[ScalarElem],
        ) -> Result<()> {
            #[cfg(feature = "device")]
            {
                let desc = &self.inner.desc();
                let kernel_name = &desc.name;
                let mut buffers = Vec::with_capacity(desc.slice_descs.len());
                let mut items: Option<usize> = None;
                let device = self.inner.device();
                for (slice, slice_desc) in slices.iter().zip(desc.slice_descs.iter()) {
                    debug_assert_eq!(slice.scalar_type(), slice_desc.scalar_type);
                    debug_assert!(!slice_desc.mutable || slice.mutable());
                    let slice_name = &slice_desc.name;
                    let buffer = if let Some(buffer) = slice.device_buffer() {
                        buffer
                    } else {
                        bail!("Kernel `{kernel_name}`.`{slice_name}` expected device, found host!");
                    };
                    let buffer_device = buffer.device();
                    if device != buffer_device {
                        bail!(
                            "Kernel `{kernel_name}`.`{slice_name}`, expected `{device:?}`, found {buffer_device:?}!"
                        );
                    }
                    buffers.push(buffer.clone());
                    if slice_desc.item {
                        items.replace(if let Some(items) = items {
                            items.min(slice.len())
                        } else {
                            slice.len()
                        });
                    }
                }
                let groups = if let Some(groups) = self.groups {
                    groups
                } else if let Some(items) = items {
                    if desc.threads.iter().skip(1).any(|t| *t > 1) {
                        bail!("Kernel `{kernel_name}` cannot infer global_threads if threads.y > 1 or threads.z > 1, threads = {threads:?}!", threads = desc.threads);
                    }
                    global_threads_to_groups(&[items as u32], &[desc.threads[0]])
                } else {
                    bail!("Kernel `{kernel_name}` global_threads or groups not provided!");
                };
                let mut push_bytes = Vec::with_capacity(desc.push_consts_range() as usize);
                for (push, push_desc) in push_consts.iter().zip(desc.push_descs.iter()) {
                    debug_assert_eq!(push.scalar_type(), push_desc.scalar_type);
                    push_bytes.extend_from_slice(push.as_bytes());
                }
                unsafe { self.inner.dispatch(groups, &buffers, push_bytes) }
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = (slices, push_consts);
                unreachable!()
            }
        }
        /*pub fn threads(&self) -> &[u32] {
            #[cfg(feature = "device")]
            {
                return self.inner.desc().threads.as_ref();
            }
            #[cfg(not(feature = "device"))]
            {
                unreachable!()
            }
        }*/
        pub fn features(&self) -> Features {
            #[cfg(feature = "device")]
            {
                return self.inner.desc().features;
            }
            #[cfg(not(feature = "device"))]
            {
                unreachable!()
            }
        }
    }

    #[doc(hidden)]
    pub enum KernelSliceArg<'a> {
        Slice(ScalarSlice<'a>),
        SliceMut(ScalarSliceMut<'a>),
    }

    #[cfg(feature = "device")]
    impl KernelSliceArg<'_> {
        fn scalar_type(&self) -> ScalarType {
            match self {
                Self::Slice(x) => x.scalar_type(),
                Self::SliceMut(x) => x.scalar_type(),
            }
        }
        fn mutable(&self) -> bool {
            match self {
                Self::Slice(_) => false,
                Self::SliceMut(_) => true,
            }
        }
        /*fn device(&self) -> Device {
            match self {
                Self::Slice(x) => x.device(),
                Self::SliceMut(x) => x.device(),
            }
        }*/
        fn device_buffer(&self) -> Option<&DeviceBuffer> {
            match self {
                Self::Slice(x) => x.device_buffer(),
                Self::SliceMut(x) => x.device_buffer_mut(),
            }
        }
        fn len(&self) -> usize {
            match self {
                Self::Slice(x) => x.len(),
                Self::SliceMut(x) => x.len(),
            }
        }
    }

    impl<'a, T: Scalar> From<Slice<'a, T>> for KernelSliceArg<'a> {
        fn from(slice: Slice<'a, T>) -> Self {
            Self::Slice(slice.into())
        }
    }

    impl<'a, T: Scalar> From<SliceMut<'a, T>> for KernelSliceArg<'a> {
        fn from(slice: SliceMut<'a, T>) -> Self {
            Self::SliceMut(slice.into())
        }
    }
}
