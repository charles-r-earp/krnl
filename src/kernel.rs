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
        if index < y.len() {
            unsafe {
                *y.unsafe_index_mut(index) = 1;
            }
        }
    }

    #[kernel]
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
            // optionally set threads per group
            // chooses a reasonable default if not provided
            .with_threads(128)
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

    #[kernel]
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
    /// **Errors**
    /// - The kernel wasn't compiled (with `#[krnl(no_build)]` applied to `#[module]`).
    /// - The kernel could not be deserialized. For stable releases, this is a bug, as `#[module]` should produce a compile error.
    pub fn builder() -> Result<KernelBuilder>;

    impl KernelBuilder {
        /// Threads per group.
        ///
        /// Defaults to [`DeviceInfo::default_threads()`](DeviceInfo::default_threads).
        pub fn with_threads(self, threads: u32) -> Self;
        /// Builds the kernel for `device`.
        ///
        /// The kernel is cached, so subsequent calls to `.build()` with identical
        /// builders (ie threads and spec constants) may avoid recompiling.
        ///
        /// **Errors**
        /// - `device` doesn't have required features.
        /// - The kernel requires [specialization](kernel#specialization), but `.specialize(..)` was not called.
        /// - The kernel is not supported on `device`.
        /// - [`DeviceLost`].
        pub fn build(&self, device: Device) -> Result<Kernel>;
    }

    /// Kernel.
    pub struct Kernel { /* .. */ }

    impl Kernel {
        /// Threads per group.
        pub fn threads(&self) -> u32;
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
        /// **Errors**
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

Functions and items can be shared between modules:
```no_run
use krnl::macros::module;

mod util {
    use krnl::macros::module;

    #[module]
    # #[krnl(no_build)]
    pub mod functional {
        #[cfg(not(target_arch = "spirv"))]
        use krnl::krnl_core;
        use krnl_core::scalar::Scalar;

        pub fn add<T: Scalar>(a: T, b: T) -> T {
            a + b
        }
    }
}

#[module]
# #[krnl(no_build)]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use crate::util::functional::add;

    #[kernel]
    fn add_i32(#[item] a: i32, #[item] b: i32, #[item] c: &mut i32) {
        *c = add(a, b);
    }
}
```

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
        #[kernel]
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
Kernels are dispatched in groups of threads (CUDA thread blocks). The threads provided to `.with_threads(..)`
sets the number of threads per group, which defaults to [`DeviceInfo::default_threads()`](crate::device::DeviceInfo::default_threads).

For simple kernels, threads can be arbitrary, typically 128, 256, or 512. This can be tuned for optimal performance. It must be less than
[`DeviceInfo::max_threads()`](crate::device::DeviceInfo::max_threads).

Item kernels can infer the global_threads by the sizes of the item arguments. This is functionally equivalent
to `.iter().zip(..)`, where the number of items is the minimum of the lengths of the buffers.

Note that `global_threads = groups * threads`. When provided to `.with_global_threads()` prior to dispatch, global_threads
are rounded up to the next multiple of threads. Because of this, it is typical to check that the global_id is
in bounds as implemented above in `fill_ones_impl`.

Kernels can declare group shared memory:
```no_run
# use krnl::macros::module;
# #[module]
# #[krnl(no_build)]
# mod kernels {
# use krnl::krnl_core;
# use krnl_core::macros::kernel;
#[kernel]
fn group_sum(
    #[global] x: Slice<f32>,
    #[group] x_group: UnsafeSlice<f32, 64>,
    #[global] y: UnsafeSlice<f32>,
) {
    use krnl_core::{buffer::UnsafeIndex, spirv_std::arch::workgroup_memory_barrier_with_group_sync as group_barrier};

    let global_id = kernel.global_id as usize;
    let group_id = kernel.group_id as usize;
    let thread_id = kernel.thread_id as usize;
    unsafe {
        x_group.unsafe_index_mut(thread_id) = x[global_id];
        // Barriers are used to synchronize access to group memory.
        // This call must be reached by all active threads in the group!
        group_barrier();
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
Thread groups are composed of subgroups of threads (CUDA warps). The number of threads per subgroup is at
most 128. Typical values are:
- 32: NVIDIA
- 64: AMD

Note that it must be at least 1 and not greater than 128. It can be accessed in a kernel via [`Kernel::subgroup_threads()`](krnl_core::kernel::Kernel::subgroup_threads),
or on the host via [`DeviceInfo::subgroup_threads()`](crate::device::DeviceInfo::subgroup_threads).

# Features
Kernels implicitly declare [`Features`](device::Features) based on types and or operations used.
If the [device](device::Device) does not support these features, `.build()` will return an
error.

See [`DeviceInfo::features()`](device::DeviceInfo::features).

# Specialization
SpecConstants are constants that are set when the kernel is compiled via `.specialize(..)`.

```no_run
# use krnl::{macros::module, anyhow::Result, device::Device};
# #[module]
# #[krnl(no_build)]
# mod kernels {
# use krnl::krnl_core;
# use krnl_core::macros::kernel;
#[kernel]
pub fn group_sum<const N: u32 /*, ...additional spec constants */>(
    #[global] x: Slice<f32>,
    #[group] x_group: UnsafeSlice<f32, { 2 * N as usize }>,
    #[global] y: UnsafeSlice<f32>,
) {
    /* N is available here, but isn't const. */
}
# }
# fn foo(device: Device) -> Result<()> {
# use kernels::group_sum;
let n = 128;
let kernel = group_sum::builder()?
    .with_threads(n)
    .specialize(n /*, ...additional spec constants */)
    .build(device)?;
# todo!()
# }
```

# Panics
Panics will abort the thread, but this will not be caught from the host.

# Compiling
Kernels are compiled with **krnlc**.

## Toolchains
To locate modules, **krnlc** will use the nightly toolchain. Install it with:
```text
rustup toolchain install nightly
```
To compile kernels with [spirv-builder](https://docs.rs/crate/spirv-builder), a specific nightly is required:
```text
rustup component add --toolchain nightly-2023-04-15 rust-src rustc-dev llvm-tools-preview
```

## Installing
With spirv-tools installed (will save significant compile time):
```text
cargo +nightly-2023-04-15 install krnlc --locked --no-default-features --features use-installed-tools
```
Otherwise:
```text
cargo +nightly-2023-04-15 install krnlc --locked
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
# source is inherited from host target
foo = { default-features = false, features = ["foo"] }
# keys are inherited if not provided
bar = {}
# private dependency
baz = { path = "baz" }
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
use dry::macro_wrap;
#[cfg(feature = "device")]
use rspirv::{binary::Assemble, dr::Operand};
use std::{borrow::Cow, sync::Arc};
#[cfg(feature = "device")]
use std::{collections::HashMap, hash::Hash};

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Clone, Debug)]
pub(crate) struct KernelDesc {
    pub(crate) name: Cow<'static, str>,
    pub(crate) spirv: Vec<u32>,
    features: Features,
    pub(crate) threads: u32,
    spec_descs: &'static [SpecDesc],
    pub(crate) slice_descs: &'static [SliceDesc],
    push_descs: &'static [PushDesc],
}

#[cfg(feature = "device")]
impl KernelDesc {
    pub(crate) fn push_consts_range(&self) -> u32 {
        let mut size = 0;
        for push_desc in self.push_descs.iter() {
            while size % push_desc.scalar_type.size() != 0 {
                size += 1;
            }
            size += push_desc.scalar_type.size()
        }
        while size % 4 != 0 {
            size += 1;
        }
        size += self.slice_descs.len() * 2 * 4;
        size.try_into().unwrap()
    }
    fn specialize(
        &self,
        threads: u32,
        spec_consts: &[ScalarElem],
        debug_printf: bool,
    ) -> Result<Self> {
        use rspirv::spirv::{Decoration, Op};
        let mut module = rspirv::dr::load_words(&self.spirv).unwrap();
        let mut spec_ids = HashMap::<u32, u32>::with_capacity(spec_consts.len());
        let mut spec_string = format!("threads={threads}");
        use std::fmt::Write;
        for (desc, spec) in self.spec_descs.iter().zip(spec_consts) {
            if !spec_string.is_empty() {
                spec_string.push_str(", ");
            }
            let n = desc.name;
            macro_wrap!(match spec {
                macro_for!($T in [U8, I8, U16, I16, F16, BF16, U32, I32, F32, U64, I64, F64] {
                    ScalarElem::$T(x) => write!(&mut spec_string, "{n}={x}").unwrap(),
                })
                _ => unreachable!("{spec:?}"),
            });
        }
        let name = if !spec_string.is_empty() {
            format!("{}<{spec_string}>", self.name).into()
        } else {
            self.name.clone()
        };
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
                    if let Some(spec_id) = spec_ids.get(&result_id).copied().map(|x| x as usize) {
                        let value = if let Some(value) = spec_consts.get(spec_id).copied() {
                            value
                        } else if spec_id == spec_consts.len() {
                            ScalarElem::U32(threads)
                        } else {
                            unreachable!("{inst:?}")
                        };
                        match inst.operands.as_mut_slice() {
                            [Operand::LiteralInt32(a)] => {
                                bytemuck::bytes_of_mut(a).copy_from_slice(value.as_bytes());
                            }
                            [Operand::LiteralInt32(a), Operand::LiteralInt32(b)] => {
                                bytemuck::bytes_of_mut(a).copy_from_slice(&value.as_bytes()[..8]);
                                bytemuck::bytes_of_mut(b).copy_from_slice(&value.as_bytes()[9..]);
                            }
                            _ => unreachable!("{:?}", inst.operands),
                        }
                    }
                }
            }
        }
        if !debug_printf {
            strip_debug_printf(&mut module);
        }
        let spirv = module.assemble();
        Ok(Self {
            name,
            spirv,
            spec_descs: &[],
            threads,
            ..self.clone()
        })
    }
}

#[cfg(feature = "device")]
fn strip_debug_printf(module: &mut rspirv::dr::Module) {
    use rspirv::spirv::Op;
    use std::collections::HashSet;

    module.extensions.retain(|inst| {
        inst.operands.first().unwrap().unwrap_literal_string() != "SPV_KHR_non_semantic_info"
    });
    let mut ext_insts = HashSet::new();
    module.ext_inst_imports.retain(|inst| {
        if inst
            .operands
            .first()
            .unwrap()
            .unwrap_literal_string()
            .starts_with("NonSemantic.DebugPrintf")
        {
            ext_insts.insert(inst.result_id.unwrap());
            false
        } else {
            true
        }
    });
    if ext_insts.is_empty() {
        return;
    }
    module.debug_string_source.clear();
    for func in module.functions.iter_mut() {
        for block in func.blocks.iter_mut() {
            block.instructions.retain(|inst| {
                if inst.class.opcode == Op::ExtInst {
                    let id = inst.operands.first().unwrap().unwrap_id_ref();
                    if ext_insts.contains(&id) {
                        return false;
                    }
                }
                !matches!(inst.class.opcode, Op::Line | Op::NoLine)
            })
        }
    }
}

#[cfg(feature = "device")]
#[derive(PartialEq, Eq, Hash, Debug)]
pub(crate) struct KernelKey {
    id: usize,
    spec_bytes: Vec<u8>,
}

#[doc(hidden)]
pub mod __private {
    #[cfg(feature = "device")]
    use num_traits::ToPrimitive;

    use super::*;
    #[cfg(feature = "device")]
    use crate::device::{DeviceBuffer, RawKernel};
    use crate::{
        buffer::{ScalarSlice, ScalarSliceMut, Slice, SliceMut},
        scalar::Scalar,
    };

    #[derive(Clone, Copy)]
    pub struct KernelDesc {
        name: &'static str,
        spirv: &'static [u8],
        features: Features,
        safe: bool,
        spec_descs: &'static [SpecDesc],
        slice_descs: &'static [SliceDesc],
        push_descs: &'static [PushDesc],
    }

    #[derive(Clone, Copy)]
    pub struct KernelDescArgs {
        pub name: &'static str,
        pub spirv: &'static [u8],
        pub features: Features,
        pub safe: bool,
        pub spec_descs: &'static [SpecDesc],
        pub slice_descs: &'static [SliceDesc],
        pub push_descs: &'static [PushDesc],
    }

    const fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    pub const fn find_kernel(name: &str, kernels: &[KernelDesc]) -> Option<KernelDesc> {
        let mut i = 0;
        while i < kernels.len() {
            if bytes_eq(name.as_bytes(), kernels[i].name.as_bytes()) {
                return Some(kernels[i]);
            }
            i += 1;
        }
        None
    }

    pub const fn validate_kernel(
        kernel: Option<Option<KernelDesc>>,
        safety: Safety,
        spec_descs: &[SpecDesc],
        slice_descs: &[SliceDesc],
        push_descs: &[PushDesc],
    ) -> Option<KernelDesc> {
        if let Some(kernel) = kernel {
            let success = if let Some(kernel) = kernel.as_ref() {
                kernel.check_declaration(safety, spec_descs, slice_descs, push_descs)
            } else {
                false
            };
            if !success {
                panic!("recompile with krnlc");
            }
            kernel
        } else {
            None
        }
    }

    impl KernelDesc {
        pub const fn from_args(args: KernelDescArgs) -> Self {
            let KernelDescArgs {
                name,
                spirv,
                features,
                safe,
                spec_descs,
                slice_descs,
                push_descs,
            } = args;
            Self {
                name,
                spirv,
                features,
                safe,
                spec_descs,
                slice_descs,
                push_descs,
            }
        }
        const fn check_declaration(
            &self,
            safety: Safety,
            spec_descs: &[SpecDesc],
            slice_descs: &[SliceDesc],
            push_descs: &[PushDesc],
        ) -> bool {
            if self.safe != safety.is_safe() {
                return false;
            }
            {
                if self.spec_descs.len() != spec_descs.len() {
                    return false;
                }
                let mut index = spec_descs.len();
                while index < spec_descs.len() {
                    if !self.spec_descs[index].const_eq(&spec_descs[index]) {
                        return false;
                    }
                    index += 1;
                }
            }
            {
                if self.slice_descs.len() != slice_descs.len() {
                    return false;
                }
                let mut index = slice_descs.len();
                while index < slice_descs.len() {
                    if !self.slice_descs[index].const_eq(&slice_descs[index]) {
                        return false;
                    }
                    index += 1;
                }
            }
            {
                if self.push_descs.len() != push_descs.len() {
                    return false;
                }
                let mut index = push_descs.len();
                while index < push_descs.len() {
                    if !self.push_descs[index].const_eq(&push_descs[index]) {
                        return false;
                    }
                    index += 1;
                }
            }
            true
        }
    }

    #[derive(Clone, Copy)]
    pub enum Safety {
        Safe,
        Unsafe,
    }

    impl Safety {
        const fn is_safe(&self) -> bool {
            matches!(self, Self::Safe)
        }
    }

    const fn scalar_type_const_eq(a: ScalarType, b: ScalarType) -> bool {
        a as u32 == b as u32
    }

    #[derive(Clone, Copy, Debug)]
    pub struct SpecDesc {
        pub name: &'static str,
        pub scalar_type: ScalarType,
    }

    impl SpecDesc {
        const fn const_eq(&self, other: &Self) -> bool {
            bytes_eq(self.name.as_bytes(), other.name.as_bytes())
                && scalar_type_const_eq(self.scalar_type, other.scalar_type)
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct SliceDesc {
        pub name: &'static str,
        pub scalar_type: ScalarType,
        pub mutable: bool,
        pub item: bool,
    }

    impl SliceDesc {
        const fn const_eq(&self, other: &Self) -> bool {
            bytes_eq(self.name.as_bytes(), other.name.as_bytes())
                && scalar_type_const_eq(self.scalar_type, other.scalar_type)
                && self.mutable == other.mutable
                && self.item == other.item
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct PushDesc {
        pub name: &'static str,
        pub scalar_type: ScalarType,
    }

    impl PushDesc {
        const fn const_eq(&self, other: &Self) -> bool {
            bytes_eq(self.name.as_bytes(), other.name.as_bytes())
                && scalar_type_const_eq(self.scalar_type, other.scalar_type)
        }
    }

    fn decode_spirv(name: &str, input: &[u8]) -> Result<Vec<u32>, String> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut output = Vec::new();
        GzDecoder::new(bytemuck::cast_slice(input))
            .read_to_end(&mut output)
            .map_err(|e| format!("Kernel `{name}` failed to decode! {e}"))?;
        let output = output
            .chunks_exact(4)
            .map(|x| u32::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        Ok(output)
    }

    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    #[derive(Clone)]
    pub struct KernelBuilder {
        id: usize,
        desc: Arc<super::KernelDesc>,
        spec_consts: Vec<ScalarElem>,
        threads: Option<u32>,
    }

    impl KernelBuilder {
        pub fn from_desc(desc: KernelDesc) -> Result<Self, String> {
            let KernelDesc {
                name,
                spirv,
                features,
                safe: _,
                spec_descs,
                slice_descs,
                push_descs,
            } = desc;
            let spirv = decode_spirv(name, spirv)?;
            let desc = super::KernelDesc {
                name: name.into(),
                spirv,
                features,
                threads: 0,
                spec_descs,
                slice_descs,
                push_descs,
            };
            Ok(Self {
                id: name.as_ptr() as usize,
                desc: desc.into(),
                spec_consts: Vec::new(),
                threads: None,
            })
        }
        pub fn with_threads(self, threads: u32) -> Self {
            Self {
                threads: Some(threads),
                ..self
            }
        }
        pub fn specialize(self, spec_consts: &[ScalarElem]) -> Self {
            assert_eq!(spec_consts.len(), self.desc.spec_descs.len());
            for (spec_const, spec_desc) in
                spec_consts.iter().copied().zip(self.desc.spec_descs.iter())
            {
                assert_eq!(spec_const.scalar_type(), spec_desc.scalar_type);
            }
            Self {
                spec_consts: spec_consts.to_vec(),
                ..self
            }
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
                    let info = device.info();
                    let device_features = info.features();
                    if !device_features.contains(&features) {
                        bail!("Kernel {name} requires {features:?}, {device:?} has {device_features:?}!");
                    }
                    let threads = self.threads.unwrap_or(info.default_threads());
                    let max_threads = info.max_threads();
                    if threads > max_threads {
                        bail!("Kernel {name} threads {threads} is greater than max_threads {max_threads}!");
                    }
                    let spec_bytes = {
                        if !self.desc.spec_descs.is_empty() && self.spec_consts.is_empty() {
                            bail!("Kernel `{name}` must be specialized!");
                        }
                        debug_assert_eq!(self.spec_consts.len(), desc.spec_descs.len());
                        #[cfg(debug_assertions)]
                        {
                            for (spec_const, spec_desc) in
                                self.spec_consts.iter().zip(desc.spec_descs.iter())
                            {
                                assert_eq!(spec_const.scalar_type(), spec_desc.scalar_type);
                            }
                        }
                        self.spec_consts
                            .iter()
                            .flat_map(|x| x.as_bytes())
                            .copied()
                            .chain(threads.to_ne_bytes())
                            .collect()
                    };
                    let key = KernelKey {
                        id: self.id,
                        spec_bytes,
                    };
                    let debug_printf = info.debug_printf();
                    let inner = RawKernel::cached(device.clone(), key, || {
                        desc.specialize(threads, &self.spec_consts, debug_printf)
                            .map(Arc::new)
                    })?;
                    Ok(Kernel {
                        inner,
                        threads,
                        groups: None,
                    })
                }
            }
        }
    }

    #[derive(Clone)]
    pub struct Kernel {
        #[cfg(feature = "device")]
        inner: RawKernel,
        threads: u32,
        #[cfg(feature = "device")]
        groups: Option<u32>,
    }

    impl Kernel {
        pub fn threads(&self) -> u32 {
            self.threads
        }
        pub fn with_global_threads(self, global_threads: u32) -> Self {
            #[cfg(feature = "device")]
            {
                let desc = &self.inner.desc();
                let threads = desc.threads;
                let groups = global_threads / threads + u32::from(global_threads % threads != 0);
                self.with_groups(groups)
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = global_threads;
                unreachable!()
            }
        }
        pub fn with_groups(self, groups: u32) -> Self {
            #[cfg(feature = "device")]
            {
                Self {
                    groups: Some(groups),
                    ..self
                }
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
                let mut items: Option<u32> = None;
                let device = self.inner.device();
                let mut push_bytes = Vec::with_capacity(desc.push_consts_range() as usize);
                debug_assert_eq!(push_consts.len(), desc.push_descs.len());
                for (push, push_desc) in push_consts.iter().zip(desc.push_descs.iter()) {
                    debug_assert_eq!(push.scalar_type(), push_desc.scalar_type);
                    debug_assert_eq!(push_bytes.len() % push.scalar_type().size(), 0);
                    push_bytes.extend_from_slice(push.as_bytes());
                }
                while push_bytes.len() % 4 != 0 {
                    push_bytes.push(0);
                }
                for (slice, slice_desc) in slices.iter().zip(desc.slice_descs.iter()) {
                    debug_assert_eq!(slice.scalar_type(), slice_desc.scalar_type);
                    debug_assert!(!slice_desc.mutable || slice.mutable());
                    let slice_name = &slice_desc.name;
                    if slice.len() == 0 {
                        bail!("Kernel `{kernel_name}`.`{slice_name}` is empty!");
                    }
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
                            items.min(slice.len() as u32)
                        } else {
                            slice.len() as u32
                        });
                    }
                    let width = slice_desc.scalar_type.size();
                    let offset = buffer.offset() / width;
                    let len = buffer.len() / width;
                    push_bytes.extend_from_slice(&offset.to_u32().unwrap().to_ne_bytes());
                    push_bytes.extend_from_slice(&len.to_u32().unwrap().to_ne_bytes());
                }
                let info = self.inner.device().info().clone();
                let max_groups = info.max_groups();
                let groups = if let Some(groups) = self.groups {
                    if groups > max_groups {
                        bail!("Kernel `{kernel_name}` groups {groups} is greater than max_groups {max_groups}!");
                    }
                    groups
                } else if let Some(items) = items {
                    let threads = self.threads;
                    let groups = items / threads + u32::from(items % threads != 0);
                    groups.min(max_groups)
                } else {
                    bail!("Kernel `{kernel_name}` global_threads or groups not provided!");
                };
                unsafe {
                    self.inner.dispatch(groups, &buffers, push_bytes)?;
                }
                Ok(())
            }
            #[cfg(not(feature = "device"))]
            {
                let _ = (slices, push_consts);
                unreachable!()
            }
        }
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

pub(crate) use __private::{PushDesc, SliceDesc, SpecDesc};
