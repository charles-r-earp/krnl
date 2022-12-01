/*use crate::{
    krnl_core::vek::vec::{Vec2, Vec3},
    device::{Device, DeviceInner, Features},
    scalar::{ScalarType, ScalarElem},
    buffer::{Slice, SliceMut, ScalarSlice, ScalarSliceMut},
};
#[cfg(feature = "device")]
use crate::device::{VulkanDevice, KernelCache, DeviceBufferInner, Compute};
use std::{marker::PhantomData, borrow::Cow, sync::Arc, fmt::{self, Debug}, convert::TryInto, collections::{HashMap, HashSet}};
use anyhow::{Result, format_err, bail};
*/
#[doc(inline)]
pub use krnl_macros::module;

#[doc(hidden)]
pub mod __private {
    #[cfg(feature = "device")]
    use crate::device::{Compute, DeviceBufferInner, KernelCache};
    use crate::{
        anyhow::{bail, format_err, Result},
        buffer::{RawSlice, ScalarSlice, ScalarSliceMut, Slice, SliceMut},
        device::{Device, DeviceInner, Features},
        krnl_core::glam::{UVec2, UVec3},
        scalar::{Scalar, ScalarElem, ScalarType},
    };
    use std::{
        borrow::Cow,
        collections::HashMap,
        fmt::{self, Debug},
        marker::PhantomData,
        sync::Arc,
    };

    pub mod builder {
        use super::*;

        pub struct KernelBuilder {
            name: Cow<'static, str>,
            spirv: &'static [u32],
            threads: UVec3,
            spec_consts: Vec<u32>,
            buffer_descs: &'static [BufferDesc],
            push_consts_size: u32,
            features: Features,
        }

        impl KernelBuilder {
            pub(super) fn new(name: Cow<'static, str>, spirv: &'static [u32]) -> Self {
                KernelBuilder {
                    name,
                    spirv,
                    threads: UVec3::new(1, 1, 1),
                    spec_consts: Vec::new(),
                    buffer_descs: &[],
                    push_consts_size: 0,
                    features: Features::default(),
                }
            }
            pub fn threads(mut self, threads: UVec3) -> Self {
                self.threads = threads;
                self
            }
            pub fn specs(mut self, specs: &[ScalarElem]) -> Self {
                self.spec_consts = specs
                    .iter()
                    .flat_map(|x| {
                        use ScalarElem::*;
                        match x.to_scalar_bits() {
                            U8(x) => [u32::from_ne_bytes([x, 0, 0, 0]), 0].into_iter().take(1),
                            U16(x) => {
                                let [x1, x2] = x.to_ne_bytes();
                                [u32::from_ne_bytes([x1, x2, 0, 0]), 0].into_iter().take(1)
                            }
                            U32(x) => [x, 0].into_iter().take(1),
                            U64(x) => {
                                let x = x.to_ne_bytes();
                                [
                                    u32::from_ne_bytes([x[0], x[1], x[2], x[3]]),
                                    u32::from_ne_bytes([x[4], x[5], x[6], x[7]]),
                                ]
                                .into_iter()
                                .take(2)
                            }
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                self
            }
            pub fn buffer_descs(mut self, buffer_descs: &'static [BufferDesc]) -> Self {
                self.buffer_descs = buffer_descs;
                self
            }
            pub fn push_consts_size(mut self, push_consts_size: u32) -> Self {
                self.push_consts_size = push_consts_size;
                self
            }
            pub fn features(mut self, features: Features) -> Self {
                self.features = features;
                self
            }
            pub fn build(self, device: Device) -> Result<Kernel> {
                let kernel_name = &self.name;
                match &device.inner {
                    #[cfg(feature = "device")]
                    DeviceInner::Device(vulkan_device) => {
                        let features = &self.features;
                        let device_features = vulkan_device.features();
                        if !device_features.contains(&self.features) {
                            bail!("Kernel `{kernel_name}` requires {features:?}, device has {device_features:?}!");
                        }
                        let desc = KernelDesc {
                            name: self.name,
                            buffer_descs: self.buffer_descs,
                            threads: self.threads,
                            push_consts_size: self.push_consts_size,
                        };
                        let cache =
                            vulkan_device.kernel_cache(self.spirv, self.spec_consts, desc)?;
                        Ok(Kernel { device, cache })
                    }
                    DeviceInner::Host => Err(format_err!(
                        "Kernel `{kernel_name}` cannot be built for the host!"
                    )),
                }
            }
        }

        pub enum DispatchDim {
            U32(u32),
            UVec2(UVec2),
            UVec3(UVec3),
        }

        impl DispatchDim {
            fn to_array(&self) -> [u32; 3] {
                match self {
                    Self::U32(dim) => [*dim, 1, 1],
                    Self::UVec2(dim) => [dim.x, dim.y, 1],
                    Self::UVec3(dim) => dim.to_array(),
                }
            }
        }

        impl From<u32> for DispatchDim {
            fn from(dim: u32) -> Self {
                Self::U32(dim)
            }
        }

        impl From<UVec2> for DispatchDim {
            fn from(dim: UVec2) -> Self {
                Self::UVec2(dim)
            }
        }

        impl From<UVec3> for DispatchDim {
            fn from(dim: UVec3) -> Self {
                Self::UVec3(dim)
            }
        }

        impl Debug for DispatchDim {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self {
                    Self::U32(dim) => dim.fmt(f),
                    Self::UVec2(dim) => dim.fmt(f),
                    Self::UVec3(dim) => dim.fmt(f),
                }
            }
        }

        pub enum DispatchSlice<'a> {
            Slice(ScalarSlice<'a>),
            SliceMut(ScalarSliceMut<'a>),
        }

        impl DispatchSlice<'_> {
            fn as_raw_slice(&self) -> &RawSlice {
                match self {
                    Self::Slice(slice) => slice.as_raw_slice(),
                    Self::SliceMut(slice) => slice.as_raw_slice(),
                }
            }
        }

        impl<'a, T: Scalar> From<Slice<'a, T>> for DispatchSlice<'a> {
            fn from(slice: Slice<'a, T>) -> Self {
                Self::Slice(ScalarSlice::from(slice))
            }
        }

        impl<'a, T: Scalar> From<SliceMut<'a, T>> for DispatchSlice<'a> {
            fn from(slice: SliceMut<'a, T>) -> Self {
                Self::SliceMut(ScalarSliceMut::from(slice))
            }
        }

        pub struct DispatchBuilder<'a> {
            device: Device,
            #[cfg(feature = "device")]
            cache: Arc<KernelCache>,
            groups: Option<[u32; 3]>,
            #[cfg(feature = "device")]
            buffers: Vec<Arc<DeviceBufferInner>>,
            push_consts: Vec<u32>,
            _m: PhantomData<&'a ()>,
        }

        impl<'a> DispatchBuilder<'a> {
            #[cfg(feature = "device")]
            pub(super) fn new(
                device: Device,
                cache: Arc<KernelCache>,
                slices: &mut [DispatchSlice<'a>],
                push_consts: &[u8],
            ) -> Result<Self> {
                let desc = cache.desc();
                let kernel_name = &desc.name;
                if device.is_host() {
                    bail!("Kernel `{kernel_name}` expected Device(_), found Host!");
                }
                let push_consts_u8 = push_consts;
                let mut push_consts: Vec<u32> = Vec::with_capacity(desc.push_consts_size as usize);
                push_consts.extend(push_consts_u8.chunks(4).map(|x| {
                    u32::from_ne_bytes([
                        x.get(0).copied().unwrap_or_default(),
                        x.get(1).copied().unwrap_or_default(),
                        x.get(2).copied().unwrap_or_default(),
                        x.get(3).copied().unwrap_or_default(),
                    ])
                }));
                let mut items: Option<usize> = None;
                let mut buffers = Vec::with_capacity(slices.len());
                for (buffer_desc, slice) in desc.buffer_descs.iter().zip(slices) {
                    let name = buffer_desc.name;
                    let slice = slice.as_raw_slice();
                    if buffer_desc.item {
                        if let Some(items) = items {
                            if items != slice.len() {
                                bail!("Kernel `{kernel_name}` `{name}` has {} items, expected {items}!", slice.len());
                            }
                        } else {
                            items.replace(slice.len());
                        }
                    }
                    let slice_device = slice.device_ref();
                    if slice_device != &device {
                        bail!("Kernel `{kernel_name}` `{name}` expected {device:?} found {slice_device:?}!");
                    }
                    let device_slice = slice.as_device_slice().unwrap();
                    if let Some(device_buffer) = device_slice.device_buffer() {
                        buffers.push(device_buffer.inner().clone());
                    } else {
                        bail!("Kernel `{kernel_name}` `{name}` is empty!");
                    }
                    push_consts.extend([slice.offset() as u32, slice.len() as u32]);
                }
                let mut builder = Self {
                    device,
                    cache,
                    groups: None,
                    buffers,
                    push_consts,
                    _m: PhantomData::default(),
                };
                if let Some(items) = items {
                    builder = builder.global_threads((items as u32).into())?;
                }
                Ok(builder)
            }
            pub fn global_threads(mut self, global_threads: DispatchDim) -> Result<Self> {
                #[cfg(feature = "device")]
                {
                    let desc = self.cache.desc();
                    let threads = desc.threads;
                    let kernel_name = &desc.name;
                    if global_threads.to_array().iter().any(|x| *x == 0) {
                        bail!("Kernel `{kernel_name}` global_threads cannot be 0, found {global_threads:?}!");
                    }
                    let mut groups = [0; 3];
                    for (g, (gt, t)) in groups
                        .as_mut()
                        .iter_mut()
                        .zip(global_threads.to_array().iter().zip(threads.as_ref()))
                    {
                        *g = gt / t;
                        if gt % t != 0 {
                            *g += 1;
                        }
                    }
                    self.groups.replace(groups);
                    Ok(self)
                }
                #[cfg(not(feature = "device"))]
                {
                    unreachable!()
                }
            }
            pub fn groups(mut self, groups: DispatchDim) -> Result<Self> {
                #[cfg(feature = "device")]
                {
                    let kernel_name = &self.cache.desc().name;
                    if groups.to_array().iter().any(|x| *x == 0) {
                        bail!("Kernel `{kernel_name}` groups cannot be 0, found {groups:?}!");
                    }
                    self.groups.replace(groups.to_array());
                    Ok(self)
                }
                #[cfg(not(feature = "device"))]
                {
                    unreachable!()
                }
            }
            pub unsafe fn dispatch(self) -> Result<()> {
                #[cfg(feature = "device")]
                {
                    let groups = if let Some(groups) = self.groups {
                        groups
                    } else {
                        let kernel_name = &self.cache.desc().name;
                        bail!("Kernel `{kernel_name}` global_threads or groups are required!");
                    };
                    let compute = Compute {
                        cache: self.cache,
                        groups,
                        buffers: self.buffers,
                        push_consts: self.push_consts,
                    };
                    self.device.as_device().unwrap().compute(compute)
                }
                #[cfg(not(feature = "device"))]
                {
                    unreachable!()
                }
            }
        }
    }
    use builder::{DispatchBuilder, DispatchSlice, KernelBuilder};

    #[derive(Debug)]
    pub(crate) struct KernelDesc {
        name: Cow<'static, str>,
        buffer_descs: &'static [BufferDesc],
        threads: UVec3,
        push_consts_size: u32,
    }

    impl KernelDesc {
        pub(crate) fn name(&self) -> &str {
            &self.name
        }
        pub(crate) fn buffer_descs(&self) -> &[BufferDesc] {
            &self.buffer_descs
        }
        pub(crate) fn threads(&self) -> UVec3 {
            self.threads
        }
        pub(crate) fn push_consts_size(&self) -> u32 {
            self.push_consts_size
        }
    }

    #[derive(Debug)]
    pub struct BufferDesc {
        name: &'static str,
        scalar_type: ScalarType,
        mutable: bool,
        item: bool,
    }

    impl BufferDesc {
        pub const fn new(name: &'static str, scalar_type: ScalarType) -> Self {
            Self {
                name,
                scalar_type,
                mutable: false,
                item: false,
            }
        }
        pub const fn with_mutable(mut self, mutable: bool) -> Self {
            self.mutable = mutable;
            self
        }
        pub(crate) fn mutable(&self) -> bool {
            self.mutable
        }
        pub const fn with_item(mut self, item: bool) -> Self {
            self.item = true;
            self
        }
    }

    pub struct Kernel {
        device: Device,
        #[cfg(feature = "device")]
        cache: Arc<KernelCache>,
    }

    impl Kernel {
        pub fn builder(name: impl Into<Cow<'static, str>>, spirv: &'static [u32]) -> KernelBuilder {
            KernelBuilder::new(name.into(), spirv)
        }
        pub fn dispatch_builder<'a>(
            &self,
            slices: &mut [DispatchSlice<'a>],
            push_consts: &[u8],
        ) -> Result<DispatchBuilder<'a>> {
            #[cfg(feature = "device")]
            {
                DispatchBuilder::new(self.device.clone(), self.cache.clone(), slices, push_consts)
            }
            #[cfg(not(feature = "device"))]
            {
                unreachable!()
            }
        }
    }
}

/*
pub struct KernelProto {
    desc: Arc<ProtoDesc>,
}

#[derive(Debug)]
pub(crate) struct ProtoDesc {
    spirv: Arc<Spirv>,
    name: String,
    entry: String,
    features: Features,
    thread_descs: [ThreadDesc; 3],
    spec_descs: Vec<SpecDesc>,
    slice_descs: Vec<SliceDesc>,
    push_consts_desc: PushConstsDesc,
}

// engine interface
impl ProtoDesc {
    pub(crate) fn entry_point(&self) -> &str {
        &self.entry
    }
    pub(crate) fn bindings(&self) -> impl Iterator<Item=(u32, bool)> + '_ {
        self.slice_descs.iter().map(|x| (x.binding, x.mutable))
    }
    pub(crate) fn push_consts_size(&self) -> u32 {
        self.push_consts_desc.size
    }
}

pub(crate) struct Spirv(Vec<u32>);

// engine interface
impl Spirv {
    pub(crate) fn words(&self) -> &[u32] {
        &self.0
    }
}

impl Debug for Spirv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Spirv")
            .field("len", &self.0.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
enum ThreadDesc {
    Constant(u32),
    SpecId(u32),
}

#[derive(Debug)]
struct SliceDesc {
    name: String,
    binding: u32,
    scalar_type: ScalarType,
    vector: Option<u32>,
    mutable: bool,
}

#[derive(Default, Debug)]
struct PushConstsDesc {
    push_descs: Vec<PushDesc>,
    size: u32,
}

#[derive(Debug)]
struct PushDesc {
    name: String,
    offset: u32,
    scalar_type: ScalarType,
    vector: Option<u32>,
}

#[derive(Debug)]
struct SpecDesc {
    name: String,
    id: u32,
    scalar_type: ScalarType,
}

impl KernelProto {
    pub fn from_spirv_words<'w>(name: impl Into<String>, words: &[u32]) -> Result<Self> {
        use rspirv::{
            dr::{Operand, Builder, InsertPoint, Instruction},
            spirv::{ExecutionModel, ExecutionMode, Op, StorageClass, Decoration, BuiltIn},
            binary::Assemble
        };
        use spirv_tools::opt::Passes;
        let kernel_name = name.into();
        let module = rspirv::dr::load_words(&words).map_err(|e| format_err!("{e}"))?;
        let (entry_id, entry) = if let [entry_point] = module.entry_points.as_slice() {
            if let [Operand::ExecutionModel(exec_model), Operand::IdRef(entry_id), Operand::LiteralString(entry), ..] = entry_point.operands.as_slice() {
                if *exec_model != ExecutionModel::GLCompute {
                    bail!("Kernel `{kernel_name}` execution model should be GLCompute, found {exec_model:?}!");
                }
                (*entry_id, entry.to_string())
            } else {
                bail!("Kernel `{kernel_name}` unable to parse entry point!");
            }
        } else {
            bail!("Kernel `{kernel_name}` should have 1 entry point!");
        };
        let exec_mode_inst = if let [exec_mode] = module.execution_modes.as_slice() {
                exec_mode
        } else {
            bail!("Kernel `{kernel_name}` should have 1 execution mode!");
        };
        let exec_mode = exec_mode_inst.operands[1].unwrap_execution_mode();
        let mut thread_descs = match exec_mode {
            ExecutionMode::LocalSize => {
                let x = exec_mode_inst.operands[2].unwrap_literal_int32();
                let y = exec_mode_inst.operands[3].unwrap_literal_int32();
                let z = exec_mode_inst.operands[4].unwrap_literal_int32();
                [ThreadDesc::Constant(x), ThreadDesc::Constant(y), ThreadDesc::Constant(z)]
            }
            ExecutionMode::LocalSizeId => {
                let x = exec_mode_inst.operands[2].unwrap_id_ref();
                let y = exec_mode_inst.operands[3].unwrap_id_ref();
                let z = exec_mode_inst.operands[4].unwrap_id_ref();
                [ThreadDesc::SpecId(x), ThreadDesc::SpecId(y), ThreadDesc::SpecId(z)]
            }
            other => {
                bail!("Kernel `{kernel_name}` execution mode should be LocalSize or LocalSizeId, found {other:?}!");
            }
        };
        let mut names = HashMap::<u32, &str>::new();
        let mut member_names = HashMap::<(u32, u32), &str>::new();
        for inst in module.debug_names.iter() {
            let op = inst.class.opcode;
            if op == Op::Name {
                if let [Operand::IdRef(id), Operand::LiteralString(name)] = inst.operands.as_slice() {
                    if !name.is_empty() {
                        names.insert(*id, name);
                    }
                }
            } else if op == Op::MemberName {
                if let [Operand::IdRef(id), Operand::LiteralInt32(index), Operand::LiteralString(name)] = inst.operands.as_slice() {
                    if !name.is_empty() {
                        member_names.insert((*id, *index), name);
                    }
                }
            }
        }
        let mut workgroup_size_id = Option::<u32>::None;
        let mut spec_ids = HashMap::<u32, u32>::new();
        let mut spec_names = HashMap::<u32, &str>::new();
        let mut bindings = HashMap::<u32, u32>::new();
        let mut non_writable = HashSet::<u32>::new();
        for inst in module.annotations.iter() {
            let op = inst.class.opcode;
            let operands = inst.operands.as_slice();
            if op == Op::Decorate {
                match operands {
                    [Operand::IdRef(id), Operand::Decoration(Decoration::BuiltIn), Operand::BuiltIn(BuiltIn::WorkgroupSize)] => {
                        workgroup_size_id.replace(*id);
                    }
                    [Operand::IdRef(id), Operand::Decoration(Decoration::SpecId), Operand::LiteralInt32(spec_id)] => {
                        spec_ids.insert(*id, *spec_id);
                        if let Some(name) = names.get(id) {
                            if let Some(other_name) = spec_names.get(spec_id) {
                                if name != other_name {
                                    bail!("Kernel `{kernel_name}` duplicate spec constants `name` and `name` for id `{spec_id}`!");
                                }
                            } else {
                                spec_names.insert(*spec_id, *name);
                            }
                        }
                    }
                    [Operand::IdRef(id), Operand::Decoration(Decoration::DescriptorSet), Operand::LiteralInt32(set)] => {
                        if *set != 0 {
                            bail!("Kernel `{kernel_name}` descriptor set mut be 0!");
                        }
                    }
                    [Operand::IdRef(id), Operand::Decoration(Decoration::Binding), Operand::LiteralInt32(binding)] => {
                        bindings.insert(*id, *binding);
                    }
                    [Operand::IdRef(id), Operand::Decoration(Decoration::NonWritable)] => {
                        non_writable.insert(*id);
                    }
                    [Operand::IdRef(_), Operand::Decoration(decoration @ Decoration::BufferBlock)] => {
                        // *decoration = Decoration::Block;
                    }
                    _ => (),
                }
            } else if op == Op::MemberDecorate {
                if let [Operand::IdRef(id), Operand::LiteralInt32(0), Operand::Decoration(Decoration::NonWritable)] = operands {
                    non_writable.insert(*id);
                }
            }
        }
        let mut const_u32s = HashMap::<u32, u32>::new();
        let mut spec_descs = HashMap::<u32, SpecDesc>::with_capacity(spec_ids.len());
        let mut vars = HashMap::<u32, (StorageClass, u32)>::new();
        let mut ptrs = HashMap::<u32, u32>::new();
        let mut structs = HashMap::<u32, Vec<u32>>::new();
        let mut dyn_arrays = HashMap::<u32, u32>::new();
        let mut vectors = HashMap::<u32, (u32, u32)>::new();
        let mut scalars = HashMap::<u32, ScalarType>::new();
        for inst in module.types_global_values.iter() {
            let op = inst.class.opcode;
            let operands = inst.operands.as_slice();
            match op {
                Op::Constant => {
                    if let [Operand::LiteralInt32(value)] = operands {
                        if let Some(id) = inst.result_id {
                            const_u32s.insert(id, *value);
                        }
                    }
                }
                Op::SpecConstant => {
                    if let Some((ty, id)) = inst.result_type.zip(inst.result_id) {
                        if let Some((&scalar_type, &spec_id)) = scalars.get(&ty).zip(spec_ids.get(&id)) {
                            if let Some(name) = spec_names.get(&spec_id) {
                                let desc = SpecDesc {
                                    name: name.to_string(),
                                    id: spec_id,
                                    scalar_type,
                                };
                                spec_descs.insert(spec_id, desc);
                            } else {
                                bail!("Kernel `{kernel_name}` spec constant with id {spec_id} must have name!");
                            }
                        }
                    }
                }
                Op::SpecConstantComposite => {
                    if let Some(workgroup_size_id) = workgroup_size_id {
                        if let [Operand::IdRef(x), Operand::IdRef(y), Operand::IdRef(z)] = operands {
                            if let Some(id) = inst.result_id {
                                if id == workgroup_size_id {
                                    for (id, desc) in [x, y, z].into_iter().zip(thread_descs.iter_mut()) {
                                        if let Some(constant) = const_u32s.get(id) {
                                            *desc = ThreadDesc::Constant(*constant);
                                        } else if let Some(spec_id) = spec_ids.get(id) {
                                            *desc = ThreadDesc::SpecId(*spec_id);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Op::Variable => {
                    if let [Operand::StorageClass(storage_class), ..] = operands {
                        use StorageClass::*;
                        if matches!(*storage_class, StorageBuffer | Uniform | PushConstant) {
                            if let Some((id, ptr)) = inst.result_id.zip(inst.result_type) {
                                if *storage_class == Uniform {
                                    // *storage_class = StorageBuffer;
                                }
                                vars.insert(id, (*storage_class, ptr));
                            }
                        } else if *storage_class != Input {
                            bail!("Kernel `{kernel_name}` unexpected storage class {storage_class:?}!");
                        }
                    }
                }
                Op::TypePointer => {
                    if let [Operand::StorageClass(storage_class), Operand::IdRef(ty)] = operands {
                        use StorageClass::*;
                        if matches!(*storage_class, StorageBuffer | Uniform | PushConstant) {
                            if let Some(id) = inst.result_id {
                                if *storage_class == Uniform {
                                    // *storage_class = StorageBuffer;
                                }
                                ptrs.insert(id, *ty);
                            }
                        }
                    }
                }
                Op::TypeStruct => {
                    if let Some(id) = inst.result_id {
                        let mut members = operands.into_iter()
                            .filter_map(|x| {
                                if let Operand::IdRef(id) = x {
                                    Some(*id)
                                } else {
                                    None
                                }
                            }).collect();
                        structs.insert(id, members);
                    }
                }
                Op::TypeRuntimeArray => {
                    if let [Operand::IdRef(elem)] = operands {
                        if let Some(id) = inst.result_id {
                            dyn_arrays.insert(id, *elem);
                        }
                    }
                }
                Op::TypeVector => {
                    if let [Operand::IdRef(elem), Operand::LiteralInt32(len)] = operands {
                        if let Some(id) = inst.result_id {
                            vectors.insert(id, (*elem, *len));
                        }
                    }
                }
                Op::TypeInt => {
                    if let [Operand::LiteralInt32(bits), Operand::LiteralInt32(sign)] = operands {
                        if let Some(id) = inst.result_id {
                            use ScalarType::*;
                            let scalar_type = if *sign == 0 {
                                match *bits {
                                    8 => U8,
                                    16 => U16,
                                    32 => U32,
                                    64 => U64,
                                    _ => {
                                        bail!("Kernel `{kernel_name}` {bits}-ints are not supported!");
                                    }
                                }
                            } else {
                                match *bits {
                                    8 => I8,
                                    16 => I16,
                                    32 => I32,
                                    64 => I64,
                                    _ => {
                                        bail!("Kernel `{kernel_name}` {bits}-ints are not supported!");
                                    }
                                }
                            };
                            scalars.insert(id, scalar_type);
                        }
                    }
                }
                Op::TypeFloat => {
                    if let [Operand::LiteralInt32(bits)] = operands {
                        if let Some(id) = inst.result_id {
                            use ScalarType::*;
                            let scalar_type = match *bits {
                                16 => F16,
                                32 => F32,
                                64 => F64,
                                _ => {
                                    bail!("Kernel `{kernel_name}` {bits}-floats are not supported!");
                                }
                            };
                            scalars.insert(id, scalar_type);
                        }
                    }
                }
                _ => (),
            }
        }
        let spec_descs: Vec<_> = spec_descs.into_values().collect();
        let mut slice_descs = Vec::with_capacity(bindings.len());
        let mut push_consts_desc = PushConstsDesc {
            push_descs: Vec::new(),
            size: 0,
        };
        let mut slice_ids = Vec::with_capacity(bindings.len());
        let mut push_consts_id = None;
        let mut push_consts_struct = None;
        let mut push_ids = Vec::new();
        for (var, (storage_class, ptr)) in vars.iter() {
            match *storage_class {
                StorageClass::StorageBuffer | StorageClass::Uniform => {
                    let binding = *bindings.get(var).ok_or_else(|| {
                        format_err!("Kernel `{kernel_name}` buffer does not have binding!")
                    })?;
                    let struct_ = ptrs.get(ptr).ok_or_else(|| {
                        format_err!("Kernel `{kernel_name}` unable to parse buffer with binding {binding}!")
                    })?;
                    let name = names.get(var).map(|x| x.to_string()).unwrap_or_default();
                    let name = if name.is_empty() {
                        member_names.get(&(*struct_, 0)).map(|x| x.to_string()).unwrap_or_default()
                    } else {
                        name
                    };
                    if name.is_empty() {
                        bail!("Kernel `{kernel_name}` buffer with binding {binding} must have name!");
                    }
                    let members = structs.get(struct_).ok_or_else(|| {
                        format_err!("Kernel `{kernel_name}` unable to parse buffer `{name}`!")
                    })?;
                    let dyn_array = if let [dyn_array] = members.as_slice() {
                        dyn_array
                    } else {
                        bail!("Kernel `{kernel_name}` buffer `{name}` expected struct with 1 field!");
                    };
                    let element = dyn_arrays.get(dyn_array).ok_or_else(|| {
                        format_err!("Kernel `{kernel_name}` buffer `{name}` id not found for array element!")
                    })?;
                    let (scalar_type, vector) = if let Some((scalar, len)) = vectors.get(element) {
                        let scalar_type = scalars.get(scalar).ok_or_else(|| {
                            format_err!("Kernel `{kernel_name}` buffer `{name}` id not found for vector element")
                        })?;
                        (*scalar_type, Some(*len))
                    } else {
                        let scalar_type = scalars.get(element).ok_or_else(|| {
                            format_err!("Kernel `{kernel_name}` buffer `{name}` id not found for scalar")
                        })?;
                        (*scalar_type, None)
                    };
                    let mutable = !([var, struct_, dyn_array].iter().any(|x| non_writable.contains(x)));
                    let desc = SliceDesc {
                        name,
                        binding,
                        scalar_type,
                        vector,
                        mutable,
                    };
                    slice_descs.push(desc);
                    slice_ids.push(*var);
                }
                StorageClass::PushConstant => {
                    if push_consts_desc.size != 0 {
                        bail!("Kernel `{kernel_name}` cannot have more than 1 push constant block!");
                    }
                    let struct_ = ptrs.get(ptr).ok_or_else(|| {
                        format_err!("Kernel `{kernel_name}` unable to parse push constant block!")
                    })?;
                    let members = structs.get(struct_).ok_or_else(|| {
                        format_err!("Kernel `{kernel_name}` unable to parse push constants!")
                    })?;
                    push_consts_desc.push_descs = Vec::with_capacity(members.len());
                    for (i, member) in members.into_iter().enumerate() {
                        let name = member_names.get(&(*struct_, 0)).map(|x| x.to_string()).unwrap_or_default();
                        if name.is_empty() {
                            bail!("Kernel `{kernel_name}` push constant field at index {i} must have name!");
                        }
                        let (scalar_type, vector) = if let Some((scalar, len)) = vectors.get(member) {
                            let scalar_type = scalars.get(scalar).ok_or_else(|| {
                                format_err!("Kernel `{kernel_name}` push `{name}` id not found for vector element")
                            })?;
                            (*scalar_type, Some(*len))
                        } else {
                            let scalar_type = scalars.get(member).ok_or_else(|| {
                                format_err!("Kernel `{kernel_name}` push `{name}` id not found for scalar")
                            })?;
                            (*scalar_type, None)
                        };
                        let desc = PushDesc {
                            name,
                            offset: push_consts_desc.size,
                            scalar_type,
                            vector,
                        };
                        push_consts_desc.push_descs.push(desc);
                        push_consts_desc.size += scalar_type.size() as u32 * vector.unwrap_or(1);
                    }
                    push_consts_id.replace(*var);
                    push_consts_struct.replace(*struct_);
                    push_ids = members.to_vec();
                }
                _ => {
                    if cfg!(debug_assertions) {
                        unreachable!("{storage_class:?}");
                    }
                }
            }
        }
        /*let (major, mut minor) = module.header.as_ref().map(|x| x.version()).unwrap_or((1, 3));
        if major > 1 {
            todo!();
        }
        if minor < 3 {
            minor = 3;
        }
        let mut builder = Builder::new_from_module(module);
        builder.set_version(major, minor);*/
        let mut module = module;
        if !slice_descs.is_empty() {
            let entry_fn_index = module
                .functions
                .iter()
                .position(|f| f.def.as_ref().unwrap().result_id.unwrap() == entry_id).unwrap();
            let mut builder = Builder::new_from_module(module);
            let type_i32 = if let Some(type_i32_index) = builder.module_ref().types_global_values.iter().position(|inst| {
                let op = inst.class.opcode;
                let operands = inst.operands.as_slice();
                op == Op::TypeInt && matches!(operands, [Operand::LiteralInt32(32), Operand::LiteralInt32(1)])
            }) {
                let inst = builder.module_mut().types_global_values.remove(type_i32_index);
                let type_i32 = inst.result_id.unwrap();
                builder.module_mut().types_global_values.insert(0, inst);
                type_i32
            } else {
                let type_i32 = builder.id();
                let inst = Instruction::new(
                    Op::TypeInt,
                    None,
                    Some(type_i32),
                    vec![Operand::LiteralInt32(32), Operand::LiteralInt32(1)],
                );
                builder.insert_types_global_values(InsertPoint::Begin, inst);
                type_i32
            };
            let type_u32 = if let Some(type_u32_index) = builder.module_ref().types_global_values.iter().position(|inst| {
                let op = inst.class.opcode;
                let operands = inst.operands.as_slice();
                op == Op::TypeInt && matches!(operands, [Operand::LiteralInt32(32), Operand::LiteralInt32(0)])
            }) {
                let inst = builder.module_mut().types_global_values.remove(type_u32_index);
                let type_u32 = inst.result_id.unwrap();
                builder.module_mut().types_global_values.insert(0, inst);
                type_u32
            } else {
                let type_u32 = builder.id();
                let inst = Instruction::new(
                    Op::TypeInt,
                    None,
                    Some(type_u32),
                    vec![Operand::LiteralInt32(32), Operand::LiteralInt32(0)],
                );
                builder.insert_types_global_values(InsertPoint::Begin, inst);
                type_u32
            };
            let type_ptr_u32 = builder.type_pointer(None, StorageClass::PushConstant, type_u32);
            let type_ptr_i32 = builder.type_pointer(None, StorageClass::PushConstant, type_i32);
            let (push_consts_id, push_consts_struct) = if let Some((push_consts_id, push_consts_struct)) = push_consts_id.zip(push_consts_struct) {
                let inst = builder.module_mut().types_global_values.iter_mut().find(|x| x.result_id.as_ref() == Some(&push_consts_struct)).unwrap();
                inst.operands.extend(std::iter::repeat(type_u32).take(2 * slice_descs.len()).map(Operand::IdRef));
                (push_consts_id, push_consts_struct)
            } else {
                let push_consts_struct = builder.type_struct(std::iter::repeat(type_u32).take(2 * slice_descs.len()));
                builder.decorate(push_consts_struct, Decoration::Block, []);
                let push_consts_ptr = builder.type_pointer(None, StorageClass::PushConstant, push_consts_struct);
                let push_consts_id = builder.variable(push_consts_ptr, None, StorageClass::PushConstant, None);
                (push_consts_id, push_consts_struct)
            };
            let mut push_consts_struct_offset = push_consts_desc.size;
            for i in 0 .. (slice_descs.len() * 2) {
                let member = (i + push_consts_desc.push_descs.len()) as u32;
                let offset = i as u32 * 4 + push_consts_desc.size;
                builder.member_decorate(push_consts_struct, member, Decoration::Offset, [Operand::LiteralInt32(offset)]);
            }
            {
                let mut member = push_consts_desc.push_descs.len() as u32;
                for slice_desc in slice_descs.iter() {
                    builder.member_name(push_consts_struct, member, format!("__krnl_offset__{}", slice_desc.name));
                    member += 1;
                    builder.member_name(push_consts_struct, member, format!("__krnl_len__{}", slice_desc.name));
                }
            }
            // dynamic offset and len
            {
                let mut b = 0;
                let mut i = 0;
                loop {
                    #[derive(Debug)]
                    enum DynOp {
                        Offset {
                            index: u32,
                            slice_index: u32,
                        },
                        Len {
                            slice_index: u32,
                        }
                    }
                    let mut dyn_op = None;
                    'block: for block in builder.module_mut().functions[entry_fn_index].blocks.iter_mut().skip(b) {
                        for inst in block.instructions.iter_mut().skip(i) {
                            let op = inst.class.opcode;
                            if op == Op::AccessChain {
                                let var = inst.operands[0].unwrap_id_ref();
                                if let Some(slice_index) = slice_ids.iter().position(|id| var == *id) {
                                    let index = inst.operands[2].unwrap_id_ref();
                                    dyn_op.replace(DynOp::Offset {
                                        index,
                                        slice_index: slice_index as u32,
                                    });
                                    break 'block;
                                }
                            } else if op == Op::ArrayLength {
                                let var = inst.operands[0].unwrap_id_ref();
                                if let Some(slice_index) = slice_ids.iter().position(|id| var == *id) {
                                    dyn_op.replace(DynOp::Len {
                                        slice_index: slice_index as u32,
                                    });
                                    break 'block;
                                }
                            }
                            i += 1;
                        }
                        b += 1;
                    }
                    if let Some(dyn_op) = dyn_op.take() {
                        builder.select_function(Some(entry_fn_index));
                        builder.select_block(Some(b));
                        let n_push_descs = push_consts_desc.push_descs.len() as u32;
                        match dyn_op {
                            DynOp::Offset {
                                index,
                                slice_index,
                            } => {
                                let member = builder.constant_u32(type_i32, 2 * slice_index + n_push_descs);
                                let offset_ptr = builder.insert_access_chain(InsertPoint::FromBegin(i), type_ptr_u32, None, push_consts_id, [member])?;
                                i += 1;
                                let offset = builder.insert_load(InsertPoint::FromBegin(i), type_u32, None, offset_ptr, None, [])?;
                                i += 1;
                                let index = builder.insert_i_add(InsertPoint::FromBegin(i), type_u32, None, index, offset)?;
                                i += 1;
                                let inst = &mut builder.module_mut().functions[entry_fn_index].blocks[b].instructions[i];
                                inst.operands[2] = Operand::IdRef(index);
                                i += 1;
                            }
                            DynOp::Len {
                                slice_index,
                            } => {
                                let inst = builder.module_mut().functions[entry_fn_index].blocks[b].instructions.remove(i);
                                let member = builder.constant_u32(type_i32, 2 * slice_index + 1 + n_push_descs);
                                let len_ptr = builder.insert_access_chain(InsertPoint::FromBegin(i), type_ptr_u32, None, push_consts_id, [member])?;
                                i += 1;
                                let load = builder.insert_load(InsertPoint::FromBegin(i), type_u32, inst.result_id, len_ptr, None, [])?;
                                i += 1;
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
            module = builder.module();
        }
        let words = module.assemble();
        let (features, spirv) = optimize_spirv(&kernel_name, &words, &[])?;
        let desc = Arc::new(ProtoDesc {
            spirv: Arc::new(spirv),
            name: kernel_name,
            entry,
            features,
            thread_descs,
            spec_descs,
            slice_descs,
            push_consts_desc,
        });
        Ok(Self {
            desc,
        })
    }
    pub fn kernel_builder(&self) -> KernelBuilder {
        let proto_desc = self.desc.clone();
        let spec_consts = HashMap::with_capacity(proto_desc.spec_descs.len());
        KernelBuilder {
            proto_desc,
            spec_consts,
        }
    }
}

fn optimize_spirv(kernel_name: &str, words: &[u32], passes: &[spirv_tools::opt::Passes]) -> Result<(Features, Spirv)> {
    use spirv_tools::{opt::{Optimizer, Passes}, val::Validator, TargetEnv};
    use rspirv::{dr::{Builder, Operand}, binary::Assemble, spirv::{Op, Capability}};

    #[cfg(debug_assertions)] {
        spirv_tools::val::create(None).validate(words, None).unwrap();
    }

    let mut optimizer = spirv_tools::opt::create(None);
    for pass in passes {
        optimizer.register_pass(*pass);
    }
    optimizer.register_performance_passes()
        .register_size_passes();
    let binary = optimizer.optimize(words.as_ref(), &mut |_| (), None)?;
    let mut module = rspirv::dr::load_words(words/*binary.as_words()*/).map_err(|e| format_err!("{e}"))?;
    module.capabilities.clear();
    module.extensions.clear();
    let mut capabilities = HashSet::new();
    let mut extensions = HashSet::<&'static str>::new();
    for inst in module.all_inst_iter() {
        let class = inst.class;
        let op = class.opcode;
        match (op, inst.operands.first()) {
            (Op::TypeInt, Some(Operand::LiteralInt32(8))) => {
                capabilities.insert(Capability::Int8);
            }
            (Op::TypeInt, Some(Operand::LiteralInt32(16))) => {
                capabilities.insert(Capability::Int16);
            }
            (Op::TypeInt, Some(Operand::LiteralInt32(64))) => {
                capabilities.insert(Capability::Int64);
            }
            (Op::TypeFloat, Some(Operand::LiteralInt32(16))) => {
                capabilities.insert(Capability::Float16);
            }
            (Op::TypeFloat, Some(Operand::LiteralInt32(64))) => {
                capabilities.insert(Capability::Float64);
            }
            _ => (),
        }
        capabilities.extend(class.capabilities);
        extensions.extend(class.extensions);
    }
    let mut builder = Builder::new_from_module(module);
    let core_capabilities = [Capability::Shader, Capability::VulkanMemoryModel];
    let mut features = Features::default();
    for cap in core_capabilities.into_iter().chain(capabilities.iter().copied()) {
        use Capability::*;
        match cap {
            Int8 => {
                features.set_shader_int8(true);
            }
            Int16 => {
                features.set_shader_int16(true);
            }
            Int64 => {
                features.set_shader_int64(true);
            }
            Float64 => {
                features.set_shader_float64(true);
            }
            Shader | VulkanMemoryModel => (),
            _ => {
                bail!("Kernel `{kernel_name}` has unsupported capability {cap:?}!");
            }
        }
        for ext in extensions.iter().copied() {
            match ext {
                _ => {
                    bail!("Kernel `{kernel_name}` has unsupported extension {ext:?}!");
                }
            }
            builder.extension(ext);
        }
        builder.capability(cap);
    }
    let spirv = Spirv(builder.module().assemble());
    Ok((features, spirv))
}

pub struct KernelBuilder {
    proto_desc: Arc<ProtoDesc>,
    spec_consts: HashMap<u32, ScalarElem>,
}

impl KernelBuilder {
    pub fn spec(mut self, name: &str, spec: impl Into<ScalarElem>) -> Result<Self> {
        let kernel_name = &self.proto_desc.name;
        let spec = spec.into();
        if let Some(spec_desc) = self.proto_desc.spec_descs.iter().find(|x| x.name == name) {
            if spec_desc.scalar_type == spec.scalar_type() {
                self.spec_consts.insert(spec_desc.id, spec);
            } else {
                let expected = spec_desc.scalar_type;
                let found = spec.scalar_type();
                bail!("Kernel `{kernel_name}` spec `{name}` expected {expected:?} found {found:?}!");
            }
        } else {
            bail!("Kernel `{kernel_name}` unexpected spec `{name}`!");
        }
        Ok(self)
    }
    pub fn build(self, device: Device) -> Result<Kernel> {
        let proto_desc = &self.proto_desc;
        let kernel_name = &proto_desc.name;
        match device.inner {
            DeviceInner::Host => Err(
                format_err!("Kernel `{kernel_name}` can only be built for a device, found host!")
            ),
            #[cfg(feature = "device")]
            DeviceInner::Device(vulkan_device) => self.build_impl(vulkan_device),
        }
    }
    #[cfg(feature = "device")]
    fn build_impl(self, vulkan_device: VulkanDevice) -> Result<Kernel> {
        use rspirv::{dr::{Builder, Operand, Instruction}, binary::{Assemble, Disassemble}, spirv::{Op, Decoration, Capability}};
        use spirv_tools::opt::Passes;
        let proto_desc = &self.proto_desc;
        let kernel_name = &proto_desc.name;
        fn extend_scalar_elem_bytes(vec: &mut Vec<u8>, elem: ScalarElem) {
            use ScalarElem::*;
            match elem {
                U8(x) => vec.push(x),
                I8(x) => vec.push(x.to_ne_bytes()[0]),
                U16(x) => vec.extend_from_slice(&x.to_ne_bytes()),
                I16(x) => vec.extend_from_slice(&x.to_ne_bytes()),
                // TODO f16, bf16
                U32(x) => vec.extend_from_slice(&x.to_ne_bytes()),
                I32(x) => vec.extend_from_slice(&x.to_ne_bytes()),
                F32(x) => vec.extend_from_slice(&x.to_ne_bytes()),
                U64(x) => vec.extend_from_slice(&x.to_ne_bytes()),
                I64(x) => vec.extend_from_slice(&x.to_ne_bytes()),
                F64(x) => vec.extend_from_slice(&x.to_ne_bytes()),
            }
        }
        fn update_spec_const(inst: &mut Instruction, spec: ScalarElem) {
            use rspirv::dr::Operand::{LiteralInt32 as Int32, LiteralInt64 as Int64, LiteralFloat32 as Float32, LiteralFloat64 as Float64};
            use ScalarElem::*;
            inst.operands[0] = match spec {
                U8(x) => Int32(x.into()),
                I8(x) => Int32(x.to_ne_bytes()[0].into()),
                U16(x) => Int32(x.into()),
                I16(x) => Int32(u16::from_ne_bytes(x.to_ne_bytes()).into()),
                // TODO f16, bf16
                U32(x) => Int32(x),
                I32(x) => Int32(u32::from_ne_bytes(x.to_ne_bytes())),
                F32(x) => Float32(x),
                U64(x) => Int64(x),
                I64(x) => Int64(u64::from_ne_bytes(x.to_ne_bytes())),
                F64(x) => Float64(x),
            }
        }
        let spec_descs = &proto_desc.spec_descs;
        let spec_const_bytes = if !spec_descs.is_empty() {
            let spec_consts_size = spec_descs.iter().map(|x| x.scalar_type.size()).sum();
            let mut spec_const_bytes = Vec::with_capacity(spec_consts_size);
            for spec_desc in spec_descs.iter() {
                if let Some(spec) = self.spec_consts.get(&spec_desc.id) {
                    extend_scalar_elem_bytes(&mut spec_const_bytes, *spec);
                } else {
                    let name = &spec_desc.name;
                    bail!("Kernel `{kernel_name}` expected spec `{name}`!");
                }
            }
            Some(Arc::from(spec_const_bytes))
        } else {
            None
        };
        let cache = vulkan_device.kernel_cache(
            self.proto_desc.clone(),
            spec_const_bytes,
            || {
            let (features, spirv) = if !proto_desc.spec_descs.is_empty() {
                let mut module = rspirv::dr::load_words(&proto_desc.spirv.0).map_err(|e| format_err!("{e}"))?;
                let mut spec_ids = HashMap::with_capacity(proto_desc.spec_descs.len());
                for inst in module.annotations.iter() {
                    let op = inst.class.opcode;
                    let operands = inst.operands.as_slice();
                    if op == Op::Decorate {
                        if let [Operand::IdRef(id), Operand::Decoration(Decoration::SpecId), Operand::LiteralInt32(spec_id)] = operands {
                            debug_assert!(proto_desc.spec_descs.iter().find(|x| x.id == *spec_id).is_some());
                            spec_ids.insert(*id, *spec_id);
                        }
                    }
                }
                for inst in module.types_global_values.iter_mut() {
                    if let Some(id) = inst.result_id {
                        if let Some(spec_id) = spec_ids.get(&id) {
                            let spec_id = spec_ids.get(&id).unwrap();
                            let spec = self.spec_consts.get(spec_id).unwrap();
                            update_spec_const(inst, *spec);
                        }
                    }
                }
                let words = module.assemble();
                let (features, spirv) = optimize_spirv(kernel_name, &words, &[Passes::FreezeSpecConstantValue])?;
                (features, Arc::new(spirv))
            } else {
                (proto_desc.features, proto_desc.spirv.clone())
            };
            let device_features = vulkan_device.features();
            if !device_features.contains(&features) {
                bail!("Kernel `{kernel_name}` requires {features:?}, device has {device_features:?}!");
            }
            Ok(spirv)
        })?;
        Ok(Kernel {
            vulkan_device: vulkan_device.clone(),
            cache,
        })
    }
}

pub struct Kernel {
    #[cfg(feature = "device")]
    vulkan_device: VulkanDevice,
    #[cfg(feature = "device")]
    cache: Arc<KernelCache>,
}

#[cfg(feature = "device")]
impl Kernel {
    pub fn dispatch_builder(&self) -> DispatchBuilder {
        let proto_desc = self.cache.proto_desc();
        let buffers = HashMap::with_capacity(proto_desc.slice_descs.len());
        let n_push_descs = proto_desc.push_consts_desc.push_descs.len();
        let push_const_indices = HashSet::with_capacity(n_push_descs);
        let mut push_consts_size = proto_desc.push_consts_desc.size as usize;
        if push_consts_size % 4 != 0 {
            push_consts_size += 4 - (push_consts_size % 4);
        }
        let hidden_push_consts_size = 2 * proto_desc.slice_descs.len();
        let push_consts = vec![0u32; push_consts_size + hidden_push_consts_size];
        let hidden_start = push_consts_size / 4;
        DispatchBuilder {
            inner: DispatchBuilderInner {
                vulkan_device: self.vulkan_device.clone(),
                cache: self.cache.clone(),
                groups: None,
                buffers,
                push_const_indices,
                push_consts,
                hidden_start,
            },
            _m: PhantomData::default(),
        }
    }
}

#[cfg(feature = "device")]
struct DispatchBuilderInner {
    vulkan_device: VulkanDevice,
    cache: Arc<KernelCache>,
    groups: Option<Vec3<u32>>,
    buffers: HashMap<u32, Arc<DeviceBufferInner>>,
    push_const_indices: HashSet<u32>,
    push_consts: Vec<u32>,
    hidden_start: usize,
}

#[cfg(feature = "device")]
pub struct DispatchBuilder<'a> {
    inner: DispatchBuilderInner,
    _m: PhantomData<&'a ()>,
}

#[cfg(feature = "device")]
impl<'a> DispatchBuilder<'a> {
    /*
    errors
        - check name
        - must not be mutable
        - not empty
    */
    pub fn slice<'b>(self, name: &str, slice: impl Into<ScalarSlice<'b>>) -> Result<DispatchBuilder<'b>> where 'b: 'a {
        let slice = slice.into().into_raw_slice();
        unsafe {
            self.slice_impl(name, slice, false)
        }
    }
    /*
        - check name
        - not empty
    */
    pub fn slice_mut<'b>(self, name: &str, slice: impl Into<ScalarSliceMut<'b>>) -> Result<DispatchBuilder<'b>> where 'b: 'a {
        let slice = slice.into().as_scalar_slice().into_raw_slice();
        unsafe {
            self.slice_impl(name, slice, true)
        }
    }
    unsafe fn slice_impl<'b>(self, name: &str, slice: crate::buffer::RawSlice, mutable: bool) -> Result<DispatchBuilder<'b>> where 'b: 'a {
        let mut inner = self.inner;
        let proto_desc = inner.cache.proto_desc();
        let kernel_name = &proto_desc.name;
        if let Some((i, slice_desc)) = proto_desc.slice_descs.iter().enumerate().find(|(_, x)| x.name == name) {
            let slice_device = slice.device_ref();
            if slice.scalar_type() != slice_desc.scalar_type {
                let expected = slice_desc.scalar_type;
                let found = slice.scalar_type();
                bail!("Kernel `{kernel_name}` push `{name}` expected {expected:?} found {found:?}!");
            }
            if slice_device.as_device() != Some(&inner.vulkan_device) {
                let vulkan_device = &inner.vulkan_device;
                bail!("Kernel `{kernel_name}` slice `{name}` expected {slice_device:?} found {vulkan_device:?}!");
            }
            if slice.is_empty() {
                bail!("Kernel `{kernel_name}` slice `{name}` is empty!");
            }
            if !mutable && slice_desc.mutable {
                bail!("Kernel `{kernel_name}` modifies slice `{name}`, must be mutable!");
            }
            let buffer = slice.as_device_slice()
                .unwrap()
                .device_buffer()
                .unwrap()
                .inner();
            inner.buffers.insert(slice_desc.binding, buffer);
            let hidden_start = inner.hidden_start;
            let index = inner.hidden_start + 2 * i;
            inner.push_consts[index] = 0; // TODO offset
            inner.push_consts[index+1] = slice.len() as u32;
            Ok(DispatchBuilder {
                inner,
                _m: PhantomData::default(),
            })
        } else {
            bail!("Kernel `{kernel_name}` unexpected slice `{name}`!");
        }
    }
    /*
        - check name
        - check type
    */
    pub fn push(mut self, name: &str, push: impl Into<ScalarElem>) -> Result<Self> {
        let mut inner = &mut self.inner;
        let proto_desc = inner.cache.proto_desc();
        let kernel_name = &proto_desc.name;
        if let Some((i, push_desc)) = proto_desc.push_consts_desc.push_descs.iter().enumerate().find(|(_, x)| x.name == name) {
            let push = push.into();
            if push.scalar_type() != push_desc.scalar_type {
                let expected = push_desc.scalar_type;
                let found = push.scalar_type();
                bail!("Kernel `{kernel_name}` push `{name}` expected {expected:?} found {found:?}!");
            }
            let push_consts = bytemuck::cast_slice_mut(&mut inner.push_consts);
            let offset = push_desc.offset as usize;
            let size = push_desc.scalar_type.size();
            let push_bytes = &mut push_consts[offset..offset + size];
            use ScalarElem::*;
            use bytemuck::cast_slice;
            match push {
                U8(x) => {
                    push_bytes.copy_from_slice([x].as_ref());
                }
                I8(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
                U16(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
                I16(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
                // TODO f16, bf16
                U32(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
                I32(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
                F32(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
                U64(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
                I64(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
                F64(x) => {
                    push_bytes.copy_from_slice(x.to_ne_bytes().as_ref());
                }
            }
            inner.push_const_indices.insert(i as u32);
        } else {
            bail!("Kernel `{kernel_name}` unexpected push `{name}`!");
        }
        Ok(self)
    }
    pub fn push_vector(self, name: &str, vector: &[impl Into<ScalarElem>]) -> Result<Self> {
        todo!()
    }
    /*
    errors
        - slices
            - cannot be missing
        - push consts
            - cannot be missing
        - the device panicked
    # Safety
        - The caller should ensure that the kernel does not:
            - Index a slice or array out of bounds
            - Improperly handle shared access to global, group, or subgroup memory
        - Panics in kernel execution will not be caught
    */
    pub fn global_threads(mut self, global_threads: impl Into<Vec3<u32>>) -> Result<Self> {
        todo!()
    }
    pub fn groups(mut self, groups: impl Into<Vec3<u32>>) -> Result<Self> {
        let mut inner = self.inner;
        let proto_desc = inner.cache.proto_desc();
        let kernel_name = &proto_desc.name;
        let groups = groups.into();
        if groups.iter().any(|x| *x == 0) {
            bail!("Kernel `{kernel_name}` group dims cannot be 0, found {groups:?}!");
        }
        inner.groups.replace(groups);
        Ok(Self {
            inner,
            _m: PhantomData::default(),
        })
    }
    pub unsafe fn dispatch(self) -> Result<()> {
        use std::fmt::Write;
        let inner = self.inner;
        let proto_desc = inner.cache.proto_desc();
        let kernel_name = &proto_desc.name;
        let missing_slice = inner.buffers.len() != proto_desc.slice_descs.len();
        let missing_push_consts = inner.push_const_indices.len() != proto_desc.push_consts_desc.push_descs.len();
        if missing_slice || missing_push_consts {
            let mut msg = String::new();
            for slice_desc in proto_desc.slice_descs.iter() {
                if !inner.buffers.contains_key(&slice_desc.binding) {
                    if !msg.is_empty() {
                        msg.push(',');
                    }
                    write!(&mut msg, " slice `{}`", slice_desc.name);
                }
            }
            for (i, push_desc) in proto_desc.push_consts_desc.push_descs.iter().enumerate() {
                let index = i as u32;
                if !inner.push_const_indices.contains(&index) {
                    if !msg.is_empty() {
                        msg.push(',');
                    }
                    write!(&mut msg, " push `{}`", push_desc.name);
                }
            }
            bail!("Kernel `{kernel_name}` expected{msg}!");
        }
        let groups = if let Some(groups) = inner.groups {
            groups.into_array()
        } else {
            bail!("Kernel `{kernel_name}` expected `global_threads` or `groups`!");
        };
        let compute = Compute {
            cache: inner.cache,
            groups,
            buffers: inner.buffers,
            push_consts: inner.push_consts,
        };
        inner.vulkan_device.compute(compute)
    }
}

#[cfg(all(test, feature = "device"))]
mod tests {
    use super::*;
    use crate::{
        buffer::Slice,
        future::BlockableFuture,
    };
    use shaderc::{Compiler, ShaderKind, CompileOptions, TargetEnv, EnvVersion, SpirvVersion};

    static GLSL: &'static str = r#"
        #version 450

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer X {
            float x[];
        };

        layout(binding = 1) buffer Y {
            float y[];
        };

        layout(push_constant) uniform P {
            float alpha;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx < min(x.length(), y.length())) {
                y[idx] += alpha * x[idx];
            }
        }
    "#;

    #[test]
    fn foo() -> Result<()> {;
        let compiler = Compiler::new().unwrap();
        let mut compile_options = CompileOptions::new().unwrap();
        //compile_options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_ as u32);
        //compile_options.set_target_spirv(SpirvVersion::V1_5);
        //let asm = compiler.compile_into_spirv_assembly(GLSL, ShaderKind::Compute, "foo", "foo", Some(&compile_options))?.as_text();
        //println!("{asm}");
        let spv = compiler.compile_into_spirv(GLSL, ShaderKind::Compute, "foo", "foo", Some(&compile_options))?.as_binary().to_vec();
        let proto = KernelProto::from_spirv_words("foo", &spv)?;
        let device = Device::new(0)?;
        let mut x = Slice::from([1f32; 10].as_ref()).into_device(device.clone())?.block()?;
        let alpha = 1f32;
        let mut y = Slice::from([0f32; 10].as_ref()).into_device(device.clone())?.block()?;
        let kernel = proto.kernel_builder()
            .build(device.clone())?;
        let builder = kernel.dispatch_builder()
            .slice("x", x.as_slice())?
            .push("alpha", alpha)?
            .slice_mut("y", y.as_slice_mut())?
            .groups([1, 1, 1])?;
        unsafe {
            builder.dispatch()?;
        }
        let y = y.to_vec()?.block()?;
        dbg!(y);
        todo!()
    }
}
*/
