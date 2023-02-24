use crate::scalar::ScalarElem;
#[cfg(feature = "device")]
use anyhow::format_err;
use anyhow::Result;
use rspirv::binary::Disassemble;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{self, Debug},
    mem::{forget, size_of},
    ops::RangeBounds,
    sync::Arc,
    time::Duration,
};

#[cfg(feature = "device")]
mod vulkan_engine;
#[cfg(feature = "device")]
use vulkan_engine::Engine;

mod error {
    use std::fmt::{self, Debug, Display};

    #[derive(Clone, Copy, Debug, thiserror::Error)]
    #[error("DeviceUnavailable")]
    pub(super) struct DeviceUnavailable;

    #[cfg(feature = "device")]
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    pub(super) struct DeviceIndexOutOfRange {
        pub(super) index: usize,
        pub(super) devices: usize,
    }

    #[cfg(feature = "device")]
    impl Display for DeviceIndexOutOfRange {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(self, f)
        }
    }
    #[cfg(feature = "device")]
    pub(super) struct DeviceNotSupported;

    #[derive(Clone, Copy, thiserror::Error)]
    pub struct DeviceLost {
        #[cfg(feature = "device")]
        pub(super) index: usize,
        #[cfg(feature = "device")]
        pub(super) handle: u64,
    }

    impl Debug for DeviceLost {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            #[cfg(feature = "device")]
            {
                f.debug_tuple("DeviceLost")
                    .field(&self.index)
                    .field(&(self.handle as *const ()))
                    .finish()
            }
            #[cfg(not(feature = "device"))]
            {
                write!(f, "DeviceLost")
            }
        }
    }

    impl Display for DeviceLost {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(self, f)
        }
    }
}
use error::*;

pub mod builder {
    use super::*;

    pub struct DeviceBuilder {
        #[cfg(feature = "device")]
        pub(super) options: DeviceOptions,
    }

    impl DeviceBuilder {
        pub fn index(self, index: usize) -> Self {
            #[cfg(feature = "device")]
            {
                let mut this = self;
                this.options.index = index;
                this
            }
            #[cfg(not(feature = "device"))]
            {
                self
            }
        }
        pub fn build(self) -> Result<Device> {
            #[cfg(feature = "device")]
            {
                let raw = RawDevice::new(self.options)?;
                Ok(Device {
                    inner: DeviceInner::Device(raw),
                })
            }
            #[cfg(not(feature = "device"))]
            {
                Err(DeviceUnavailable.into())
            }
        }
    }
}
use builder::*;

#[cfg(feature = "device")]
trait DeviceEngine {
    type DeviceBuffer: DeviceEngineBuffer<Engine = Self>;
    type Kernel: DeviceEngineKernel<Engine = Self, DeviceBuffer = Self::DeviceBuffer>;
    fn new(options: DeviceOptions) -> Result<Arc<Self>>;
    fn handle(&self) -> u64;
    fn info(&self) -> &Arc<DeviceInfo>;
    fn wait(&self) -> Result<(), DeviceLost>;
    //fn performance_metrics(&self) -> PerformanceMetrics;
}

#[cfg(feature = "device")]
struct DeviceOptions {
    index: usize,
    optimal_features: Features,
}

struct KernelProto {
    spirv: Arc<Vec<u32>>,
    buffers: Vec<bool>,
    push_consts: u32,
    spec_consts: Vec<ScalarElem>,
}

#[cfg(feature = "device")]
trait DeviceEngineBuffer: Sized {
    type Engine;
    unsafe fn uninit(engine: Arc<Self::Engine>, len: usize) -> Result<Arc<Self>>;
    fn upload(engine: Arc<Self::Engine>, data: &[u8]) -> Result<Arc<Self>>;
    fn download(&self, data: &mut [u8]) -> Result<()>;
    fn transfer(&self, engine: Arc<Self::Engine>) -> Result<Arc<Self>>;
    fn engine(&self) -> &Arc<Self::Engine>;
    fn len(&self) -> usize;
    fn slice(self: &Arc<Self>, bounds: impl RangeBounds<usize>) -> Option<Arc<Self>>;
}

#[cfg(feature = "device")]
trait DeviceEngineKernel: Sized {
    type Engine;
    type DeviceBuffer;
    fn new(
        engine: Arc<Self::Engine>,
        spirv: Arc<[u32]>,
        spec_consts: &[ScalarElem],
    ) -> Result<Arc<Self>>;
    unsafe fn dispatch(
        &self,
        groups: [u32; 3],
        buffers: &[Arc<Self::DeviceBuffer>],
        push_consts: &[ScalarElem],
    ) -> Result<()>;
    fn engine(&self) -> &Arc<Self::Engine>;
    fn threads(&self) -> [u32; 3];
    fn features(&self) -> Features;
    fn buffer_descs(&self) -> &[BufferDesc];
}

#[derive(Clone, Eq, PartialEq)]
pub struct Device {
    inner: DeviceInner,
}

impl Device {
    pub const fn host() -> Self {
        Self {
            inner: DeviceInner::Host,
        }
    }
    pub fn builder() -> DeviceBuilder {
        DeviceBuilder {
            #[cfg(feature = "device")]
            options: DeviceOptions {
                index: 0,
                optimal_features: Features::empty()
                    .with_shader_int8(true)
                    .with_shader_int16(true)
                    .with_shader_int64(true)
                    .with_shader_float16(true)
                    .with_shader_float64(true),
            },
        }
    }
    pub fn is_host(&self) -> bool {
        self.inner.is_host()
    }
    pub fn is_device(&self) -> bool {
        self.inner.is_device()
    }
    pub(crate) fn inner(&self) -> &DeviceInner {
        &self.inner
    }
    pub fn info(&self) -> Option<&Arc<DeviceInfo>> {
        match self.inner() {
            DeviceInner::Host => None,
            #[cfg(feature = "device")]
            DeviceInner::Device(raw) => Some(raw.info()),
        }
    }
    pub fn wait(&self) -> Result<(), DeviceLost> {
        match self.inner() {
            DeviceInner::Host => Ok(()),
            #[cfg(feature = "device")]
            DeviceInner::Device(raw) => raw.wait(),
        }
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "device")]
impl From<RawDevice> for Device {
    fn from(device: RawDevice) -> Self {
        Self {
            inner: DeviceInner::Device(device),
        }
    }
}

#[derive(Clone, Eq, PartialEq, derive_more::Unwrap)]
pub(crate) enum DeviceInner {
    Host,
    #[cfg(feature = "device")]
    Device(RawDevice),
}

impl DeviceInner {
    pub(crate) fn is_host(&self) -> bool {
        if let Self::Host = self {
            true
        } else {
            false
        }
    }
    pub(crate) fn is_device(&self) -> bool {
        !self.is_host()
    }
}

impl Debug for DeviceInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => f.debug_struct("Host").finish(),
            #[cfg(feature = "device")]
            Self::Device(raw_device) => raw_device.fmt(f),
        }
    }
}

#[cfg(feature = "device")]
#[derive(Clone)]
pub(crate) struct RawDevice {
    engine: Arc<Engine>,
}

#[cfg(feature = "device")]
impl RawDevice {
    fn new(options: DeviceOptions) -> Result<Self> {
        let engine = Engine::new(options)?;
        Ok(Self { engine })
    }
    fn info(&self) -> &Arc<DeviceInfo> {
        self.engine.info()
    }
    fn wait(&self) -> Result<(), DeviceLost> {
        self.engine.wait()
    }
}

#[cfg(feature = "device")]
impl PartialEq for RawDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.engine, &other.engine)
    }
}

#[cfg(feature = "device")]
impl Eq for RawDevice {}

#[cfg(feature = "device")]
impl Debug for RawDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let index = self.info().index;
        let handle = self.engine.handle() as *const ();
        f.debug_tuple("Device")
            .field(&index)
            .field(&handle)
            .finish()
    }
}

#[cfg(feature = "device")]
#[repr(transparent)]
#[derive(Clone)]
pub(crate) struct DeviceBuffer {
    inner: Arc<<Engine as DeviceEngine>::DeviceBuffer>,
}

#[cfg(feature = "device")]
impl DeviceBuffer {
    pub(crate) unsafe fn uninit(device: RawDevice, len: usize) -> Result<Self> {
        let inner = unsafe { <Engine as DeviceEngine>::DeviceBuffer::uninit(device.engine, len)? };
        Ok(Self { inner })
    }
    pub(crate) fn upload(device: RawDevice, data: &[u8]) -> Result<Self> {
        let inner = <Engine as DeviceEngine>::DeviceBuffer::upload(device.engine, data)?;
        Ok(Self { inner })
    }
    pub(crate) fn download(&self, data: &mut [u8]) -> Result<()> {
        self.inner.download(data)
    }
    pub(crate) fn transfer(&self, device: RawDevice) -> Result<Self> {
        let inner = self.inner.transfer(device.engine)?;
        Ok(Self { inner })
    }
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
    pub(crate) fn device(&self) -> RawDevice {
        RawDevice {
            engine: self.inner.engine().clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    shader_int8: bool,
    shader_int16: bool,
    shader_int64: bool,
    shader_float16: bool,
    shader_float64: bool,
}

impl Features {
    pub const fn empty() -> Self {
        Self {
            shader_int8: false,
            shader_int16: false,
            shader_int64: false,
            shader_float16: false,
            shader_float64: false,
        }
    }
    pub fn shader_int8(&self) -> bool {
        self.shader_int8
    }
    pub const fn with_shader_int8(mut self, shader_int8: bool) -> Self {
        self.shader_int8 = shader_int8;
        self
    }
    pub fn shader_int16(&self) -> bool {
        self.shader_int16
    }
    pub const fn with_shader_int16(mut self, shader_int16: bool) -> Self {
        self.shader_int16 = shader_int16;
        self
    }
    pub fn shader_int64(&self) -> bool {
        self.shader_int64
    }
    pub const fn with_shader_int64(mut self, shader_int64: bool) -> Self {
        self.shader_int64 = shader_int64;
        self
    }
    pub fn shader_float16(&self) -> bool {
        self.shader_float16
    }
    pub const fn with_shader_float16(mut self, shader_float16: bool) -> Self {
        self.shader_float16 = shader_float16;
        self
    }
    pub fn shader_float64(&self) -> bool {
        self.shader_float64
    }
    pub const fn with_shader_float64(mut self, shader_float64: bool) -> Self {
        self.shader_float64 = shader_float64;
        self
    }
    pub fn contains(&self, other: &Features) -> bool {
        (self.shader_int8 || !other.shader_int8)
            && (self.shader_int16 || !other.shader_int16)
            && (self.shader_int64 || !other.shader_int64)
            && (self.shader_float16 || !other.shader_float16)
            && (self.shader_float64 || !other.shader_float64)
    }
    pub fn union(mut self, other: &Features) -> Self {
        self.shader_int8 |= other.shader_int8;
        self.shader_int16 |= other.shader_int16;
        self.shader_int64 |= other.shader_int64;
        self.shader_float16 |= other.shader_float16;
        self.shader_float64 |= other.shader_float64;
        self
    }
}

#[derive(Debug)]
pub struct DeviceInfo {
    index: usize,
    name: String,
    compute_queues: usize,
    transfer_queues: usize,
    features: Features,
}

#[derive(Clone, Copy, Debug)]
struct TransferMetrics {
    bytes: usize,
    time: Duration,
}

#[derive(Clone, Copy, Debug)]
struct KernelMetrics {
    dispatches: usize,
    time: Duration,
}

#[derive(Clone, Debug)]
pub struct PerformanceMetrics {
    upload: TransferMetrics,
    download: TransferMetrics,
    kernels: HashMap<String, KernelMetrics>,
}

#[cfg(feature = "device")]
#[derive(Clone)]
pub(crate) struct RawKernel {
    inner: Arc<<Engine as DeviceEngine>::Kernel>,
}

#[cfg(feature = "device")]
impl RawKernel {
    pub(crate) fn new(
        device: RawDevice,
        spirv: Arc<[u32]>,
        spec_consts: &[ScalarElem],
    ) -> Result<Self> {
        Ok(Self {
            inner: <Engine as DeviceEngine>::Kernel::new(device.engine, spirv, spec_consts)?,
        })
    }
    pub(crate) unsafe fn dispatch(
        &self,
        groups: [u32; 3],
        buffers: &[DeviceBuffer],
        push_consts: &[ScalarElem],
    ) -> Result<()> {
        let buffers = unsafe { std::mem::transmute(buffers) };
        unsafe { self.inner.dispatch(groups, buffers, push_consts) }
    }
    pub(crate) fn device(&self) -> RawDevice {
        RawDevice {
            engine: self.inner.engine().clone(),
        }
    }
    pub(crate) fn threads(&self) -> [u32; 3] {
        self.inner.threads()
    }
    pub(crate) fn features(&self) -> Features {
        self.inner.features()
    }
    pub(crate) fn buffer_descs(&self) -> &[BufferDesc] {
        self.inner.buffer_descs()
    }
}

#[derive(Clone)]
pub(crate) struct BufferDesc {
    name: String,
    mutable: bool,
}

impl BufferDesc {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
    pub(crate) fn mutable(&self) -> bool {
        self.mutable
    }
}

#[cfg(feature = "device")]
struct KernelDesc {
    name: String,
    spirv: Vec<u32>,
    spirv_version: (u32, u32),
    threads: [u32; 3],
    features: Features,
    buffer_descs: Vec<BufferDesc>,
    push_consts_size: u32,
}

#[cfg(feature = "device")]
impl KernelDesc {
    fn new(spirv: &[u32], spec_consts: &[ScalarElem]) -> Result<Self> {
        use rspirv::{
            binary::Assemble,
            dr::{Instruction, Operand},
            spirv::{Decoration, ExecutionMode, Op, StorageClass},
        };
        /*let spirv = {
            use spirv_tools::opt::{Optimizer, Passes};
            let mut optimizer = spirv_tools::opt::compiled::CompiledOptimizer::default();
            // optimizer.register_pass(Passes::UpgradeMemoryModel);
            optimizer.register_performance_passes();
            optimizer
                .optimize(&spirv, &mut |_| (), None)
                .unwrap()
                .as_words()
                .to_vec()
        };*/
        let mut module = rspirv::dr::load_words(&spirv).map_err(|e| format_err!("{e}"))?;
        let mut name = {
            let entry_point = &mut module.entry_points.first_mut().unwrap().operands[2];
            let name = entry_point.unwrap_literal_string().to_string();
            *entry_point = Operand::LiteralString("main".to_string());
            name
        };
        let mut features = Features::empty();
        for inst in module.types_global_values.iter() {
            let class = inst.class;
            let op = class.opcode;
            match (op, inst.operands.first()) {
                (Op::TypeInt, Some(Operand::LiteralInt32(8))) => {
                    features.shader_int8 = true;
                }
                (Op::TypeInt, Some(Operand::LiteralInt32(16))) => {
                    features.shader_int16 = true;
                }
                (Op::TypeInt, Some(Operand::LiteralInt32(64))) => {
                    features.shader_int64 = true;
                }
                (Op::TypeFloat, Some(Operand::LiteralInt32(16))) => {
                    features.shader_float16 = true;
                }
                (Op::TypeFloat, Some(Operand::LiteralInt32(64))) => {
                    features.shader_float64 = true;
                }
                _ => (),
            }
        }
        let mut threads = [1u32; 3];
        match module.execution_modes.first().unwrap().operands.as_slice() {
            [Operand::IdRef(_), Operand::ExecutionMode(ExecutionMode::LocalSize), Operand::LiteralInt32(x), Operand::LiteralInt32(y), Operand::LiteralInt32(z)] =>
            {
                threads = [*x, *y, *z];
            }
            x => unreachable!("{x:#?}"),
        }
        let bindings: HashMap<_, _> = module.annotations.iter().filter(|x| x.class.opcode == Op::Decorate)
                .filter_map(|x| {
                    match x.operands.as_slice() {
                        [Operand::IdRef(id), Operand::Decoration(Decoration::Binding), Operand::LiteralInt32(binding)] => Some((*id, *binding)),
                        _ => None,
                    }
                }).collect();
        let num_buffers = bindings
            .values()
            .copied()
            .max()
            .map(|x| x as usize + 1)
            .unwrap_or_default();
        let mut buffer_descs = vec![
            BufferDesc {
                name: String::new(),
                mutable: true,
            };
            num_buffers
        ];
        for inst in module.debug_names.iter() {
            let op = inst.class.opcode;
            let operands = inst.operands.as_slice();
            if op == Op::Name {
                match operands {
                    [Operand::IdRef(id), Operand::LiteralString(name)] => {
                        if let Some(binding) = bindings.get(id) {
                            buffer_descs[*binding as usize].name = name.to_string();
                        }
                    }
                    _ => (),
                }
            }
        }
        for inst in module.annotations.iter() {
            let op = inst.class.opcode;
            let operands = inst.operands.as_slice();
            if op == Op::Decorate {
                match operands {
                    [Operand::IdRef(id), Operand::Decoration(Decoration::NonWritable)] => {
                        buffer_descs[bindings[id] as usize].mutable = false;
                    }
                    _ => (),
                }
            }
        }
        let push_consts_size = {
            let push_consts_ptr = module
                .types_global_values
                .iter()
                .filter(|x| x.class.opcode == Op::Variable)
                .find_map(|inst| {
                    if let [Operand::StorageClass(StorageClass::PushConstant)] =
                        inst.operands.as_slice()
                    {
                        inst.result_type
                    } else {
                        None
                    }
                });
            if let Some(push_consts_ptr) = push_consts_ptr {
                let push_consts_struct = module
                        .types_global_values
                        .iter()
                        .filter(|x| x.class.opcode == Op::TypePointer)
                        .filter(|x| x.result_id == Some(push_consts_ptr))
                        .find_map(|inst| {
                            if let [Operand::StorageClass(StorageClass::PushConstant), Operand::IdRef(push_consts_struct)] = inst.operands.as_slice() {
                                Some(*push_consts_struct)
                            } else {
                                None
                            }
                        })
                        .unwrap();
                let push_const_field = module
                    .types_global_values
                    .iter()
                    .filter(|x| x.class.opcode == Op::TypeStruct)
                    .find(|x| x.result_id == Some(push_consts_struct))
                    .unwrap()
                    .operands
                    .last()
                    .unwrap()
                    .unwrap_id_ref();
                let mut push_const_size = module
                    .types_global_values
                    .iter()
                    .find(|inst| inst.result_id == Some(push_const_field))
                    .map(|inst| match (inst.class.opcode, inst.operands.as_slice()) {
                        (
                            Op::TypeInt,
                            [Operand::LiteralInt32(width), Operand::LiteralInt32(_sign)],
                        ) => *width / 8,
                        (Op::TypeFloat, [Operand::LiteralInt32(width)]) => *width / 8,
                        _ => unreachable!("{inst:?}"),
                    })
                    .unwrap();
                let mut push_const_offset = 0;
                for inst in module.annotations.iter() {
                    if inst.class.opcode == Op::MemberDecorate {
                        if let [Operand::IdRef(id), Operand::LiteralInt32(member), Operand::Decoration(Decoration::Offset), Operand::LiteralInt32(offset)] =
                            inst.operands.as_slice()
                        {
                            if *id == push_consts_struct {
                                push_const_offset = push_const_offset.max(*offset);
                            }
                        }
                    }
                }
                push_const_offset + push_const_size
            } else {
                0
            }
        };
        if !spec_consts.is_empty() {
            name.push('<');
            todo!();
            name.push('>');
        }
        module.debug_names.clear();
        let spirv_version = module.header.as_ref().unwrap().version();
        let spirv_version = (spirv_version.0 as _, spirv_version.1 as _);
        //println!("{}", module.disassemble());
        let spirv = module.assemble();
        Ok(Self {
            name,
            spirv,
            spirv_version,
            threads,
            features,
            buffer_descs,
            push_consts_size,
        })
    }
}
