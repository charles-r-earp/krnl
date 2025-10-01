use crate::{
    scalar::ScalarType,
    spirv::{get_constant_u32, get_element_size, pointee_type, struct_element_type, variable_name},
};
use fxhash::{FxBuildHasher, FxHasher};
use indexmap::{
    map::{MutableEntryKey, MutableKeys},
    IndexMap, IndexSet,
};
use krnl_core::__private::__KrnlInst as KrnlInst;
use num_traits::FromPrimitive;
use smallvec::SmallVec;
use spirt::{
    print::Plan,
    spv::{
        encode_literal_string, extract_literal_string,
        spec::{ExtInstSetDesc, ExtInstSetInstructionDesc, Spec},
        Imm, Inst,
    },
    transform::{InnerInPlaceTransform, Transformer},
    visit::{InnerVisit, Visit, Visitor},
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstForm, DataInstFormDef, DataInstKind, DeclDef, ExportKey, Exportee, Func,
    GlobalVar, GlobalVarDecl, GlobalVarDefBody, InternedStr, Module, ModuleDialect, Type, TypeDef,
    TypeKind, TypeOrConst, Value,
};
use spirv_headers::{Capability, Decoration, ExecutionModel, MemoryModel, StorageClass};
use spirv_tools::{binary::Binary, opt::Optimizer, val::Validator, TargetEnv};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

#[derive(Debug)]
pub struct KernelDesc {
    pub buffers: Vec<BufferDesc>,
    pub push_constants: Vec<PushConstantDesc>,
}

impl KernelDesc {
    pub(crate) fn reflect(module: &Module, func: Func) -> Self {
        let cx = module.cx_ref();
        let globals = UsedGlobals::parse_module(module, Some(func));

        let bindings: BTreeMap<u32, GlobalVar> = globals
            .vars
            .iter()
            .filter_map(|(gv, storage_class)| {
                let (gv, storage_class) = (*gv, *storage_class);
                if storage_class != StorageClass::StorageBuffer {
                    return None;
                }
                let spec = Spec::get();
                let mut binding = None;
                let var_decl = &module.global_vars[gv];
                for attr in cx[var_decl.attrs].attrs.iter() {
                    if let Attr::SpvAnnotation(inst) = attr {
                        let opcode = inst.opcode;
                        if opcode == spec.well_known.OpDecorate {
                            if let [Imm::Short(_, decoration), Imm::Short(_, value)] =
                                inst.imms.as_slice()
                            {
                                let decoration = Decoration::from_u32(*decoration).unwrap();
                                let value = *value;
                                match decoration {
                                    Decoration::DescriptorSet => {
                                        let descriptor_set = value;
                                        assert!(descriptor_set == 0);
                                    }
                                    Decoration::Binding => {
                                        binding.replace(value);
                                    }
                                    _ => (),
                                }
                            }
                        }
                    }
                }
                Some((binding.unwrap(), gv))
            })
            .collect();

        let mut kernel_desc = KernelDesc {
            buffers: Vec::new(),
            push_constants: Vec::new(),
        };

        for gv in bindings.values().copied() {
            let var_decl = &module.global_vars[gv];
            let ptr_ty = var_decl.type_of_ptr_to;
            let struct_ty = pointee_type(&cx, ptr_ty).unwrap();
            let ptr_ty = struct_element_type(cx, struct_ty).unwrap();
            let elem_ty = pointee_type(&cx, ptr_ty).unwrap();
            let name = variable_name(module, gv).unwrap();
            let element_type = ElementType::from_type(module.cx_ref(), elem_ty).unwrap();
            let scalar = element_type.scalar_type();
            let scalar_type = get_scalar_type(cx, scalar).unwrap();
            let array = if let ElementType::Array(_, len) = element_type {
                Some(len)
            } else {
                None
            };
            let mut access = AccessMode::READ_WRITE;
            let mut binding = None;
            let spec = Spec::get();
            for attr in cx[var_decl.attrs].attrs.iter() {
                if let Attr::SpvAnnotation(inst) = attr {
                    let opcode = inst.opcode;
                    if opcode == spec.well_known.OpDecorate {
                        if let [Imm::Short(_, decoration)] = inst.imms.as_slice() {
                            let decoration = Decoration::from_u32(*decoration).unwrap();
                            if let Decoration::NonWritable = decoration {
                                access = AccessMode::READ;
                            }
                        } else if let [Imm::Short(_, decoration), Imm::Short(_, value)] =
                            inst.imms.as_slice()
                        {
                            let decoration = Decoration::from_u32(*decoration).unwrap();
                            let value = *value;
                            match decoration {
                                Decoration::DescriptorSet => {
                                    let descriptor_set = value;
                                    assert!(descriptor_set == 0);
                                }
                                Decoration::Binding => {
                                    binding.replace(value);
                                }
                                _ => (),
                            }
                        }
                    }
                }
            }
            let binding = binding.unwrap();
            kernel_desc.buffers.push(BufferDesc {
                name,
                scalar_type,
                array,
                access,
            });
        }

        let push_var = globals.vars.iter().find_map(|(gv, storage_class)| {
            if *storage_class == StorageClass::PushConstant {
                Some(*gv)
            } else {
                None
            }
        });
        if let Some(gv) = push_var {
            let var_decl = &module.global_vars[gv];
            let name = variable_name(module, gv);
            let ptr_ty = var_decl.type_of_ptr_to;
            let struct_ty = pointee_type(&cx, ptr_ty).unwrap();
            let field_types = get_struct_field_types(cx, struct_ty).unwrap();
            let mut push_constants = Vec::new();
            for elem_ty in field_types {
                let element_type = ElementType::from_type(cx, elem_ty).unwrap();
                let scalar = element_type.scalar_type();
                let scalar_type = get_scalar_type(cx, scalar).unwrap();
                let array = if let ElementType::Array(_, len) = element_type {
                    Some(len)
                } else {
                    None
                };
                push_constants.push(PushConstantDesc {
                    name: String::new(),
                    offset: 0,
                    scalar_type,
                    array,
                });
            }
            let spec = Spec::get();
            let struct_def = &cx[struct_ty];
            let attrs = &cx[struct_def.attrs].attrs;
            for attr in attrs.iter() {
                if let Attr::SpvAnnotation(inst) = attr {
                    let opcode = inst.opcode;
                    if opcode == spec.well_known.OpMemberName {
                        let member = if let Imm::Short(_, member) = inst.imms[0] {
                            member
                        } else {
                            unreachable!()
                        };
                        let name = extract_literal_string(&inst.imms[1..]).unwrap();
                        push_constants[member as usize].name = name;
                    } else if opcode == spec.well_known.OpMemberDecorate {
                        if let [Imm::Short(_, member), Imm::Short(_, decoration), Imm::Short(_, offset)] =
                            inst.imms.as_slice()
                        {
                            let decoration = Decoration::from_u32(*decoration).unwrap();
                            if decoration == Decoration::Offset {
                                push_constants[*member as usize].offset = *offset;
                            }
                        }
                    }
                }
            }
            kernel_desc.push_constants = push_constants;
        }
        kernel_desc

        /*
        struct KernelVisitor<'a> {
            module: &'a Module,
            func: Func,
            buffers: IndexMap<GlobalVar, (u32, BufferDesc), FxBuildHasher>,
            push_constants_var: Option<GlobalVar>,
            push_constants: Vec<PushConstantDesc>,
        }

        impl KernelVisitor<'_> {
            fn visit_buffer(&mut self, gv: GlobalVar) {
                let cx = self.module.cx_ref();
            }
            fn visit_push_constants(&mut self, gv: GlobalVar) {
                self.push_constants_var.replace(gv);

                }
            }
        }

        impl Visitor<'_> for KernelVisitor<'_> {
            fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
            fn visit_type_use(&mut self, _ty: Type) {}
            fn visit_const_use(&mut self, ct: Const) {
                let const_def = &self.module.cx_ref()[ct];
                if let ConstKind::PtrToGlobalVar(gv) = const_def.kind {
                    self.visit_global_var_use(gv);
                }
            }
            fn visit_data_inst_form_use(&mut self, data_inst_form: DataInstForm) {}
            fn visit_global_var_use(&mut self, gv: GlobalVar) {
                let var_decl = &self.module.global_vars[gv];
                if let AddrSpace::SpvStorageClass(storage_class) = var_decl.addr_space {
                    let storage_class = StorageClass::from_u32(storage_class).unwrap();
                    match storage_class {
                        StorageClass::StorageBuffer => {
                            self.visit_buffer(gv);
                        }
                        StorageClass::PushConstant => {
                            self.visit_push_constants(gv);
                        }
                        _ => (),
                    }
                }
            }
            fn visit_func_use(&mut self, func: Func) {
                self.func = func;
                self.module.funcs[func].inner_visit_with(self);
            }

            fn visit_data_inst_def(&mut self, data_inst_def: &DataInstDef) {
                let cx = self.module.cx_ref();
                let form_def = &cx[data_inst_def.form];
                if let DataInstKind::SpvExtInst { ext_set, inst } = &form_def.kind {
                    let ext_set_str = &cx[*ext_set];
                    if ext_set_str == KrnlInst::SET_NAME && *inst == KrnlInst::DataType as u32 {
                        let (var, type_name) = if let [Value::DataInstOutput(var), Value::Const(type_name)] =
                            data_inst_def.inputs.as_slice()
                        {
                            (*var, *type_name)
                        } else {
                            unreachable!()
                        };
                        let scalar_type = {
                            let const_def = &cx[type_name];
                            if let ConstKind::SpvStringLiteralForExtInst(type_name) = const_def.kind
                            {
                                match &cx[type_name] {
                                    "f16" => ScalarType::F16,
                                    "bf16" => ScalarType::BF16,
                                    _ => unreachable!(),
                                }
                            } else {
                                unreachable!();
                            }
                        };
                        let func_decl = &self.module.funcs[self.func];
                        let body = if let DeclDef::Present(body) = &func_decl.def {
                            body
                        } else {
                            unreachable!()
                        };
                        let data_inst_def = &body.data_insts[var];
                        let form_def = &cx[data_inst_def.form];
                        if let [Value::Const(var), Value::Const(index1), Value::Const(index2)] =
                            data_inst_def.inputs.as_slice()
                        {
                            let const_def = &cx[*var];
                            let gv = if let ConstKind::PtrToGlobalVar(gv) = const_def.kind {
                                gv
                            } else {
                                unreachable!()
                            };
                            let var_decl = &self.module.global_vars[gv];
                            let storage_class = if let AddrSpace::SpvStorageClass(storage_class) =
                                var_decl.addr_space
                            {
                                StorageClass::from_u32(storage_class).unwrap()
                            } else {
                                unreachable!()
                            };
                            match storage_class {
                                StorageClass::StorageBuffer => {
                                    if cfg!(debug_assertions) {
                                        let index1 = get_constant_u32(cx, *index1).unwrap();
                                        let index2 = get_constant_u32(cx, *index2).unwrap();
                                        debug_assert!(index1 == 0);
                                        debug_assert!(index2 == 0);
                                    }
                                    self.buffers[&gv].1.scalar_type = scalar_type;
                                }
                                StorageClass::PushConstant => {
                                    if cfg!(debug_assertions) {
                                        let index1 = get_constant_u32(cx, *index1).unwrap();
                                        debug_assert!(index1 == 0);
                                    }
                                    let member = get_constant_u32(cx, *index2).unwrap();
                                    self.push_constants[member as usize].scalar_type = scalar_type;
                                }
                                _ => (),
                            }
                        } else {
                            unreachable!();
                        }
                    }
                }
                data_inst_def.inner_visit_with(self);
            }
        }

        let mut visitor = KernelVisitor {
            module,
            func,
            buffers: IndexMap::default(),
            push_constants_var: None,
            push_constants: Vec::new(),
        };
        module.funcs[func].inner_visit_with(&mut visitor);

        let mut buffers: Vec<BufferDesc> = (0..visitor.buffers.len())
            .map(|_| BufferDesc::null())
            .collect();
        for (_gv, (binding, desc)) in visitor.buffers {
            buffers[binding as usize] = desc;
        }
        Self {
            buffers,
            push_constants: visitor.push_constants,
        }
        */
    }
    pub fn push_constant_bytes(&self) -> usize {
        self.push_constants
            .last()
            .map(|x| x.offset as usize + x.size())
            .unwrap_or_default()
    }
    pub fn push_constant_offset(&self, name: &str) -> Option<u32> {
        self.push_constants
            .iter()
            .find(|x| x.name == name)
            .map(|x| x.offset)
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct AccessMode(u8);

impl AccessMode {
    const UNUSED: Self = Self(0);
    const READ: Self = Self(1);
    const WRITE: Self = Self(2);
    const READ_WRITE: Self = Self(3);
    fn mutable(&self) -> bool {
        self.0 & Self::WRITE.0 != 0
    }
}

impl std::fmt::Debug for AccessMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match *self {
            Self::UNUSED => "UNUSED",
            Self::READ => "READ",
            Self::WRITE => "WRITE",
            Self::READ_WRITE => "READ_WRITE",
            _ => unreachable!(),
        };
        write!(f, "{s}")
    }
}

#[derive(Clone, Debug)]
pub struct BufferDesc {
    name: String,
    pub(crate) scalar_type: ScalarType,
    array: Option<u32>,
    access: AccessMode,
}

impl BufferDesc {
    pub fn name(&self) -> &str {
        &self.name
    }
    fn null() -> Self {
        Self {
            name: String::new(),
            scalar_type: ScalarType::U8,
            array: None,
            access: AccessMode::UNUSED,
        }
    }
    pub fn mutable(&self) -> bool {
        self.access.mutable()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }
    pub(crate) fn array(&self) -> Option<u32> {
        self.array
    }
    fn elem_size(&self) -> usize {
        self.array.unwrap_or(1) as usize * self.scalar_type.size()
    }
}

#[derive(Clone, Debug)]
pub struct PushConstantDesc {
    name: String,
    offset: u32,
    pub(crate) scalar_type: ScalarType,
    array: Option<u32>,
}

impl PushConstantDesc {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn offset(&self) -> u32 {
        self.offset
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }
    pub(crate) fn array(&self) -> Option<u32> {
        self.array
    }
    pub fn size(&self) -> usize {
        self.array.unwrap_or(1) as usize * self.scalar_type.size()
    }
}

trait ScalarTypeExt {
    fn size(&self) -> usize;
}

impl ScalarTypeExt for ScalarType {
    fn size(&self) -> usize {
        use ScalarType::*;
        match *self {
            U8 | I8 => 1,
            U16 | I16 | F16 | BF16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }
}

pub(crate) fn get_scalar_type(cx: &Context, ty: Type) -> Option<ScalarType> {
    let type_def = &cx[ty];
    if let TypeKind::SpvInst {
        spv_inst,
        type_and_const_inputs,
    } = &type_def.kind
    {
        let spec = Spec::get();
        let opcode = spv_inst.opcode;
        if opcode == spec.well_known.OpTypeInt {
            let (bits, signed) = match (type_and_const_inputs.as_slice(), spv_inst.imms.as_slice())
            {
                ([], [Imm::Short(_, bits), Imm::Short(_, signed)]) => (*bits, *signed),
                ([TypeOrConst::Const(bits), TypeOrConst::Const(signed)], _) => {
                    let bits = get_constant_u32(cx, *bits).unwrap();
                    let signed = get_constant_u32(cx, *signed).unwrap();
                    (bits, signed)
                }
                _ => unreachable!(),
            };
            let signed = signed != 0;
            let scalar_type = match (bits, signed) {
                (8, false) => ScalarType::U8,
                (8, true) => ScalarType::I8,
                (16, false) => ScalarType::U16,
                (16, true) => ScalarType::I16,
                (32, false) => ScalarType::U32,
                (32, true) => ScalarType::I32,
                (64, false) => ScalarType::U64,
                (64, true) => ScalarType::I64,
                _ => unreachable!(),
            };
            return Some(scalar_type);
        } else if opcode == spec.well_known.OpTypeFloat {
            let (bits, encoding) =
                match (type_and_const_inputs.as_slice(), spv_inst.imms.as_slice()) {
                    ([], [Imm::Short(_, bits)]) => (*bits, None),
                    ([], [Imm::Short(_, bits), Imm::Short(_, encoding)]) => {
                        (*bits, Some(*encoding))
                    }
                    ([TypeOrConst::Const(bits)], _) => {
                        let bits = get_constant_u32(cx, *bits).unwrap();
                        (bits, None)
                    }
                    ([TypeOrConst::Const(bits), TypeOrConst::Const(encoding)], _) => {
                        let bits = get_constant_u32(cx, *bits).unwrap();
                        let encoding = get_constant_u32(cx, *encoding).unwrap();
                        (bits, Some(encoding))
                    }
                    _ => unreachable!(),
                };
            let scalar_type = match (bits, encoding) {
                (16, None) => ScalarType::F16,
                (32, None) => ScalarType::F32,
                (64, None) => ScalarType::F64,
                _ => unreachable!(),
            };
            return Some(scalar_type);
        } else if opcode == spec.well_known.OpTypeStruct {
            let ty = struct_element_type(cx, ty).unwrap();
            return get_scalar_type(cx, ty);
        }
    }
    None
}

pub(crate) enum ElementType {
    Scalar(Type),
    Array(Type, u32),
}

impl ElementType {
    pub(crate) fn from_type(cx: &Context, ty: Type) -> Option<Self> {
        let type_def = &cx[ty];
        if let TypeKind::SpvInst {
            spv_inst,
            type_and_const_inputs,
        } = &type_def.kind
        {
            let spec = Spec::get();
            let opcode = spv_inst.opcode;
            if opcode == spec.well_known.OpTypeInt || opcode == spec.well_known.OpTypeFloat {
                return Some(Self::Scalar(ty));
            } else if opcode == spec.well_known.OpTypeArray {
                if let &[TypeOrConst::Type(elem_ty), TypeOrConst::Const(len)] =
                    type_and_const_inputs.as_slice()
                {
                    let len = get_constant_u32(cx, len).unwrap();
                    return Some(Self::Array(elem_ty, len));
                }
            } else if opcode == spec.well_known.OpTypeStruct {
                let ty = struct_element_type(cx, ty).unwrap();
                return Self::from_type(cx, ty);
            }
        }
        None
    }
    fn scalar_type(&self) -> Type {
        match self {
            Self::Scalar(x) => *x,
            Self::Array(x, _) => *x,
        }
    }
}

fn get_struct_field_types(cx: &Context, ty: Type) -> Option<impl Iterator<Item = Type> + '_> {
    let type_def = &cx[ty];
    if let TypeKind::SpvInst {
        spv_inst,
        type_and_const_inputs,
    } = &type_def.kind
    {
        let spec = Spec::get();
        let opcode = spv_inst.opcode;
        if opcode == spec.well_known.OpTypeStruct {
            return Some(type_and_const_inputs.iter().map(|x| {
                if let TypeOrConst::Type(x) = x {
                    *x
                } else {
                    unreachable!()
                }
            }));
        }
    }
    None
}

#[derive(Default, Clone, Debug)]
pub struct Features {
    capabilities: BTreeSet<Capability>,
    extensions: BTreeSet<String>,
}

impl Features {
    pub(crate) fn reflect(module: &Module) -> Self {
        struct FeaturesVisitor<'a> {
            module: &'a Module,
            features: Features,
        }

        impl FeaturesVisitor<'_> {
            fn visit_buffer(&mut self, elem_ty: Type) {
                self.visit_type_use(elem_ty);
                self.features
                    .capabilities
                    .insert(Capability::VariablePointersStorageBuffer);
                let element_type = ElementType::from_type(self.module.cx_ref(), elem_ty).unwrap();
                let scalar = element_type.scalar_type();
                let size = get_element_size(self.module.cx_ref(), scalar).unwrap();
                if size == 1 {
                    self.features
                        .capabilities
                        .insert(Capability::StorageBuffer8BitAccess);
                } else if size == 2 {
                    self.features
                        .capabilities
                        .insert(Capability::StorageBuffer16BitAccess);
                }
            }
            fn visit_push_constant_struct(&mut self, struct_ty: Type) {
                let field_types = get_struct_field_types(self.module.cx_ref(), struct_ty).unwrap();
                for elem_ty in field_types {
                    self.visit_type_use(elem_ty);
                    let element_type =
                        ElementType::from_type(self.module.cx_ref(), elem_ty).unwrap();
                    let scalar = element_type.scalar_type();
                    let size = get_element_size(self.module.cx_ref(), scalar).unwrap();
                    if size == 1 {
                        self.features
                            .capabilities
                            .insert(Capability::StoragePushConstant8);
                    } else if size == 2 {
                        self.features
                            .capabilities
                            .insert(Capability::StoragePushConstant16);
                    }
                }
            }
        }

        impl Visitor<'_> for FeaturesVisitor<'_> {
            fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
            fn visit_type_use(&mut self, ty: Type) {
                if let Some(scalar_type) = get_scalar_type(self.module.cx_ref(), ty) {
                    if scalar_type.size() == 1 {
                        self.features.capabilities.insert(Capability::Int8);
                    } else if scalar_type.size() == 2 {
                        self.features.capabilities.insert(Capability::Int16);
                        if scalar_type == ScalarType::F16 {
                            self.features.capabilities.insert(Capability::Float16);
                        }
                    } else if scalar_type.size() == 8 {
                        self.features.capabilities.insert(Capability::Int64);
                        if scalar_type == ScalarType::F64 {
                            self.features.capabilities.insert(Capability::Float64);
                        }
                    }
                }
            }
            fn visit_const_use(&mut self, _ct: Const) {}
            fn visit_data_inst_form_use(&mut self, data_inst_form: DataInstForm) {
                let form_def = &self.module.cx_ref()[data_inst_form];
                if let DataInstKind::SpvExtInst { ext_set, inst } = &form_def.kind {
                    let ext_set_str = &self.module.cx_ref()[*ext_set];
                    if ext_set_str.starts_with("NonSemantic.") {
                        let non_semantic_info = "SPV_KHR_non_semantic_info";
                        if !self.features.extensions.contains(non_semantic_info) {
                            self.features
                                .extensions
                                .insert(non_semantic_info.to_string());
                        }
                    }
                }
            }
            fn visit_global_var_use(&mut self, gv: GlobalVar) {
                let cx = self.module.cx_ref();
                let var_decl = &self.module.global_vars[gv];
                if let AddrSpace::SpvStorageClass(storage_class) = var_decl.addr_space {
                    let storage_class = StorageClass::from_u32(storage_class).unwrap();
                    match storage_class {
                        StorageClass::StorageBuffer => {
                            let ptr_ty = var_decl.type_of_ptr_to;
                            let struct_ty = pointee_type(&cx, ptr_ty).unwrap();
                            let ptr_ty = struct_element_type(cx, struct_ty).unwrap();
                            let elem_ty = pointee_type(&cx, ptr_ty).unwrap();
                            self.visit_buffer(elem_ty);
                        }
                        StorageClass::PushConstant => {
                            let ptr_ty = var_decl.type_of_ptr_to;
                            let struct_ty = pointee_type(&cx, ptr_ty).unwrap();
                            self.visit_push_constant_struct(struct_ty);
                        }
                        _ => (),
                    }
                }
            }
            fn visit_func_use(&mut self, func: Func) {
                self.module.funcs[func].inner_visit_with(self);
            }
            fn visit_spv_dialect(&mut self, dialect: &spirt::spv::Dialect) {
                let memory_model = MemoryModel::from_u32(dialect.memory_model).unwrap();
                self.features
                    .capabilities
                    .extend(memory_model.required_capabilities().into_iter().copied());
            }
        }

        let mut visitor = FeaturesVisitor {
            module,
            features: Features::default(),
        };
        module.visit_with(&mut visitor);
        visitor.features
    }
    pub(crate) fn write_to_module(&self, module: &mut Module) {
        let ModuleDialect::Spv(ref mut dialect) = module.dialect;
        dialect.capabilities = self
            .capabilities
            .iter()
            .copied()
            .chain([Capability::Shader])
            .map(|x| x as u32)
            .collect();
        dialect.extensions = self.extensions.clone();
    }
}

#[derive(Default)]
pub(crate) struct UsedGlobals {
    pub(crate) vars: IndexMap<GlobalVar, StorageClass, FxBuildHasher>,
    pub(crate) funcs: IndexSet<Func, FxBuildHasher>,
}

impl UsedGlobals {
    pub(crate) fn parse_module(module: &Module, func: Option<Func>) -> Self {
        struct Collector<'a> {
            module: &'a Module,
            output: UsedGlobals,
        }

        impl Visitor<'_> for Collector<'_> {
            fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
            fn visit_type_use(&mut self, ty: Type) {
                self.module.cx_ref()[ty].inner_visit_with(self);
            }
            fn visit_const_use(&mut self, ct: Const) {
                self.module.cx_ref()[ct].inner_visit_with(self);
            }
            fn visit_data_inst_form_use(&mut self, data_inst_form: DataInstForm) {
                self.module.cx_ref()[data_inst_form].inner_visit_with(self);
            }
            fn visit_global_var_use(&mut self, gv: GlobalVar) {
                if self.output.vars.contains_key(&gv) {
                    return;
                }
                let var_decl = &self.module.global_vars[gv];
                let storage_class =
                    if let AddrSpace::SpvStorageClass(storage_class) = var_decl.addr_space {
                        StorageClass::from_u32(storage_class).unwrap()
                    } else {
                        unreachable!()
                    };
                self.output.vars.insert(gv, storage_class);
            }
            fn visit_func_use(&mut self, func: Func) {
                self.output.funcs.insert(func);
                self.module.funcs[func].inner_visit_with(self);
            }
        }

        let mut output = Self::default();
        let mut collector = Collector {
            module,
            output: UsedGlobals::default(),
        };
        if let Some(func) = func {
            Exportee::Func(func).inner_visit_with(&mut collector);
        } else {
            module.inner_visit_with(&mut collector);
        }
        collector.output
    }
}
