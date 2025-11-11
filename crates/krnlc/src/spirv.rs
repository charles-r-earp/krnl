use fxhash::{FxBuildHasher, FxHashSet};
use indexmap::{
    IndexMap, IndexSet,
    map::{MutableEntryKey, MutableKeys},
};
use krnl_core::__private::__KrnlInst as KrnlInst;
use smallvec::{SmallVec, smallvec};
use spirt::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstForm, DataInstFormDef, DataInstKind, DeclDef, ExportKey, Exportee, Func,
    GlobalVar, GlobalVarDecl, GlobalVarDefBody, InternedStr, Module, Type, TypeDef, TypeKind,
    TypeOrConst, Value,
    spv::{
        Imm, Inst, encode_literal_string, extract_literal_string,
        spec::{ExtInstSetDesc, ExtInstSetInstructionDesc, Opcode, OperandMode, Spec},
    },
    transform::{InnerInPlaceTransform, Transformer},
    visit::{InnerVisit, Visitor},
};
use spirv_headers::{Capability, Decoration, ExecutionModel, Op, StorageClass};
use spirv_tools::{TargetEnv, binary::Binary, opt::Optimizer, val::Validator};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

pub(crate) fn assemble(module: &Module) -> std::io::Result<Vec<u32>> {
    Ok(module.lift_to_spv_module_emitter()?.words)
}

pub(crate) fn validate(binary: &[u32]) -> Result<(), spirv_tools::Error> {
    spirv_tools::val::create(Some(TargetEnv::Vulkan_1_3)).validate(binary, None)
}

pub(crate) fn krnl_inst_set() -> ExtInstSetDesc {
    let mut instructions = BTreeMap::new();
    for inst in KrnlInst::iter() {
        instructions.insert(
            inst as u32,
            ExtInstSetInstructionDesc {
                name: format!("{inst:?}").into(),
                operand_names: inst
                    .operand_names()
                    .into_iter()
                    .copied()
                    .map(Into::into)
                    .collect(),
                is_debuginfo: true,
            },
        );
    }
    ExtInstSetDesc {
        short_alias: Some(KrnlInst::SET_SHORT_NAME.into()),
        instructions,
    }
}

/*
pub(crate) struct Kernel {
    pub(crate) entry_point: ExportKey,
    pub(crate) func: Func,
    pub(crate) name: InternedStr,
    pub(crate) generics: String,
    pub(crate) inputs: IndexMap<u32, InternedStr>,
}

impl Kernel {
    pub(crate) fn new(module: &Module, entry_point: ExportKey) -> Self {
        if let ExportKey::SpvEntryPoint {
            imms,
            interface_global_vars: _,
        } = &entry_point
        {
            let func = if let Exportee::Func(func) = module.exports[&entry_point] {
                func
            } else {
                unreachable!()
            };
            if let Imm::Short(_, word) = imms[0] {
                assert_eq!(word, ExecutionModel::GLCompute as u32);
            } else {
                unreachable!()
            }
            let entry_name = extract_literal_string(&imms[1..]).unwrap();
            struct KernelDataCollector<'a> {
                module: &'a Module,
                krnl_set: InternedStr,
                data: Vec<u32>,
            }

            impl Visitor<'_> for KernelDataCollector<'_> {
                fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
                fn visit_type_use(&mut self, _ty: Type) {}
                fn visit_const_use(&mut self, _ct: Const) {}
                fn visit_data_inst_form_use(&mut self, data_inst_form: DataInstForm) {
                    let data_inst_form_def = &self.module.cx_ref()[data_inst_form];
                    data_inst_form_def.inner_visit_with(self);
                }
                fn visit_global_var_use(&mut self, _gv: GlobalVar) {}
                fn visit_func_use(&mut self, func: Func) {
                    self.module.funcs[func].inner_visit_with(self);
                }
                fn visit_data_inst_def(&mut self, data_inst_def: &DataInstDef) {
                    let data_inst_form_def = &self.module.cx_ref()[data_inst_def.form];
                    if let DataInstKind::SpvExtInst { ext_set, inst } = data_inst_form_def.kind {
                        if ext_set == self.krnl_set && inst == KrnlInst::KernelData as u32 {
                            if let &[Value::Const(ct)] = data_inst_def.inputs.as_slice() {
                                let data = get_constant_u32(self.module.cx_ref(), ct).unwrap();
                                self.data.push(data);
                            }
                        }
                    }
                    data_inst_def.inner_visit_with(self);
                }
            }

            let krnl_set = module.cx_ref().intern(KrnlInst::SET_NAME);

            let mut collector = KernelDataCollector {
                module,
                krnl_set,
                data: Vec::new(),
            };
            Exportee::Func(func).inner_visit_with(&mut collector);
            let data = collector.data;
            let data = std::str::from_utf8(bytemuck::cast_slice(&data)).unwrap();
            dbg!(data);
            let (generics, inputs) = data.split_once(';').unwrap();
            let generics = generics.to_owned();
            let name = module.cx().intern(entry_name);
            let inputs: IndexMap<_, _> = inputs
                .split(',')
                .enumerate()
                .map(|(i, input)| {
                    let i = i.try_into().unwrap();
                    let input = module.cx_ref().intern(input.trim());
                    (i, input)
                })
                .collect();

            Self {
                entry_point,
                func,
                name,
                generics,
                inputs,
            }
        } else {
            unreachable!()
        }
    }
}
*/

pub(crate) fn get_name_from_attrs<'a>(attrs: impl IntoIterator<Item = &'a Attr>) -> Option<String> {
    attrs.into_iter().find_map(|attr| {
        if let Attr::SpvAnnotation(inst) = attr {
            if inst.opcode == Spec::get().well_known.OpName {
                let name = extract_literal_string(&inst.imms).unwrap();
                return Some(name);
            }
        }
        None
    })
}

pub(crate) fn constant_name(cx: &Context, ct: Const) -> Option<String> {
    let const_def = &cx[ct];
    let attrs_def = &cx[const_def.attrs];
    get_name_from_attrs(attrs_def.attrs.iter())
}

pub(crate) fn variable_name(module: &Module, var: GlobalVar) -> Option<String> {
    let decl = &module.global_vars[var];
    let attrs_def = &module.cx_ref()[decl.attrs];
    get_name_from_attrs(attrs_def.attrs.iter())
    /*
    for attr in attrs_def.attrs.iter() {
        if let Attr::SpvAnnotation(inst) = attr {
            if inst.opcode == Spec::get().well_known.OpName {
                let name = extract_literal_string(&inst.imms).unwrap();
                return Some(name);
            }
        }
    }
    None
    */
}

/*
pub(crate) fn member_names(cx: &Context, ty: Type) -> impl Iterator<Item = String> + '_ {
    let ty_def = &cx[ty];
    let attrs_def = &cx[ty_def.attrs];
    attrs_def.attrs.iter().filter_map(|attr| {
        if let Attr::SpvAnnotation(inst) = attr {
            if inst.opcode == Spec::get().well_known.OpMemberName {
                let name = extract_literal_string(&inst.imms).unwrap();
                return Some(name);
            }
        }
        None
    })
}
*/

pub(crate) fn op_type_pointer(cx: &Context, ty: Type, storage_class: StorageClass) -> Type {
    let spec = Spec::get();
    let opcode = spec.well_known.OpTypePointer;
    let operand_kinds: Vec<_> = opcode.def().all_operands().map(|(_, x)| x).collect();
    let spv_inst = Inst {
        opcode,
        imms: [Imm::Short(operand_kinds[0], storage_class as u32)]
            .into_iter()
            .collect(),
    };
    let def = TypeDef {
        attrs: AttrSet::default(),
        kind: TypeKind::SpvInst {
            spv_inst,
            type_and_const_inputs: [TypeOrConst::Type(ty)].into_iter().collect(),
        },
    };
    cx.intern(def)
}

pub(crate) fn pointee_type(cx: &Context, pointer_ty: Type) -> Option<Type> {
    let type_def = &cx[pointer_ty];
    if let TypeKind::SpvInst {
        spv_inst: _,
        type_and_const_inputs,
    } = &type_def.kind
    {
        if let Some(TypeOrConst::Type(ty)) = type_and_const_inputs.first().copied() {
            return Some(ty);
        }
    }
    None
}

pub(crate) fn struct_element_type(cx: &Context, struct_ty: Type) -> Option<Type> {
    let type_def = &cx[struct_ty];
    if let TypeKind::SpvInst {
        spv_inst,
        type_and_const_inputs,
    } = &type_def.kind
    {
        assert!(spv_inst.opcode == Spec::get().well_known.OpTypeStruct);
        if let Some(TypeOrConst::Type(ty)) = type_and_const_inputs.first().copied() {
            return Some(ty);
        }
    }
    None
}

pub(crate) fn runtime_array_element_type(cx: &Context, runtime_array_ty: Type) -> Option<Type> {
    let type_def = &cx[runtime_array_ty];
    if let TypeKind::SpvInst {
        spv_inst,
        type_and_const_inputs,
    } = &type_def.kind
    {
        assert!(
            spv_inst.opcode == Spec::get().well_known.OpTypeRuntimeArray,
            "{}",
            spv_inst.opcode.name()
        );
        if let Some(TypeOrConst::Type(ty)) = type_and_const_inputs.first().copied() {
            return Some(ty);
        }
    }
    None
}

pub(crate) fn get_constant_u32(cx: &Context, ct: Const) -> Option<u32> {
    let const_def = &cx[ct];
    if let ConstKind::SpvInst {
        spv_inst_and_const_inputs,
    } = &const_def.kind
    {
        if let Imm::Short(_, x) = spv_inst_and_const_inputs.0.imms[0] {
            return Some(x);
        }
    }
    None
}

pub(crate) fn get_element_size(cx: &Context, ty: Type) -> Option<u32> {
    let type_def = &cx[ty];
    if let TypeKind::SpvInst {
        spv_inst,
        type_and_const_inputs,
    } = &type_def.kind
    {
        let spec = Spec::get();
        let opcode = spv_inst.opcode;
        if opcode == spec.well_known.OpTypeInt || opcode == spec.well_known.OpTypeFloat {
            if let Some(TypeOrConst::Const(bits)) = type_and_const_inputs.first().copied() {
                let bits = get_constant_u32(cx, bits).unwrap();
                return Some(bits / 8);
            } else if let Imm::Short(_, bits) = spv_inst.imms[0] {
                return Some(bits / 8);
            }
            unreachable!();
        } else if opcode == spec.well_known.OpTypeArray {
            if let &[TypeOrConst::Type(elem_ty), TypeOrConst::Const(len)] =
                type_and_const_inputs.as_slice()
            {
                let size = get_element_size(cx, elem_ty).unwrap();
                let len = get_constant_u32(cx, len).unwrap();
                return Some(len * size);
            }
            unreachable!();
        }
    }
    None
}

pub(crate) fn op_nop(cx: &Context) -> DataInstDef {
    let inst = Inst {
        opcode: Spec::get().well_known.OpNop,
        imms: SmallVec::default(),
    };
    let data_inst_form_def = DataInstFormDef {
        kind: DataInstKind::SpvInst(inst),
        output_type: None,
    };
    let data_inst_form = cx.intern(data_inst_form_def);
    let data_inst_def = DataInstDef {
        attrs: AttrSet::default(),
        form: data_inst_form,
        inputs: SmallVec::default(),
    };
    data_inst_def
}

pub(crate) fn op_variable(
    module: &mut Module,
    pointer_ty: Type,
    storage_class: StorageClass,
    attrs: AttrSet,
) -> GlobalVar {
    let cx = module.cx();
    module.global_vars.define(
        &cx,
        GlobalVarDecl {
            attrs,
            type_of_ptr_to: pointer_ty,
            shape: None,
            addr_space: AddrSpace::SpvStorageClass(storage_class as u32),
            def: DeclDef::Present(GlobalVarDefBody { initializer: None }),
        },
    )
}

pub(crate) fn op_access_chain(
    cx: &Context,
    pointer_ty: Type,
    base: Value,
    indices: impl IntoIterator<Item = Value>,
) -> DataInstDef {
    let spv_inst = Inst {
        opcode: Spec::get().well_known.OpAccessChain,
        imms: SmallVec::default(),
    };
    let form_def = DataInstFormDef {
        kind: DataInstKind::SpvInst(spv_inst),
        output_type: Some(pointer_ty),
    };
    let form = cx.intern(form_def);
    DataInstDef {
        attrs: AttrSet::default(),
        form,
        inputs: std::iter::once(base).chain(indices).collect(),
    }
}

pub(crate) fn op_load(cx: &Context, output_type: Type, pointer: Value) -> DataInstDef {
    let spv_inst = Inst {
        opcode: Spec::get().well_known.OpLoad,
        imms: SmallVec::default(),
    };
    let form_def = DataInstFormDef {
        kind: DataInstKind::SpvInst(spv_inst),
        output_type: Some(output_type),
    };
    let form = cx.intern(form_def);
    DataInstDef {
        attrs: AttrSet::default(),
        form,
        inputs: std::iter::once(pointer).collect(),
    }
}

pub(crate) fn op_array_length(cx: &Context, array: Value, member: Value) -> DataInstDef {
    let opcode = Spec::get().well_known.OpArrayLength;
    let operand_kind = opcode.def().all_operands().nth(1).unwrap().1;
    let spv_inst = Inst {
        opcode,
        imms: std::iter::once(Imm::Short(operand_kind, 0)).collect(),
    };
    let ty_u32 = op_type_int(cx, 32, false);
    let form_def = DataInstFormDef {
        kind: DataInstKind::SpvInst(spv_inst),
        output_type: Some(ty_u32),
    };
    let form = cx.intern(form_def);
    DataInstDef {
        attrs: AttrSet::default(),
        form,
        inputs: [array].into_iter().collect(),
    }
}

pub(crate) fn op_shift_right_logical(
    cx: &Context,
    output_type: Type,
    base: Value,
    shift: Value,
) -> DataInstDef {
    let opcode = Opcode::try_from_u16_with_name_and_def(Op::ShiftRightLogical as u16)
        .unwrap()
        .0;
    let spv_inst = Inst {
        opcode,
        imms: SmallVec::default(),
    };
    let form_def = DataInstFormDef {
        kind: DataInstKind::SpvInst(spv_inst),
        output_type: Some(output_type),
    };
    let form = cx.intern(form_def);
    DataInstDef {
        attrs: AttrSet::default(),
        form,
        inputs: [base, shift].into_iter().collect(),
    }
}

pub(crate) fn op_bitwise_and(
    cx: &Context,
    output_type: Type,
    lhs: Value,
    rhs: Value,
) -> DataInstDef {
    let opcode = Opcode::try_from_u16_with_name_and_def(Op::BitwiseAnd as u16)
        .unwrap()
        .0;
    let spv_inst = Inst {
        opcode,
        imms: SmallVec::default(),
    };
    let form_def = DataInstFormDef {
        kind: DataInstKind::SpvInst(spv_inst),
        output_type: Some(output_type),
    };
    let form = cx.intern(form_def);
    DataInstDef {
        attrs: AttrSet::default(),
        form,
        inputs: [lhs, rhs].into_iter().collect(),
    }
}

pub(crate) fn op_i_add(cx: &Context, output_type: Type, lhs: Value, rhs: Value) -> DataInstDef {
    let opcode = Opcode::try_from_u16_with_name_and_def(Op::IAdd as u16)
        .unwrap()
        .0;
    let spv_inst = Inst {
        opcode,
        imms: SmallVec::default(),
    };
    let form_def = DataInstFormDef {
        kind: DataInstKind::SpvInst(spv_inst),
        output_type: Some(output_type),
    };
    let form = cx.intern(form_def);
    DataInstDef {
        attrs: AttrSet::default(),
        form,
        inputs: [lhs, rhs].into_iter().collect(),
    }
}

pub(crate) fn op_i_sub(cx: &Context, output_type: Type, lhs: Value, rhs: Value) -> DataInstDef {
    let opcode = Opcode::try_from_u16_with_name_and_def(Op::ISub as u16)
        .unwrap()
        .0;
    let spv_inst = Inst {
        opcode,
        imms: SmallVec::default(),
    };
    let form_def = DataInstFormDef {
        kind: DataInstKind::SpvInst(spv_inst),
        output_type: Some(output_type),
    };
    let form = cx.intern(form_def);
    DataInstDef {
        attrs: AttrSet::default(),
        form,
        inputs: [lhs, rhs].into_iter().collect(),
    }
}

pub(crate) fn op_type_bool(cx: &Context) -> Type {
    let opcode = Spec::get().well_known.OpTypeBool;
    let operand_kinds = opcode.def().all_operands().map(|(_, x)| x);
    let spv_inst = Inst {
        opcode,
        imms: SmallVec::new(),
    };
    let def = TypeDef {
        attrs: AttrSet::default(),
        kind: TypeKind::SpvInst {
            spv_inst,
            type_and_const_inputs: SmallVec::new(),
        },
    };
    cx.intern(def)
}

pub(crate) fn op_type_int(cx: &Context, bits: u32, signed: bool) -> Type {
    let opcode = Spec::get().well_known.OpTypeInt;
    let operand_kinds = opcode.def().all_operands().map(|(_, x)| x);
    let spv_inst = Inst {
        opcode,
        imms: operand_kinds
            .zip([bits, signed as u32])
            .map(|(k, v)| Imm::Short(k, v))
            .collect(),
    };
    let def = TypeDef {
        attrs: AttrSet::default(),
        kind: TypeKind::SpvInst {
            spv_inst,
            type_and_const_inputs: SmallVec::new(),
        },
    };
    cx.intern(def)
}

pub(crate) fn op_constant_true(cx: &Context) -> Const {
    let opcode = Spec::get().well_known.OpConstantTrue;
    let imms = SmallVec::default();
    let ty = op_type_bool(cx);
    let inst = Inst { opcode, imms };
    let const_def = ConstDef {
        attrs: AttrSet::default(),
        ty,
        kind: ConstKind::SpvInst {
            spv_inst_and_const_inputs: Rc::new((inst, SmallVec::new())),
        },
    };
    cx.intern(const_def)
}

pub(crate) fn op_constant(cx: &Context, ty: Type, values: impl IntoIterator<Item = u32>) -> Const {
    let opcode = Spec::get().well_known.OpConstant;
    let operand_kind = opcode.def().all_operands().map(|x| x.1).nth(0).unwrap();
    let values: SmallVec<[u32; 2]> = values.into_iter().collect();
    let imms = match values.as_slice() {
        &[x] => [Imm::Short(operand_kind, x)].into_iter().collect(),
        &[a, b] => [
            Imm::LongStart(operand_kind, a),
            Imm::LongCont(operand_kind, b),
        ]
        .into_iter()
        .collect(),
        _ => unreachable!(),
    };
    let inst = Inst { opcode, imms };
    let const_def = ConstDef {
        attrs: AttrSet::default(),
        ty,
        kind: ConstKind::SpvInst {
            spv_inst_and_const_inputs: Rc::new((inst, SmallVec::new())),
        },
    };
    cx.intern(const_def)
}

pub(crate) fn op_spec_constant(
    cx: &Context,
    attrs: AttrSet,
    ty: Type,
    values: impl IntoIterator<Item = u32>,
) -> Const {
    let spec = Spec::get();

    let (opcode, opname, op_def) =
        Opcode::try_from_u16_with_name_and_def(Op::SpecConstant as u16).unwrap();
    let operand_kind = op_def.all_operands().map(|x| x.1).nth(0).unwrap();
    let values: SmallVec<[u32; 2]> = values.into_iter().collect();
    let imms = match values.as_slice() {
        &[x] => [Imm::Short(operand_kind, x)].into_iter().collect(),
        &[a, b] => [
            Imm::LongStart(operand_kind, a),
            Imm::LongCont(operand_kind, b),
        ]
        .into_iter()
        .collect(),
        _ => unreachable!(),
    };
    let inst = Inst { opcode, imms };
    let const_def = ConstDef {
        attrs,
        ty,
        kind: ConstKind::SpvInst {
            spv_inst_and_const_inputs: Rc::new((inst, SmallVec::new())),
        },
    };
    cx.intern(const_def)
}

/*
pub(crate) fn op_select(
    cx: &Context,
    ty: Type,
    condition: Value,
    a: Value,
    b: Value,
) -> DataInst {
    let spec = Spec::get();
    let (opcode, opname, op_def) =
        Opcode::try_from_u16_with_name_and_def(Op::SpecConstantOp as u16).unwrap();
    let select_opcode = Op::Select as u32;
    let operand_kind = op_def.all_operands().map(|x| x.1).nth(0).unwrap();
    let imms = [Imm::Short(operand_kind, select_opcode)]
        .into_iter()
        .collect();
    let inst = Inst { opcode, imms };
    let const_inputs = [condition, a, b].into_iter().collect();
    let const_def = ConstDef {
        attrs: AttrSet::default(),
        ty,
        kind: ConstKind::SpvInst {
            spv_inst_and_const_inputs: Rc::new((inst, const_inputs)),
        },
    };
    cx.intern(const_def)
}
*/

pub(crate) fn op_spec_constant_select(
    cx: &Context,
    ty: Type,
    condition: Const,
    a: Const,
    b: Const,
) -> Const {
    let spec = Spec::get();
    let (opcode, opname, op_def) =
        Opcode::try_from_u16_with_name_and_def(Op::SpecConstantOp as u16).unwrap();
    let select_opcode = Op::Select as u32;
    let operand_kind = op_def.all_operands().map(|x| x.1).nth(0).unwrap();
    let (_, _, select_def) = Opcode::try_from_u16_with_name_and_def(Op::Select as u16).unwrap();
    let select_operand_kind = select_def.all_operands().map(|x| x.1).nth(0).unwrap();
    let imms = [
        Imm::Short(operand_kind, select_opcode),
        Imm::Short(select_operand_kind, 0),
    ]
    .into_iter()
    .collect();
    let inst = Inst { opcode, imms };
    let const_inputs = [condition, a, b].into_iter().collect();
    let const_def = ConstDef {
        attrs: AttrSet::default(),
        ty,
        kind: ConstKind::SpvInst {
            spv_inst_and_const_inputs: Rc::new((inst, const_inputs)),
        },
    };
    cx.intern(const_def)
}

pub(crate) fn op_name(name: &str) -> Attr {
    let opcode = Spec::get().well_known.OpName;
    let operand_kinds = opcode.def().all_operands().map(|(_, x)| x);
    let inst = Inst {
        opcode,
        imms: encode_literal_string(name).collect(),
    };
    Attr::SpvAnnotation(inst)
}

pub(crate) fn op_decorate_block() -> Attr {
    let opcode = Spec::get().well_known.OpDecorate;
    let operand_kinds = opcode.def().all_operands().map(|(_, x)| x);
    let inst = Inst {
        opcode,
        imms: operand_kinds
            .skip(1)
            .zip([Decoration::Block as u32])
            .map(|(k, v)| Imm::Short(k, v))
            .collect(),
    };
    Attr::SpvAnnotation(inst)
}

pub(crate) fn op_member_decorate_offset(index: u32, offset: u32) -> Attr {
    let opcode = Spec::get().well_known.OpMemberDecorate;
    let operand_kinds: Vec<_> = opcode.def().all_operands().map(|(_, x)| x).collect();
    let inst = Inst {
        opcode,
        imms: [operand_kinds[1], operand_kinds[2], operand_kinds[1]]
            .into_iter()
            .zip([index, Decoration::Offset as u32, offset])
            .map(|(k, v)| Imm::Short(k, v))
            .collect(),
    };
    Attr::SpvAnnotation(inst)
}

pub(crate) fn op_member_name(index: u32, name: &str) -> Attr {
    let opcode = Spec::get().well_known.OpMemberName;
    let operand_kinds = opcode.def().all_operands().map(|(_, x)| x);
    let inst = Inst {
        opcode,
        imms: operand_kinds
            .skip(1)
            .zip(std::iter::once(index))
            .map(|(k, v)| Imm::Short(k, v))
            .chain(encode_literal_string(name))
            .collect(),
    };
    Attr::SpvAnnotation(inst)
}

pub(crate) fn op_type_struct(
    cx: &Context,
    attrs: AttrSet,
    types: impl IntoIterator<Item = Type>,
) -> Type {
    let types: Vec<_> = types.into_iter().collect();
    let spec = Spec::get();
    let struct_def = TypeDef {
        attrs,
        kind: TypeKind::SpvInst {
            spv_inst: Inst {
                opcode: spec.well_known.OpTypeStruct,
                imms: SmallVec::new(),
            },
            type_and_const_inputs: types.into_iter().map(TypeOrConst::Type).collect(),
        },
    };
    cx.intern(struct_def)
}

pub(crate) fn op_type_array(cx: &Context, attrs: AttrSet, elem_ty: Type, len: Const) -> Type {
    let spec = Spec::get();
    let array_def = TypeDef {
        attrs,
        kind: TypeKind::SpvInst {
            spv_inst: Inst {
                opcode: spec.well_known.OpTypeArray,
                imms: SmallVec::new(),
            },
            type_and_const_inputs: [TypeOrConst::Type(elem_ty), TypeOrConst::Const(len)]
                .into_iter()
                .collect(),
        },
    };
    cx.intern(array_def)
}

pub(crate) fn strip_krnl_insts(
    module: &mut Module,
    funcs: impl Iterator<Item = Func>,
    remove: Vec<KrnlInst>,
) {
    use spirt::transform::Transformed;

    struct KrnlInstTransformer {
        cx: Rc<Context>,
        krnl_set: InternedStr,
        remove: Vec<KrnlInst>,
        nop: DataInstDef,
    }

    impl Transformer for KrnlInstTransformer {
        fn in_place_transform_data_inst_def(
            &mut self,
            mut func_at_data_inst: spirt::func_at::FuncAtMut<'_, DataInst>,
        ) {
            let mut def = func_at_data_inst.def();
            let data_inst_form_def = &self.cx[def.form];
            if let DataInstKind::SpvExtInst { ext_set, inst } = data_inst_form_def.kind {
                if ext_set == self.krnl_set && self.remove.iter().any(|x| *x as u32 == inst) {
                    *def = self.nop.clone();
                }
            }
        }
    }

    let cx = module.cx();
    let krnl_set = module.cx_ref().intern(KrnlInst::SET_NAME);
    let nop = op_nop(&cx);

    let mut transformer = KrnlInstTransformer {
        cx,
        krnl_set,
        remove,
        nop,
    };

    for func in funcs {
        module.funcs[func].inner_in_place_transform_with(&mut transformer);
    }
}
