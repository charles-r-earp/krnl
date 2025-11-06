use crate::reflect::{
    BufferDesc, ElementType, Features, KernelDesc, PushConstantDesc, UsedGlobals, get_scalar_type,
    get_struct_field_types, get_struct_size,
};
use crate::scalar::ScalarType;
use crate::spirv::{
    assemble, constant_name, get_constant_u32, get_element_size, get_name_from_attrs,
    krnl_inst_set, op_access_chain, op_array_length, op_constant, op_decorate_block, op_i_add,
    op_i_sub, op_load, op_member_decorate_offset, op_member_name, op_name, op_nop, op_type_int,
    op_type_pointer, op_type_struct, op_variable, pointee_type, strip_krnl_insts,
    struct_element_type, validate, variable_name,
};
use camino::{Utf8Path, Utf8PathBuf};
use derive_more::IsVariant;
use fxhash::FxBuildHasher;
use indexmap::{
    IndexMap, IndexSet,
    map::{MutableEntryKey, MutableKeys},
};
use krnl_core::__private::__KrnlInst as KrnlInst;
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use smallvec::SmallVec;
use spirt::spv::spec::Opcode;
use spirt::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, ControlNodeKind,
    DataInst, DataInstDef, DataInstForm, DataInstFormDef, DataInstKind, DeclDef, ExportKey,
    Exportee, Func, GlobalVar, GlobalVarDecl, GlobalVarDefBody, InternedStr, Module, Type, TypeDef,
    TypeKind, TypeOrConst, Value,
    spv::{
        Imm, Inst, encode_literal_string, extract_literal_string,
        spec::{ExtInstSetDesc, ExtInstSetInstructionDesc, Spec},
    },
    transform::{InnerInPlaceTransform, Transformed, Transformer},
    visit::{InnerVisit, Visitor},
};
use spirt::{EntityList, FuncDefBody};
use spirv_headers::{Decoration, ExecutionModel, StorageClass};
use spirv_tools::{
    TargetEnv,
    binary::Binary,
    opt::{Optimizer, Options as OptimizerOptions, Passes},
    val::Validator,
};
use std::array;
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};
use syn::{Ident, LitInt, buffer};

#[derive(Default)]
pub struct BindingsBuilder {
    spirv: Option<Vec<u8>>,
    buffer_offsets: bool,
}

impl BindingsBuilder {
    /*
    pub fn from_spirv_bytes(bytes: Vec<u8>) -> Self {
        Self {
            spirv: Some(bytes),
            ..Self::default()
        }
    }
    */
    pub fn emit(self) -> std::io::Result<()> {
        let crate_name = std::env::var("CARGO_PKG_NAME").unwrap().replace("-", "_");
        let out_dir = Utf8PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let spirv = if let Some(spirv) = self.spirv {
            spirv
        } else {
            let manifest_dir = Utf8PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
            let path = manifest_dir.join("krnl.spv");
            std::fs::read(path)?
        };
        let kernels = process(spirv);
        for kernel in kernels {
            let name = &kernel.sig.name;
            let path = name.replace("::", "__");
            let output = kernel.emit();
            std::fs::write(out_dir.join(&path), &output)?;
            println!("cargo:rustc-env=KRNL_INCLUDE_{crate_name}::{name}={path}");
        }
        Ok(())
    }
}

fn process(spirv: Vec<u8>) -> Vec<Kernel> {
    let target_family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
    let debug_assertions = std::env::var("CARGO_CFG_DEBUG_ASSERTIONS").is_ok();
    let non_semantic_info = target_family != "wasm" && debug_assertions;
    let context = Rc::new(Context::new());
    context.register_custom_ext_inst_set(KrnlInst::SET_NAME, krnl_inst_set());
    let mut module = Module::lower_from_spv_bytes(context.clone(), spirv).unwrap();
    let sigs: Vec<_> = module
        .exports
        .keys()
        .cloned()
        .map(|entry_point| KernelSig::reflect(&module, &entry_point))
        .collect();
    let module_globals = UsedGlobals::parse_module(&module, None);
    strip_krnl_insts(
        &mut module,
        module_globals.funcs.iter().copied(),
        KrnlInst::iter().collect(),
    );
    rename_entry_points(&mut module, "main");
    spirt::passes::legalize::structurize_func_cfgs(&mut module);
    let features_spirvs: Vec<_> = split_entry_points(&module, target_family == "wasm").collect();
    let kernels = sigs
        .into_par_iter()
        .zip(features_spirvs)
        .map(|(sig, (features, mut spirv))| {
            use rspirv::binary::{Assemble, Disassemble};
            {
                use rspirv::binary::{Assemble, Disassemble};

                let module = rspirv::dr::load_words(&spirv).unwrap();
                let mut builder = rspirv::dr::Builder::new_from_module(module);
                if target_family != "wasm" {
                    fix_execution_mode(&mut builder);
                }
                spirv = builder.module().assemble();
                validate(&spirv).expect("spirv-val after fix_execution_mode");
            }
            let mut optimizer = spirv_tools::opt::create(Some(TargetEnv::Vulkan_1_3));
            if !non_semantic_info {
                optimizer.register_pass(Passes::StripNonSemanticInfo);
            }
            optimizer.register_performance_passes();
            let options = OptimizerOptions {
                preserve_bindings: true,
                preserve_spec_constants: true,
                ..OptimizerOptions::default()
            };
            let binary = optimizer
                .optimize(&spirv, &mut |_| (), Some(options))
                .unwrap();
            spirv = if let Binary::OwnedU32(words) = binary {
                words
            } else {
                binary.as_words().to_vec()
            };
            Kernel {
                sig,
                features,
                spirv,
            }
        })
        .collect();
    kernels
}

fn rename_entry_points(module: &mut Module, entry_point: &str) {
    module.exports = std::mem::take(&mut module.exports)
        .into_iter()
        .map(|(mut key, value)| {
            if let ExportKey::SpvEntryPoint {
                imms,
                interface_global_vars,
            } = &mut key
            {
                *imms = std::iter::once(imms[0])
                    .chain(encode_literal_string(entry_point))
                    .collect();
            } else {
                unreachable!()
            };
            (key, value)
        })
        .collect();
}

fn add_buffer_offsets(module: &mut Module) {
    #[derive(Default)]
    struct FuncCollector {
        funcs: IndexSet<Func, FxBuildHasher>,
    }

    impl Visitor<'_> for FuncCollector {
        fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
        fn visit_type_use(&mut self, _ty: Type) {}
        fn visit_const_use(&mut self, _ct: Const) {}
        fn visit_data_inst_form_use(&mut self, _data_inst_form: DataInstForm) {}
        fn visit_global_var_use(&mut self, _gv: GlobalVar) {}
        fn visit_func_use(&mut self, func: Func) {
            self.funcs.insert(func);
        }
    }

    struct ModuleTransformer {
        cx: Rc<Context>,
        old_push_gv: Option<GlobalVar>,
        push_constants: Value,
        buffer_pointers: IndexMap<GlobalVar, Const, FxBuildHasher>,
        buffer_offset_indices: IndexMap<GlobalVar, Const, FxBuildHasher>,
        buffer_offsets: IndexMap<GlobalVar, DataInst, FxBuildHasher>,
        array_lengths: IndexMap<GlobalVar, DataInst, FxBuildHasher>,
    }

    impl ModuleTransformer {
        fn new(module: &mut Module) -> Self {
            let cx = module.cx();
            let module_globals = UsedGlobals::parse_module(&module, None);

            let (old_push_gv, push_constants, buffer_pointers, buffer_offset_indices) = {
                let buffer_count = module_globals
                    .vars
                    .iter()
                    .filter(|x| *x.1 == StorageClass::StorageBuffer)
                    .count();
                let mut push_constant_field_types = Vec::new();
                let mut push_constant_attrs = BTreeSet::default();
                let push_constants = module_globals
                    .vars
                    .iter()
                    .find(|x| *x.1 == StorageClass::PushConstant)
                    .map(|x| x.0)
                    .copied();
                let old_push_gv = push_constants;
                let mut offset = 0;
                if let Some(push_constants) = push_constants {
                    let push_constants_decl = &mut module.global_vars[push_constants];
                    let pointer_ty = push_constants_decl.type_of_ptr_to;
                    let push_struct = pointee_type(&cx, pointer_ty).unwrap();
                    offset = get_struct_size(&cx, push_struct).unwrap();
                    push_constant_field_types =
                        get_struct_field_types(&cx, push_struct).unwrap().collect();
                    let push_struct_def = &cx[push_struct];
                    push_constant_attrs = cx[push_struct_def.attrs].attrs.clone();
                } else {
                    push_constant_attrs.insert(op_decorate_block());
                }

                while offset % 4 != 0 {
                    offset += 1;
                }
                let ty_u32 = op_type_int(&cx, 32, false);
                let mut buffer_pointers = IndexMap::default();
                let mut buffer_offset_indices = IndexMap::default();
                for gv in module_globals
                    .vars
                    .iter()
                    .filter(|x| *x.1 == StorageClass::StorageBuffer)
                    .map(|x| x.0)
                    .copied()
                {
                    let name = variable_name(module, gv).unwrap();
                    let name = format!("krnl::offset_{name}");
                    let index = push_constant_field_types.len() as u32;
                    push_constant_field_types.push(ty_u32);
                    push_constant_attrs.extend([
                        op_member_name(index, &name),
                        op_member_decorate_offset(index, offset),
                    ]);
                    offset += 4;
                    let index = op_constant(&cx, ty_u32, [index]);
                    buffer_offset_indices.insert(gv, index);
                    let pointer = cx.intern(ConstDef {
                        attrs: AttrSet::default(),
                        ty: module.global_vars[gv].type_of_ptr_to,
                        kind: ConstKind::PtrToGlobalVar(gv),
                    });
                    buffer_pointers.insert(gv, pointer);
                }
                let push_constant_attrs = cx.intern(AttrSetDef {
                    attrs: push_constant_attrs,
                });
                let push_struct =
                    op_type_struct(&cx, push_constant_attrs, push_constant_field_types);
                let push_ptr = op_type_pointer(&cx, push_struct, StorageClass::PushConstant);
                let push_constants = if let Some(push_constants) = push_constants {
                    let push_constants_decl = &mut module.global_vars[push_constants];
                    push_constants_decl.type_of_ptr_to = push_ptr;
                    push_constants
                } else {
                    let attrs = cx.intern(AttrSetDef {
                        attrs: [op_name("krnl::push_constants")].into_iter().collect(),
                    });
                    let push_constants =
                        op_variable(module, push_ptr, StorageClass::PushConstant, attrs);
                    push_constants
                };
                let push_constants = Value::Const(cx.intern(ConstDef {
                    attrs: AttrSet::default(),
                    ty: push_ptr,
                    kind: ConstKind::PtrToGlobalVar(push_constants),
                }));
                (
                    old_push_gv,
                    push_constants,
                    buffer_pointers,
                    buffer_offset_indices,
                )
            };
            Self {
                cx,
                old_push_gv,
                push_constants,
                buffer_pointers,
                buffer_offset_indices,
                buffer_offsets: IndexMap::default(),
                array_lengths: IndexMap::default(),
            }
        }
        fn in_place_transform_func_def_body(&mut self, func_def_body: &mut FuncDefBody) {
            let cx = self.cx.clone();
            self.buffer_offsets.clear();
            self.array_lengths.clear();
            let ty_u32 = op_type_int(&cx, 32, false);
            let zero_u32 = op_constant(&cx, ty_u32, [0]);
            let ptr_u32_push = op_type_pointer(&cx, ty_u32, StorageClass::PushConstant);
            let func_at_mut_body = func_def_body.at_mut_body();
            let func_at_mut_children = func_at_mut_body.at_children();
            let mut func_at_mut_children_iter = func_at_mut_children.into_iter();
            let mut func_at_node = func_at_mut_children_iter.next().unwrap();
            let mut new_insts =
                if let ControlNodeKind::Block { insts } = &mut func_at_node.reborrow().def().kind {
                    std::mem::take(insts)
                } else {
                    unreachable!()
                };
            let vars = {
                let mut vars = EntityList::empty();
                let mut iter = func_at_node.reborrow().at(new_insts.iter());
                while let Some(mut func_at_inst) = iter.next() {
                    let data_inst_def = func_at_inst.reborrow().def();
                    let form_def = &cx[data_inst_def.form];
                    if let DataInstKind::SpvInst(inst) = &form_def.kind {
                        if inst.opcode == Spec::get().well_known.OpVariable {
                            let var = func_at_inst.position;
                            new_insts.remove(var, func_at_inst.data_insts);
                            vars.insert_last(var, func_at_inst.data_insts);
                        }
                    }
                }
                vars
            };
            let data_insts = &mut func_at_node.data_insts;
            for (gv, index) in self.buffer_offset_indices.iter().map(|(a, b)| (*a, *b)) {
                let offset_access_chain = data_insts.define(
                    &cx,
                    op_access_chain(
                        &self.cx,
                        ptr_u32_push,
                        self.push_constants,
                        [Value::Const(index)],
                    )
                    .into(),
                );

                let offset_load = data_insts.define(
                    &cx,
                    op_load(&cx, ty_u32, Value::DataInstOutput(offset_access_chain)).into(),
                );
                new_insts.insert_first(offset_load, data_insts);
                new_insts.insert_before(offset_access_chain, offset_load, data_insts);
                self.buffer_offsets.insert(gv, offset_load);
                let array: Value = Value::Const(self.buffer_pointers.get(&gv).copied().unwrap());
                let array_length = data_insts.define(
                    &cx,
                    op_array_length(&cx, array, Value::Const(zero_u32)).into(),
                );
                new_insts.insert_first(array_length, data_insts);
                self.array_lengths.insert(gv, array_length);
            }
            new_insts.prepend(vars, data_insts);
            if let ControlNodeKind::Block { insts } = &mut func_at_node.reborrow().def().kind {
                *insts = new_insts;
            } else {
                unreachable!()
            };
        }
        fn in_place_transform_buffer_access_chain(
            &mut self,
            mut func_at_data_inst: spirt::func_at::FuncAtMut<'_, DataInst>,
            insts: &mut EntityList<DataInst>,
            base: GlobalVar,
        ) {
            let cx = self.cx.clone();
            let data_inst_def = func_at_data_inst.reborrow().def();
            let ty_u32 = op_type_int(&cx, 32, false);
            let buffer_offset_index = self.buffer_offset_indices.get(&base).copied().unwrap();
            let index = data_inst_def.inputs.last().copied().unwrap();

            let offset_load = self.buffer_offsets.get(&base).copied().unwrap();
            let index_add = func_at_data_inst.data_insts.define(
                &cx,
                op_i_add(&cx, ty_u32, index, Value::DataInstOutput(offset_load)).into(),
            );
            insts.insert_before(
                index_add,
                func_at_data_inst.position,
                func_at_data_inst.data_insts,
            );
            let data_inst_def = func_at_data_inst.reborrow().def();
            *data_inst_def.inputs.last_mut().unwrap() = Value::DataInstOutput(index_add);
        }
        fn in_place_transform_buffer_array_length(
            &mut self,
            mut func_at_data_inst: spirt::func_at::FuncAtMut<'_, DataInst>,
            insts: &mut EntityList<DataInst>,
            base: GlobalVar,
        ) {
            let cx = self.cx.clone();
            let data_inst_def = func_at_data_inst.reborrow().def();
            let ty_u32 = op_type_int(&cx, 32, false);
            let buffer_offset_index = self.buffer_offset_indices.get(&base).copied().unwrap();
            let array_len_def = data_inst_def.clone();

            let offset_load = self.buffer_offsets.get(&base).copied().unwrap();
            let array_length = self.array_lengths.get(&base).copied().unwrap();
            let data_inst_def = func_at_data_inst.reborrow().def();
            *data_inst_def = op_i_sub(
                &cx,
                ty_u32,
                Value::DataInstOutput(array_length),
                Value::DataInstOutput(offset_load),
            );
        }
    }

    impl Transformer for ModuleTransformer {
        fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
            if let Some(old_push_gv) = self.old_push_gv {
                if let Value::Const(ct) = *v {
                    let ct_def = &self.cx[ct];
                    if let ConstKind::PtrToGlobalVar(gv) = ct_def.kind {
                        if gv == old_push_gv {
                            return Transformed::Changed(self.push_constants);
                        }
                    }
                }
            }
            Transformed::Unchanged
        }
        fn in_place_transform_func_decl(&mut self, func_decl: &mut spirt::FuncDecl) {
            if let DeclDef::Present(func_def_body) = &mut func_decl.def {
                self.in_place_transform_func_def_body(func_def_body);
            }
            func_decl.inner_in_place_transform_with(self);
        }
        fn in_place_transform_control_node_def(
            &mut self,
            mut func_at_control_node: spirt::func_at::FuncAtMut<'_, spirt::ControlNode>,
        ) {
            if let ControlNodeKind::Block { insts } = func_at_control_node.reborrow().def().kind {
                let mut func_at_iter = func_at_control_node.reborrow().at(insts).into_iter();
                let mut new_insts = insts;
                let spec = Spec::get();
                while let Some(mut func_at_data_inst) = func_at_iter.next() {
                    let data_inst = func_at_data_inst.position;
                    let data_inst_def = func_at_data_inst.reborrow().def();
                    let form_def = &self.cx[data_inst_def.form];
                    if let DataInstKind::SpvInst(inst) = &form_def.kind {
                        if let Some(Value::Const(base)) = data_inst_def.inputs.first().copied() {
                            let base = &self.cx[base];
                            if let ConstKind::PtrToGlobalVar(base) = base.kind {
                                if inst.opcode == spec.well_known.OpAccessChain {
                                    if self.buffer_offset_indices.contains_key(&base) {
                                        self.in_place_transform_buffer_access_chain(
                                            func_at_data_inst,
                                            &mut new_insts,
                                            base,
                                        );
                                    }
                                } else if inst.opcode == spec.well_known.OpArrayLength {
                                    if self.array_lengths.get(&base) != Some(&data_inst)
                                        && self.buffer_offset_indices.contains_key(&base)
                                    {
                                        self.in_place_transform_buffer_array_length(
                                            func_at_data_inst,
                                            &mut new_insts,
                                            base,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                if let ControlNodeKind::Block { insts } =
                    &mut func_at_control_node.reborrow().def().kind
                {
                    *insts = new_insts;
                }
            }
            func_at_control_node.inner_in_place_transform_with(self);
        }
    }

    /*
    let entry_func = if let Exportee::Func(func) = module.exports[entry_point] {
        func
    } else {
        unreachable!()
    };
    */

    //

    let mut collector = FuncCollector::default();
    module.inner_visit_with(&mut collector);

    let mut transformer = ModuleTransformer::new(module);

    for func in collector.funcs {
        transformer.in_place_transform_func_decl(&mut module.funcs[func]);
    }
}

fn split_entry_points(
    module: &Module,
    wgsl: bool,
) -> impl Iterator<Item = (Features, Vec<u32>)> + '_ {
    module.exports.iter().map(move |(key, value)| {
        let mut module = module.clone();
        module.exports = std::iter::once((key.clone(), value.clone())).collect();
        let features = Features::reflect(&module).wgsl(wgsl);
        features.write_to_module(&mut module);
        {
            if let ExportKey::SpvEntryPoint {
                imms,
                interface_global_vars,
            } = key
            {
                add_buffer_offsets(&mut module);
            } else {
                unreachable!()
            };
        }
        let spirv = assemble(&module).unwrap();
        validate(&spirv).expect("spirv-val after split");
        (features, spirv)
    })
}

fn fix_execution_mode(builder: &mut rspirv::dr::Builder) {
    use rspirv::{
        dr::{Instruction, Operand},
        spirv::{ExecutionMode, Op},
    };

    let uint = builder
        .module_ref()
        .types_global_values
        .iter()
        .find_map(|inst| {
            if inst.class.opcode == Op::TypeInt {
                if inst.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)] {
                    return inst.result_id;
                }
            }
            None
        });
    let uint = if let Some(uint) = uint {
        uint
    } else {
        builder.type_int(0, 0)
    };
    let threads = builder.module_ref().debug_names.iter().find_map(|inst| {
        if inst.class.opcode == Op::Name {
            if inst.operands[1].unwrap_literal_string() == "krnl::threads" {
                return Some(inst.operands[0].unwrap_id_ref());
            }
        }
        None
    });
    let threads = if let Some(threads) = threads {
        threads
    } else {
        let threads = builder.spec_constant_bit32(uint, 1);
        builder.name(threads, "krnl::threads");
        threads
    };
    let one = builder
        .module_ref()
        .types_global_values
        .iter()
        .find_map(|inst| {
            if inst.class.opcode == Op::Constant {
                if inst.result_type == Some(uint) {
                    if inst.operands == [Operand::LiteralBit32(1)] {
                        return inst.result_id;
                    }
                }
            }
            None
        });
    let one = if let Some(one) = one {
        one
    } else {
        builder.constant_bit32(uint, 1)
    };
    builder.module_mut().execution_modes.clear();
    let entry_points: Vec<u32> = builder
        .module_ref()
        .entry_points
        .iter()
        .map(|inst| inst.operands[1].unwrap_id_ref())
        .collect();
    for entry_point in entry_points {
        /*
        builder.module_mut().execution_modes.push(Instruction::new(
            Op::ExecutionModeId,
            None,
            None,
            vec![
                Operand::IdRef(entry_point),
                Operand::ExecutionMode(ExecutionMode::LocalSizeId),
                Operand::IdRef(threads),
                Operand::IdRef(zero),
                Operand::IdRef(zero),
            ],
        ));
        */
        //TODO: https://github.com/gfx-rs/rspirv/pull/263
        builder.execution_mode_id(entry_point, ExecutionMode::LocalSizeId, [threads, one, one]);
    }
}

#[derive(Debug)]
struct SpecDesc {
    name: String,
    id: u32,
    scalar_type: ScalarType,
    array: Option<u32>,
}

#[derive(Debug, derive_more::IsVariant)]
enum KernelInput {
    Spec(SpecDesc),
    Buffer(BufferDesc),
    Item(BufferDesc),
    Push(PushConstantDesc),
}

impl ToTokens for KernelInput {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Spec(desc) => {
                let ident = Ident::new(&desc.name, Span::call_site());
                let elem_ty = desc.scalar_type.type_path();
                let elem_ty = if let Some(len) = desc.array {
                    let len = LitInt::new(&len.to_string(), Span::call_site());
                    quote! {
                        [#elem_ty; #len]
                    }
                } else {
                    elem_ty.to_token_stream()
                };
                quote! {
                    #ident: krnl::kernel::__private::__Spec<#elem_ty>
                }
                .to_tokens(tokens);
            }
            Self::Buffer(desc) => {
                let ident = Ident::new(&desc.name(), Span::call_site());
                let scalar_type = desc.scalar_type();
                let mut elem_ty =
                    Ident::new(&scalar_type.to_string(), Span::call_site()).to_token_stream();
                if matches!(scalar_type, ScalarType::F16 | ScalarType::BF16) {
                    elem_ty = quote! {
                        krnl::scalar::#elem_ty
                    };
                }
                if let Some(len) = desc.array() {
                    let len = LitInt::new(&len.to_string(), Span::call_site());
                    elem_ty = quote! {
                        [#elem_ty; #len]
                    }
                }
                if desc.mutable() {
                    elem_ty = quote! {
                        ::core::cell::UnsafeCell<#elem_ty>
                    };
                }
                quote! {
                    #ident: krnl::kernel::__private::__Buffer<[#elem_ty]>
                }
                .to_tokens(tokens);
            }
            Self::Item(desc) => {
                let ident = Ident::new(&desc.name(), Span::call_site());
                let scalar_type = desc.scalar_type();
                let mut elem_ty =
                    Ident::new(&scalar_type.to_string(), Span::call_site()).to_token_stream();
                if matches!(scalar_type, ScalarType::F16 | ScalarType::BF16) {
                    elem_ty = quote! {
                        krnl::scalar::#elem_ty
                    };
                }
                if let Some(len) = desc.array() {
                    let len = LitInt::new(&len.to_string(), Span::call_site());
                    elem_ty = quote! {
                        [#elem_ty; #len]
                    }
                }
                let kind = if desc.mutable() {
                    format_ident!("__ItemMut")
                } else {
                    format_ident!("__Item")
                };
                quote! {
                    #ident: krnl::kernel::__private::#kind<#elem_ty>
                }
                .to_tokens(tokens);
            }
            Self::Push(desc) => {
                let name = desc.name();
                let name = if name == "krnl::items" {
                    "__krnl_items"
                } else {
                    name
                };
                let ident = Ident::new(name, Span::call_site());
                let scalar_type = desc.scalar_type();
                let mut elem_ty =
                    Ident::new(&scalar_type.to_string(), Span::call_site()).to_token_stream();
                if matches!(scalar_type, ScalarType::F16 | ScalarType::BF16) {
                    elem_ty = quote! {
                        krnl::scalar::#elem_ty
                    };
                }
                if let Some(len) = desc.array() {
                    let len = LitInt::new(&len.to_string(), Span::call_site());
                    elem_ty = quote! {
                        [#elem_ty; #len]
                    }
                }
                quote! {
                    #ident: krnl::kernel::__private::__Push<#elem_ty>
                }
                .to_tokens(tokens);
            }
        }
    }
}

struct KernelSig {
    name: String,
    safe: bool,
    inputs: Vec<KernelInput>,
}

impl KernelSig {
    fn reflect(module: &Module, entry_point: &ExportKey) -> Self {
        let name = if let ExportKey::SpvEntryPoint {
            imms,
            interface_global_vars,
        } = entry_point
        {
            extract_literal_string(&imms[1..]).unwrap()
        } else {
            unreachable!()
        };
        let entry_func = if let Exportee::Func(func) = module.exports[entry_point] {
            func
        } else {
            unreachable!()
        };
        let kernel_desc = KernelDesc::reflect(module, entry_func);

        struct KernelVisitor<'a> {
            module: &'a Module,
            func: Func,
            krnl_set: InternedStr,
            kernel_desc: &'a KernelDesc,
            items: IndexSet<GlobalVar, FxBuildHasher>,
            buffer_access: IndexMap<DataInst, GlobalVar, FxBuildHasher>,
            push_access: IndexMap<DataInst, (GlobalVar, u32), FxBuildHasher>,
            safe: bool,
            inputs: Vec<KernelInput>,
        }

        impl<'a> KernelVisitor<'a> {
            fn new(module: &'a Module, func: Func, kernel_desc: &'a KernelDesc) -> Self {
                let krnl_set = module.cx_ref().intern(KrnlInst::SET_NAME);
                Self {
                    module,
                    func,
                    krnl_set,
                    kernel_desc,
                    items: IndexSet::default(),
                    buffer_access: IndexMap::default(),
                    push_access: IndexMap::default(),
                    safe: false,
                    inputs: Vec::new(),
                }
            }
            fn visit_spec(&mut self, ct: Const) {
                let cx = self.module.cx_ref();
                let const_def = &cx[ct];
                let ty = const_def.ty;
                let name = constant_name(cx, ct).unwrap();
                let element_type = ElementType::from_type(cx, ty).unwrap();
                let (scalar, array) = match element_type {
                    ElementType::Scalar(scalar) => (scalar, None),
                    ElementType::Array(scalar, array) => (scalar, Some(array)),
                };
                let scalar_type = get_scalar_type(cx, scalar).unwrap();
                let input = KernelInput::Spec(SpecDesc {
                    name,
                    id: todo!(),
                    scalar_type,
                    array,
                });
                self.inputs.push(input);
            }
            fn visit_buffer(&mut self, gv: GlobalVar) {
                let name = variable_name(&self.module, gv).unwrap();
                let mut desc = self
                    .kernel_desc
                    .buffers
                    .iter()
                    .find(|x| x.name() == name)
                    .unwrap()
                    .clone();
                let input = if self.items.contains(&gv) {
                    KernelInput::Item(desc)
                } else {
                    KernelInput::Buffer(desc)
                };
                self.inputs.push(input);
            }
            fn visit_push(&mut self, gv: GlobalVar, member: u32) {
                let desc = self.kernel_desc.push_constants[member as usize].clone();
                self.inputs.push(KernelInput::Push(desc));
            }
            fn visit_data_inst(&mut self, data_inst: DataInst) {
                let cx = self.module.cx_ref();
                let func_decl = &self.module.funcs[self.func];
                let func_def = if let DeclDef::Present(func_def) = &func_decl.def {
                    func_def
                } else {
                    unreachable!()
                };
                let data_inst_def = &func_def.data_insts[data_inst];
                let data_inst_form_def = &cx[data_inst_def.form];
                let spec = Spec::get();
                match &data_inst_form_def.kind {
                    DataInstKind::SpvInst(inst) => {
                        if inst.opcode == spec.well_known.OpAccessChain {
                            if let &[Value::Const(gv), Value::Const(_), Value::Const(_), ..] =
                                data_inst_def.inputs.as_slice()
                            {
                                let const_def = &cx[gv];
                                if let &ConstKind::PtrToGlobalVar(gv) = &const_def.kind {
                                    let var_def = &self.module.global_vars[gv];
                                    if var_def.addr_space
                                        == AddrSpace::SpvStorageClass(
                                            StorageClass::StorageBuffer as u32,
                                        )
                                    {
                                        self.buffer_access.insert(data_inst, gv);
                                    }
                                }
                            } else if let &[Value::Const(gv), Value::Const(member)] =
                                data_inst_def.inputs.as_slice()
                            {
                                let const_def = &cx[gv];
                                if let &ConstKind::PtrToGlobalVar(gv) = &const_def.kind {
                                    let var_def = &self.module.global_vars[gv];
                                    if var_def.addr_space
                                        == AddrSpace::SpvStorageClass(
                                            StorageClass::PushConstant as u32,
                                        )
                                    {
                                        let member = get_constant_u32(&cx, member).unwrap();
                                        self.push_access.insert(data_inst, (gv, member));
                                    }
                                }
                            }
                        }
                    }
                    &DataInstKind::SpvExtInst { ext_set, inst } if ext_set == self.krnl_set => {
                        let inst = KrnlInst::from_u32(inst).unwrap();
                        match inst {
                            KrnlInst::Item => {
                                if let &[Value::DataInstOutput(access_chain)] =
                                    data_inst_def.inputs.as_slice()
                                {
                                    let gv =
                                        self.buffer_access.get(&access_chain).copied().unwrap();
                                    self.items.insert(gv);
                                } else {
                                    unreachable!();
                                }
                            }
                            KrnlInst::Input => {
                                if let &[Value::DataInstOutput(access_chain)] =
                                    data_inst_def.inputs.as_slice()
                                {
                                    if let Some(gv) = self.buffer_access.get(&access_chain).copied()
                                    {
                                        self.visit_buffer(gv);
                                    } else if let Some((gv, member)) =
                                        self.push_access.get(&access_chain).copied()
                                    {
                                        self.visit_push(gv, member);
                                    } else {
                                        /* group buffer */
                                        //unreachable!();
                                    }
                                } else if let &[Value::Const(ct)] = data_inst_def.inputs.as_slice()
                                {
                                    self.visit_spec(ct);
                                } else {
                                    for value in data_inst_def.inputs.iter() {
                                        match value {
                                            Value::Const(_) => {
                                                dbg!("Const");
                                            }
                                            Value::ControlNodeOutput {
                                                control_node,
                                                output_idx,
                                            } => {
                                                dbg!("Output");
                                            }
                                            Value::ControlRegionInput { region, input_idx } => {
                                                dbg!("Input");
                                            }
                                            Value::DataInstOutput(_) => {
                                                dbg!("DataInst");
                                            }
                                        }
                                    }
                                    unreachable!()
                                }
                            }
                            KrnlInst::DataType => {
                                if let &[
                                    Value::DataInstOutput(access_chain),
                                    Value::Const(data_type_ct),
                                ] = data_inst_def.inputs.as_slice()
                                {
                                    let data_type_def = &cx[data_type_ct];
                                    let scalar_type =
                                        if let ConstKind::SpvStringLiteralForExtInst(data_type) =
                                            data_type_def.kind
                                        {
                                            match &cx[data_type] {
                                                "f16" => ScalarType::F16,
                                                "bf16" => ScalarType::BF16,
                                                ty => unreachable!("{ty}"),
                                            }
                                        } else {
                                            unreachable!()
                                        };
                                    if let Some(gv) = self.buffer_access.get(&access_chain).copied()
                                    {
                                        let name = variable_name(self.module, gv).unwrap();
                                        for input in self.inputs.iter_mut() {
                                            if let KernelInput::Buffer(desc)
                                            | KernelInput::Item(desc) = input
                                            {
                                                if desc.name() == name {
                                                    desc.scalar_type = scalar_type;
                                                    break;
                                                }
                                            }
                                        }
                                    } else if let Some((_gv, member)) =
                                        self.push_access.get(&access_chain).copied()
                                    {
                                        let mut index = 0;
                                        for input in self.inputs.iter_mut() {
                                            if let KernelInput::Push(desc) = input {
                                                if index == member {
                                                    desc.scalar_type = scalar_type;
                                                    break;
                                                } else {
                                                    index += 1;
                                                }
                                            }
                                        }
                                        assert_eq!(index, member);
                                    } else {
                                        unreachable!()
                                    }
                                } else {
                                    unreachable!()
                                }
                            }
                            _ => (),
                        }
                    }
                    _ => (),
                }
            }
        }

        impl Visitor<'_> for KernelVisitor<'_> {
            fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
            fn visit_type_use(&mut self, _ty: Type) {}
            fn visit_const_use(&mut self, _ct: Const) {}
            fn visit_data_inst_form_use(&mut self, _data_inst_form: DataInstForm) {}
            fn visit_global_var_use(&mut self, _gv: GlobalVar) {}
            fn visit_func_use(&mut self, _func: Func) {}
            fn visit_data_inst_def(&mut self, data_inst_def: &DataInstDef) {}
            fn visit_control_node_def(
                &mut self,
                func_at_control_node: spirt::func_at::FuncAt<'_, spirt::ControlNode>,
            ) {
                if let ControlNodeKind::Block { insts } = &func_at_control_node.def().kind {
                    let mut iter = insts.iter();
                    while let Some((data_inst, next)) =
                        iter.split_first(func_at_control_node.data_insts)
                    {
                        self.visit_data_inst(data_inst);
                        iter = next;
                    }
                }
            }
        }

        let mut visitor = KernelVisitor::new(module, entry_func, &kernel_desc);
        module.funcs[entry_func].inner_visit_with(&mut visitor);
        let safe = visitor.safe;
        let inputs = visitor.inputs;
        Self { name, safe, inputs }
    }
}

struct Kernel {
    sig: KernelSig,
    features: Features,
    spirv: Vec<u32>,
}

impl Kernel {
    fn emit(&self) -> String {
        use proc_macro2::Span;
        use quote::{format_ident, quote};
        use spirv_headers::Capability;
        use syn::LitInt;

        let inputs = self.sig.inputs.iter();
        let safety = quote! {
            krnl::kernel::Safe
        };
        let visit_inputs = self.sig.inputs.iter().map(|x| match x {
            KernelInput::Spec(desc) => {
                let mut ty = desc.scalar_type.type_path().to_token_stream();
                if let Some(len) = desc.array {
                    let len = LitInt::new(&len.to_string(), Span::call_site());
                    ty = quote! {
                        [#ty; #len]
                    }
                }
                let name = &desc.name;
                let id = LitInt::new(&desc.id.to_string(), Span::call_site());
                quote! {
                    v.__visit_spec_id::<#ty>(#name, #id);
                }
            }
            KernelInput::Buffer(desc) | KernelInput::Item(desc) => {
                let mut ty = desc.scalar_type().type_path().to_token_stream();
                if let Some(len) = desc.array() {
                    let len = LitInt::new(&len.to_string(), Span::call_site());
                    ty = quote! {
                        [#ty; #len]
                    }
                }
                let name = desc.name();
                if desc.mutable() {
                    quote! {
                        v.__visit_buffer_mut::<#ty>(#name);
                    }
                } else {
                    quote! {
                        v.__visit_buffer::<#ty>(#name);
                    }
                }
            }
            KernelInput::Push(desc) => {
                let mut ty = desc.scalar_type().type_path().to_token_stream();
                if let Some(len) = desc.array() {
                    let len = LitInt::new(&len.to_string(), Span::call_site());
                    ty = quote! {
                        [#ty; #len]
                    }
                }
                let name = desc.name();
                let offset = LitInt::new(&desc.offset().to_string(), Span::call_site());
                quote! {
                    v.__visit_push::<#ty>(#name, #offset);
                }
            }
        });
        let visit_features: TokenStream = self
            .features
            .capabilities_iter()
            .filter_map(|x| match x {
                Capability::Int8 => Some(quote!(INT8)),
                Capability::Int16 => Some(quote!(INT16)),
                Capability::Int64 => Some(quote!(INT64)),
                Capability::Float16 => Some(quote!(FLOAT16)),
                Capability::Float64 => Some(quote!(FLOAT64)),
                _ => None,
            })
            .map(|x| quote! { v.__visit_features(krnl::context::device::Features::#x); })
            .collect();
        let spirv_lits = self
            .spirv
            .iter()
            .map(|x| LitInt::new(&x.to_string(), Span::call_site()));
        quote! {
            #[allow(non_camel_case_types, dead_code)]
            struct __krnl_Kernel {
                __krnl_safety: krnl::kernel::__private::__Safety<#safety>,
                #(#inputs),*
            }

            impl __krnl_Kernel {
                fn visit<V:  krnl::kernel::__private::__BuildArgsVisitor>(&self, v: &mut V) {
                    v.__visit_spirv([#(#spirv_lits),*].as_slice());
                    #visit_features
                    #(#visit_inputs)*
                }
            }
        }
        .to_string()
    }
}
