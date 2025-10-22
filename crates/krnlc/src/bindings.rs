use crate::reflect::{
    BufferDesc, ElementType, Features, KernelDesc, PushConstantDesc, UsedGlobals, get_scalar_type,
};
use crate::scalar::ScalarType;
use crate::spirv::{
    assemble, constant_name, get_constant_u32, get_element_size, get_name_from_attrs,
    krnl_inst_set, op_constant, op_decorate_block, op_member_decorate_offset, op_member_name,
    op_type_int, op_type_pointer, op_type_struct, pointee_type, strip_krnl_insts,
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
use spirt::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, ControlNodeKind,
    DataInst, DataInstDef, DataInstForm, DataInstFormDef, DataInstKind, DeclDef, ExportKey,
    Exportee, Func, GlobalVar, GlobalVarDecl, GlobalVarDefBody, InternedStr, Module, Type, TypeDef,
    TypeKind, TypeOrConst, Value,
    spv::{
        Imm, Inst, encode_literal_string, extract_literal_string,
        spec::{ExtInstSetDesc, ExtInstSetInstructionDesc, Spec},
    },
    transform::{InnerInPlaceTransform, Transformer},
    visit::{InnerVisit, Visitor},
};
use spirv_headers::{Decoration, ExecutionModel, StorageClass};
use spirv_tools::{
    TargetEnv,
    binary::Binary,
    opt::{Optimizer, Options as OptimizerOptions, Passes},
    val::Validator,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};
use syn::{Ident, LitInt, buffer};

#[derive(Default)]
pub struct BindingsBuilder {
    spirv: Option<Vec<u8>>,
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

fn split_entry_points(
    module: &Module,
    wgsl: bool,
) -> impl Iterator<Item = (Features, Vec<u32>)> + '_ {
    module.exports.iter().map(move |(key, value)| {
        let mut module = module.clone();
        module.exports = std::iter::once((key.clone(), value.clone())).collect();
        let features = Features::reflect(&module).wgsl(wgsl);
        features.write_to_module(&mut module);
        let spirv = assemble(&module).unwrap();
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
