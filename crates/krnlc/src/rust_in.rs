use crate::reflect::{ElementType, Features, UsedGlobals};
use crate::spirv::{
    assemble, get_element_size, krnl_inst_set, op_constant, op_constant_true, op_decorate_block,
    op_member_decorate_offset, op_member_name, op_name, op_spec_constant, op_spec_constant_select,
    op_type_array, op_type_bool, op_type_int, op_type_pointer, op_type_struct, pointee_type,
    runtime_array_element_type, strip_krnl_insts, struct_element_type, validate, variable_name,
};
use camino::{Utf8Path, Utf8PathBuf};
use cargo_gpu::spirv_builder;
use cargo_metadata::Package;
#[cfg(feature = "cli")]
use clap_cargo::{Manifest, Workspace};
use color_print::ceprintln;
use fxhash::FxBuildHasher;
use indexmap::{IndexMap, IndexSet};
use krnl_core::__private::__KrnlInst as KrnlInst;
use smallvec::SmallVec;
use spirt::spv::spec::Opcode;
use spirt::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstForm, DataInstFormDef, DataInstKind, DeclDef, ExportKey, Exportee, Func,
    GlobalVar, GlobalVarDecl, GlobalVarDefBody, InternedStr, Module, Type, Value,
    spv::{Imm, Inst, encode_literal_string, extract_literal_string, spec::Spec},
    transform::{InnerInPlaceTransform, Transformed, Transformer},
    visit::{InnerVisit, Visitor},
};
use spirt::{ControlNodeKind, EntityList};
use spirv_builder::{MetadataPrintout, ShaderPanicStrategy, SpirvMetadata};
use spirv_headers::{Decoration, StorageClass};
use spirv_tools::{
    TargetEnv,
    binary::Binary,
    opt::{Optimizer, Options as OptimizerOptions},
};
use std::{collections::BTreeSet, rc::Rc, time::Instant};

pub struct ModuleBuilder {
    package: Package,
    target_dir: Utf8PathBuf,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        let manifest_dir = Utf8PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let manifest_path = manifest_dir.join("Cargo.toml");
        Self::from_manifest_path(manifest_path, None)
    }
    pub fn from_manifest_path(manifest_path: Utf8PathBuf, target_dir: Option<Utf8PathBuf>) -> Self {
        let metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(manifest_path)
            .exec()
            .unwrap();
        let package = metadata.root_package().cloned().unwrap();
        let target_dir = target_dir.unwrap_or(metadata.target_directory);
        Self::from_package(package, target_dir)
    }
    pub(crate) fn from_package(package: Package, target_dir: Utf8PathBuf) -> Self {
        Self {
            package,
            target_dir,
        }
    }
    pub fn build(self) -> Vec<u32> {
        let start = Instant::now();
        let package_dir = self.package.manifest_path.parent().unwrap();
        let name = &self.package.name;
        ceprintln!("<s><c>krnlc</c></s> <s><g>Compiling</g></s> {name} ({package_dir})");
        let spirv = compile(name, package_dir, &self.target_dir);
        let spirv = process(spirv);
        ceprintln!(
            "<s><c>krnlc</c></s> <s><g>Finished</g></s> {name} ({package_dir}) in {:.2?}s",
            start.elapsed().as_secs_f32()
        );
        spirv
    }
}

#[cfg(feature = "cli")]
pub(crate) fn build_workspace(workspace: &Workspace, manifest: &Manifest) {
    let metadata = manifest.metadata().exec().unwrap();
    let (selected, _) = workspace.partition_packages(&metadata);
    for package in selected {
        let spirv =
            ModuleBuilder::from_package(package.clone(), metadata.target_directory.clone()).build();
        let package_dir = package.manifest_path.parent().unwrap();
        let out_path = package_dir.join("krnl.spv");
        std::fs::write(out_path, bytemuck::cast_slice(&spirv)).unwrap();
    }
}

fn compile(name: &str, path: &Utf8Path, target_dir: &Utf8Path) -> Vec<u8> {
    let install = cargo_gpu::Install::from_shader_crate(path.as_std_path().to_path_buf());
    let installed_backend = install.run().unwrap();
    let mut builder = installed_backend
        .to_spirv_builder(path, "spirv-unknown-vulkan1.2")
        .spirv_metadata(SpirvMetadata::Full)
        .print_metadata(MetadataPrintout::None)
        .shader_panic_strategy(ShaderPanicStrategy::DebugPrintfThenExit {
            print_inputs: true,
            print_backtrace: true,
        })
        .extra_arg("--no-early-report-zombies")
        .extra_arg("--no-infer-storage-classes")
        .extra_arg("--spirt-passes=qptr,reduce,fuse_selects")
        .extra_arg("--no-spirv-opt")
        .extra_arg("--no-spirv-val");
    if target_dir != path.join("target") {
        builder
            .target_dir_path
            .replace(target_dir.join("spirv-builder").into_std_path_buf());
    }
    let capabilites = {
        use spirv_builder::Capability::*;
        [Int8, Int16, Int64, Float16, Float64]
    };
    for cap in capabilites {
        builder = builder.capability(cap);
    }
    let rustgpu_rustflags = std::env::var("RUSTGPU_RUSTFLAGS").ok();
    let mut new_rustgpu_rustflags = format!("--cfg=krnlc --cfg=krnlc_pkg={name:?}");
    if let Some(rustgpu_rustflags) = rustgpu_rustflags.as_ref() {
        new_rustgpu_rustflags.push(' ');
        new_rustgpu_rustflags.push_str(rustgpu_rustflags);
    }
    unsafe {
        std::env::set_var("RUSTGPU_RUSTFLAGS", new_rustgpu_rustflags);
    }
    let result = builder.build();
    if let Some(rustgpu_rustflags) = rustgpu_rustflags {
        unsafe {
            std::env::set_var("RUSTGPU_RUSTFLAGS", rustgpu_rustflags);
        }
    } else {
        unsafe {
            std::env::remove_var("RUSTGPU_RUSTFLAGS");
        }
    }
    let result = result.unwrap();
    let path = result.module.unwrap_single();
    let output = std::fs::read(path).unwrap();
    output
}

fn process(spirv: Vec<u8>) -> Vec<u32> {
    let context = Rc::new(Context::new());
    context.register_custom_ext_inst_set(KrnlInst::SET_NAME, krnl_inst_set());
    let mut module = Module::lower_from_spv_bytes(context.clone(), spirv).unwrap();
    strip_module_source(&mut module);
    let module_globals = UsedGlobals::parse_module(&module, None);
    rename_krnl_vars(&mut module);
    remap_spec_constants(&mut module);
    unify_push_constants(&mut module);
    fix_group_slice_len(&mut module);
    strip_krnl_insts(
        &mut module,
        module_globals.funcs.iter().copied(),
        vec![KrnlInst::GroupSlice],
    );
    strip_op_line(&mut module);
    spirt::passes::legalize::structurize_func_cfgs(&mut module);
    let words = assemble(&module).unwrap();
    //validate(&words).expect("val before rspirv");
    /*
    let words = {
        let mut words = words;
        let module = rspirv::dr::load_words(&words).unwrap();
        let mut builder = rspirv::dr::Builder::new_from_module(module);
        fix_execution_mode(&mut builder);
        {
            use rspirv::binary::Disassemble;
            println!("{}", builder.module_ref().disassemble());
        }
        let words = builder.module().assemble();
        words
    };
    */
    validate(&words).expect("val before opt");
    let binary = spirv_tools::opt::create(Some(TargetEnv::Vulkan_1_3))
        .register_size_passes()
        .optimize(
            &words,
            &mut |_| (),
            Some(OptimizerOptions {
                preserve_bindings: true,
                preserve_spec_constants: true,
                ..OptimizerOptions::default()
            }),
        )
        .unwrap();
    let spirv = if let Binary::OwnedU8(bytes) = binary {
        bytes
    } else {
        binary.as_bytes().to_vec()
    };
    let mut module = Module::lower_from_spv_bytes(context.clone(), spirv).unwrap();
    Features::reflect(&module).write_to_module(&mut module);
    spirt::passes::legalize::structurize_func_cfgs(&mut module);
    let spirv = assemble(&module).unwrap();
    validate(&spirv).expect("val after opt");
    spirv
}

fn strip_module_source(module: &mut Module) {
    let spirt::ModuleDebugInfo::Spv(debug_info) = &mut module.debug_info;
    debug_info.source_languages.clear();
}

fn strip_op_line(module: &mut Module) {
    struct ModuleTransformer {
        cx: Rc<Context>,
    }

    impl Transformer for ModuleTransformer {
        fn transform_attr_set_use(&mut self, attrs: AttrSet) -> Transformed<AttrSet> {
            let cx = self.cx.clone();
            let attrs_def = &cx[attrs];
            let attrs_def = self.transform_attr_set_def(attrs_def);
            match attrs_def {
                Transformed::Changed(attrs_def) => Transformed::Changed(cx.intern(attrs_def)),
                Transformed::Unchanged => Transformed::Unchanged,
            }
        }
        fn transform_attr_set_def(&mut self, attrs_def: &AttrSetDef) -> Transformed<AttrSetDef> {
            if attrs_def
                .attrs
                .iter()
                .any(|x| matches!(x, Attr::SpvDebugLine { .. }))
            {
                let attrs = attrs_def
                    .attrs
                    .iter()
                    .filter(|x| !matches!(x, Attr::SpvDebugLine { .. }))
                    .cloned()
                    .collect();
                Transformed::Changed(AttrSetDef { attrs })
            } else {
                Transformed::Unchanged
            }
        }
    }

    let globals = UsedGlobals::parse_module(module, None);
    let mut transformer = ModuleTransformer { cx: module.cx() };
    for func in globals.funcs {
        module.funcs[func].inner_in_place_transform_with(&mut transformer);
    }
}

fn op_decorate_spec_id(spec_id: u32) -> Attr {
    let opcode = Spec::get().well_known.OpDecorate;
    let operand_kinds = opcode.def().all_operands().map(|(_, x)| x);
    let inst = Inst {
        opcode,
        imms: operand_kinds
            .skip(1)
            .zip([Decoration::SpecId as u32, spec_id])
            .map(|(k, v)| Imm::Short(k, v))
            .collect(),
    };
    Attr::SpvAnnotation(inst)
}

fn remap_spec_constants(module: &mut Module) {
    struct SpecCollector<'a> {
        module: &'a Module,
        krnl_set: InternedStr,
        spec_id: u32,
        funcs: &'a mut IndexMap<Func, bool, FxBuildHasher>,
        vars: &'a mut IndexMap<DataInst, GlobalVar, FxBuildHasher>,
        specs: &'a mut IndexMap<GlobalVar, Const, FxBuildHasher>,
    }

    impl SpecCollector<'_> {
        fn visit_spec(&mut self, gv: GlobalVar) {
            let cx = self.module.cx_ref();
            let gv_decl = &self.module.global_vars[gv];
            let struct_ty = pointee_type(cx, gv_decl.type_of_ptr_to).unwrap();
            let ty = struct_element_type(cx, struct_ty).unwrap();
            let element_ty = ElementType::from_type(cx, ty).unwrap();
            let spec = Spec::get();
            let spec = match element_ty {
                ElementType::Scalar(ty) => {
                    let attrs = {
                        let mut attrs = cx[gv_decl.attrs].attrs.clone();
                        let spec_id = self.spec_id;
                        attrs.insert(op_decorate_spec_id(spec_id));
                        self.spec_id += 1;
                        let attrs = cx.intern(AttrSetDef { attrs });
                        attrs
                    };
                    let size = get_element_size(cx, ty).unwrap();
                    if size == 8 {
                        op_spec_constant(cx, attrs, ty, [0; 2])
                    } else if size > 0 && size <= 4 {
                        op_spec_constant(cx, attrs, ty, [0])
                    } else {
                        unreachable!()
                    }
                }
                ElementType::Array(scalar_ty, len) => {
                    todo!()
                }
            };
            *self.funcs.last_mut().unwrap().1 = true;
            self.specs.insert(gv, spec);
        }
        fn visit_data_inst(&mut self, data_inst: DataInst) {
            let cx = self.module.cx_ref();
            let func = *self.funcs.last().unwrap().0;
            let func_decl = &self.module.funcs[func];
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
                        if let [Value::Const(var), Value::Const(_)] =
                            data_inst_def.inputs.as_slice()
                        {
                            let const_def = &cx[*var];
                            if let ConstKind::PtrToGlobalVar(gv) = &const_def.kind {
                                let gv = *gv;
                                let var_def = &self.module.global_vars[gv];
                                if var_def.addr_space
                                    == AddrSpace::SpvStorageClass(StorageClass::PushConstant as u32)
                                {
                                    self.vars.insert(data_inst, gv);
                                }
                            }
                        }
                    } else if inst.opcode == spec.well_known.OpLoad {
                        if let &[Value::DataInstOutput(access_chain)] =
                            data_inst_def.inputs.as_slice()
                        {
                            if let Some(gv) = self.vars.get(&access_chain).copied() {
                                self.vars.insert(data_inst, gv);
                            }
                        }
                    }
                }
                DataInstKind::SpvExtInst { ext_set, inst } => {
                    if *ext_set == self.krnl_set && *inst == KrnlInst::Spec as u32 {
                        if let &[Value::DataInstOutput(access_chain)] =
                            data_inst_def.inputs.as_slice()
                        {
                            let gv = self.vars[&access_chain];
                            self.visit_spec(gv);
                        }
                    }
                }
                _ => (),
            }
            data_inst_def.inner_visit_with(self);
        }
    }

    impl Visitor<'_> for SpecCollector<'_> {
        fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
        fn visit_type_use(&mut self, _ty: Type) {}
        fn visit_const_use(&mut self, _ct: Const) {}
        fn visit_data_inst_form_use(&mut self, _data_inst_form: DataInstForm) {}
        fn visit_global_var_use(&mut self, _gv: GlobalVar) {}
        fn visit_func_use(&mut self, func: Func) {
            if self.funcs.contains_key(&func) {
                return;
            }
            self.funcs.insert(func, false);
            self.module.funcs[func].inner_visit_with(&mut SpecCollector {
                module: self.module,
                krnl_set: self.krnl_set,
                spec_id: self.spec_id,
                funcs: self.funcs,
                vars: self.vars,
                specs: self.specs,
            });
        }

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

    let krnl_set = module.cx_ref().intern(KrnlInst::SET_NAME);
    let mut funcs = IndexMap::default();
    let mut vars = IndexMap::default();
    let mut specs = IndexMap::default();
    module.inner_visit_with(&mut SpecCollector {
        module,
        krnl_set,
        spec_id: 1,
        funcs: &mut funcs,
        vars: &mut vars,
        specs: &mut specs,
    });

    struct SpecTransformer {
        vars: IndexMap<DataInst, GlobalVar, FxBuildHasher>,
        specs: IndexMap<GlobalVar, Const, FxBuildHasher>,
    }

    impl Transformer for SpecTransformer {
        fn in_place_transform_control_node_def(
            &mut self,
            mut func_at_control_node: spirt::func_at::FuncAtMut<'_, spirt::ControlNode>,
        ) {
            let mut insts2 = EntityList::empty();
            if let ControlNodeKind::Block { insts } =
                &mut func_at_control_node.reborrow().def().kind
            {
                std::mem::swap(insts, &mut insts2);
            }
            if !insts2.is_empty() {
                let mut iter = insts2.iter();
                while let Some((data_inst, next)) =
                    iter.split_first(func_at_control_node.data_insts)
                {
                    if let Some(gv) = self.vars.get(&data_inst) {
                        if self.specs.contains_key(gv) {
                            insts2.remove(data_inst, func_at_control_node.data_insts);
                        }
                    }
                    iter = next;
                }
                if let ControlNodeKind::Block { insts } =
                    &mut func_at_control_node.reborrow().def().kind
                {
                    *insts = insts2;
                }
            }
            func_at_control_node.inner_in_place_transform_with(self);
        }
        fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
            if let Value::DataInstOutput(data_inst) = v {
                if let Some(gv) = self.vars.get(data_inst) {
                    if let Some(spec) = self.specs.get(gv).copied() {
                        return Transformed::Changed(Value::Const(spec));
                    }
                }
            }
            Transformed::Unchanged
        }
    }
    module.exports = std::mem::take(&mut module.exports)
        .into_iter()
        .map(|(mut key, value)| {
            if let ExportKey::SpvEntryPoint {
                imms: _,
                interface_global_vars,
            } = &mut key
            {
                interface_global_vars.retain(|gv| !specs.contains_key(gv));
            }
            (key, value)
        })
        .collect();
    let mut transformer = SpecTransformer { vars, specs };
    for (func, transform) in funcs {
        if transform {
            module.funcs[func].inner_in_place_transform_with(&mut transformer);
        }
    }
}

fn unify_push_constants(module: &mut Module) {
    struct PushCollector<'a> {
        module: &'a Module,
        krnl_set: InternedStr,
        func: Func,
        entry_func: Func,
        funcs: &'a mut IndexSet<Func, FxBuildHasher>,
        vars: &'a mut IndexMap<DataInst, GlobalVar, FxBuildHasher>,
        inputs: &'a mut IndexSet<(Func, GlobalVar), FxBuildHasher>,
    }

    impl PushCollector<'_> {
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
            if let DataInstKind::SpvInst(inst) = &data_inst_form_def.kind
                && inst.opcode == spec.well_known.OpAccessChain
            {
                if let [Value::Const(var), Value::Const(_)] = data_inst_def.inputs.as_slice() {
                    let const_def = &cx[*var];
                    if let ConstKind::PtrToGlobalVar(gv) = &const_def.kind {
                        let gv = *gv;
                        let var_def = &self.module.global_vars[gv];
                        if var_def.addr_space
                            == AddrSpace::SpvStorageClass(StorageClass::PushConstant as u32)
                        {
                            self.vars.insert(data_inst, gv);
                            self.inputs.insert((self.entry_func, gv));
                            self.funcs.insert(self.func);
                        }
                    }
                }
            }
            data_inst_def.inner_visit_with(self);
        }
    }

    impl Visitor<'_> for PushCollector<'_> {
        fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
        fn visit_type_use(&mut self, _ty: Type) {}
        fn visit_const_use(&mut self, _ct: Const) {}
        fn visit_data_inst_form_use(&mut self, _data_inst_form: DataInstForm) {}
        fn visit_global_var_use(&mut self, _gv: GlobalVar) {}
        fn visit_func_use(&mut self, func: Func) {
            self.module.funcs[func].inner_visit_with(&mut PushCollector {
                module: self.module,
                krnl_set: self.krnl_set,
                func,
                entry_func: self.entry_func,
                funcs: self.funcs,
                vars: self.vars,
                inputs: self.inputs,
            });
        }

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

    struct VariableEntry {
        gv: GlobalVar,
        ct: Const,
        members: IndexMap<GlobalVar, Const, FxBuildHasher>,
    }

    impl VariableEntry {
        fn new(module: &mut Module, vars: impl Iterator<Item = GlobalVar>) -> Self {
            let cx = module.cx();
            let mut attrs = BTreeSet::default();
            attrs.insert(op_decorate_block());
            let mut push_types = Vec::new();
            let mut members = IndexMap::default();
            let mut offset = 0;
            let ty_u32 = op_type_int(&cx, 32, false);
            for (member, gv) in vars.enumerate() {
                let member = member as u32;
                let name = variable_name(module, gv).unwrap();
                let gv_decl = &module.global_vars[gv];
                let struct_ty = pointee_type(&cx, gv_decl.type_of_ptr_to).unwrap();
                let ty = struct_element_type(&cx, struct_ty).unwrap();
                let size = get_element_size(&cx, ty).unwrap();
                while offset % size != 0 {
                    offset += 1;
                }
                attrs.extend([
                    op_member_name(member, &name),
                    op_member_decorate_offset(member, offset),
                ]);
                push_types.push(ty);
                members.insert(gv, op_constant(&cx, ty_u32, [member]));
                offset += size;
            }
            let attrs = cx.intern(AttrSetDef { attrs });
            let struct_ty = op_type_struct(&cx, attrs, push_types);
            let ptr_ty = op_type_pointer(&cx, struct_ty, StorageClass::PushConstant);
            let attrs = cx.intern(AttrSetDef {
                attrs: std::iter::once(op_name("krnl::push_constants")).collect(),
            });
            let gv_decl = GlobalVarDecl {
                attrs,
                type_of_ptr_to: ptr_ty,
                shape: None,
                addr_space: AddrSpace::SpvStorageClass(StorageClass::PushConstant as u32),
                def: DeclDef::Present(GlobalVarDefBody { initializer: None }),
            };
            let gv = module.global_vars.define(&cx, gv_decl);
            let const_def = ConstDef {
                attrs: AttrSet::default(),
                kind: ConstKind::PtrToGlobalVar(gv),
                ty: ptr_ty,
            };
            let ct = cx.intern(const_def);
            Self { gv, ct, members }
        }
    }

    let krnl_set = module.cx_ref().intern(KrnlInst::SET_NAME);
    let mut funcs = IndexSet::default();
    let mut vars = IndexMap::default();
    let mut inputs = IndexSet::default();

    let cx = module.cx();

    let entry_funcs: Vec<Func> = module
        .exports
        .values()
        .map(|exportee| {
            if let Exportee::Func(func) = exportee {
                *func
            } else {
                unreachable!()
            }
        })
        .collect();

    let new_vars: IndexMap<Func, VariableEntry, FxBuildHasher> = entry_funcs
        .iter()
        .copied()
        .map(|entry_func| {
            Exportee::Func(entry_func).inner_visit_with(&mut PushCollector {
                module,
                krnl_set,
                func: entry_func,
                entry_func,
                funcs: &mut funcs,
                vars: &mut vars,
                inputs: &mut inputs,
            });

            let vars = inputs
                .iter()
                .filter(|(func, _)| *func == entry_func)
                .map(|(_, gv)| *gv);
            let var = VariableEntry::new(module, vars);
            (entry_func, var)
        })
        .collect();
    struct PushTransformer {
        func: Option<Func>,
        vars: IndexMap<DataInst, GlobalVar, FxBuildHasher>,
        inputs: IndexSet<(Func, GlobalVar), FxBuildHasher>,
        new_vars: IndexMap<Func, VariableEntry, FxBuildHasher>,
    }
    impl Transformer for PushTransformer {
        fn in_place_transform_data_inst_def(
            &mut self,
            func_at_data_inst: spirt::func_at::FuncAtMut<'_, DataInst>,
        ) {
            let data_inst = func_at_data_inst.position;
            if let Some(gv) = self.vars.get(&data_inst).copied() {
                let func = self.func.unwrap();
                if self.inputs.contains(&(func, gv)) {
                    if let Some(var) = self.new_vars.get(&func) {
                        let def = func_at_data_inst.def();
                        def.inputs[0] = Value::Const(var.ct);
                        let member = var.members[&gv];
                        def.inputs[1] = Value::Const(member);
                    }
                }
            }
        }
    }
    module.exports = std::mem::take(&mut module.exports)
        .into_iter()
        .map(|(mut key, value)| {
            let entry_func = if let Exportee::Func(func) = value {
                func
            } else {
                unreachable!()
            };
            if let ExportKey::SpvEntryPoint {
                imms: _,
                interface_global_vars,
            } = &mut key
            {
                interface_global_vars.retain(|gv| !inputs.contains(&(entry_func, *gv)));
                interface_global_vars.push(new_vars[&entry_func].gv);
            }
            (key, value)
        })
        .collect();
    let mut transformer = PushTransformer {
        func: None,
        vars,
        inputs,
        new_vars,
    };
    for func in funcs {
        transformer.func.replace(func);
        module.funcs[func].inner_in_place_transform_with(&mut transformer);
    }
}

/*
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
    dbg!(uint);
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
    dbg!(threads);
    let zero = builder
        .module_ref()
        .types_global_values
        .iter()
        .find_map(|inst| {
            if inst.class.opcode == Op::Constant {
                if inst.result_type == Some(uint) {
                    if inst.operands == [Operand::LiteralBit32(0)] {
                        return inst.result_id;
                    }
                }
            }
            None
        });
    let zero = if let Some(zero) = zero {
        zero
    } else {
        builder.constant_bit32(uint, 0)
    };
    dbg!(zero);
    builder.module_mut().execution_modes.clear();
    let entry_points: Vec<u32> = builder
        .module_ref()
        .entry_points
        .iter()
        .map(|inst| inst.operands[1].unwrap_id_ref())
        .collect();
    dbg!(&entry_points);
    for entry_point in entry_points {
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
        /*
        TODO: https://github.com/gfx-rs/rspirv/pull/263
        builder.execution_mode_id(
            entry_point,
            ExecutionMode::LocalSizeId,
            [threads, zero, zero],
        );
        */
    }
}
*/

fn rename_krnl_vars(module: &mut Module) {
    let globals = UsedGlobals::parse_module(module, None);
    let cx = module.cx();
    for (gv, _) in globals.vars {
        let gv_decl = &mut module.global_vars[gv];
        let mut attrs = cx[gv_decl.attrs].attrs.clone();
        attrs = attrs
            .into_iter()
            .map(|mut attr| {
                if let Attr::SpvAnnotation(inst) = &mut attr {
                    let spec = Spec::get();
                    if inst.opcode == spec.well_known.OpName {
                        let name = extract_literal_string(&inst.imms).unwrap();
                        if let Some(name) = name.strip_prefix("__krnl_") {
                            let name = format!("krnl::{name}");
                            inst.imms = encode_literal_string(&name).collect()
                        }
                    }
                }
                attr
            })
            .collect();
        gv_decl.attrs = cx.intern(AttrSetDef { attrs });
    }
}

fn fix_group_slice_len(module: &mut Module) {
    struct GroupSliceCollector<'a> {
        module: &'a Module,
        krnl_set: InternedStr,
        func: Func,
        buffer_access: &'a mut IndexMap<DataInst, GlobalVar, FxBuildHasher>,
        lens: &'a mut IndexMap<(GlobalVar, Func), Const, FxBuildHasher>,
        array_length: &'a mut IndexMap<DataInst, GlobalVar, FxBuildHasher>,
    }

    impl GroupSliceCollector<'_> {
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
                        if let Some(Value::Const(var)) = data_inst_def.inputs.first().copied() {
                            let const_def = &cx[var];
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
                        }
                    } else if inst.opcode == spec.well_known.OpArrayLength {
                        if let Some(Value::Const(var)) = data_inst_def.inputs.first().copied() {
                            let const_def = &cx[var];
                            if let &ConstKind::PtrToGlobalVar(gv) = &const_def.kind {
                                let var_def = &self.module.global_vars[gv];
                                if var_def.addr_space
                                    == AddrSpace::SpvStorageClass(
                                        StorageClass::StorageBuffer as u32,
                                    )
                                {
                                    self.array_length.insert(data_inst, gv);
                                }
                            }
                        }
                    }
                }
                &DataInstKind::SpvExtInst { ext_set, inst }
                    if ext_set == self.krnl_set && inst == KrnlInst::GroupSlice as u32 =>
                {
                    if let &[Value::DataInstOutput(access), Value::Const(len)] =
                        data_inst_def.inputs.as_slice()
                    {
                        let gv = self.buffer_access[&access];
                        self.lens.insert((gv, self.func), len);
                    } else {
                        unreachable!()
                    }
                }
                _ => (),
            }
        }
    }

    impl Visitor<'_> for GroupSliceCollector<'_> {
        fn visit_attr_set_use(&mut self, _attrs: AttrSet) {}
        fn visit_type_use(&mut self, _ty: Type) {}
        fn visit_const_use(&mut self, _ct: Const) {}
        fn visit_data_inst_form_use(&mut self, _data_inst_form: DataInstForm) {}
        fn visit_global_var_use(&mut self, _gv: GlobalVar) {}
        fn visit_func_use(&mut self, func: Func) {
            self.module.funcs[func].inner_visit_with(self);
        }

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

    let krnl_set = module.cx_ref().intern(KrnlInst::SET_NAME);

    let cx = module.cx();
    let ty_bool = op_type_bool(&cx);
    let ty_u32 = op_type_int(&cx, 32, false);

    let mut buffer_access = IndexMap::default();
    let mut entry_lens = IndexMap::default();
    let mut array_length = IndexMap::default();

    let mut entry_point_selectors = IndexMap::<Func, Const, FxBuildHasher>::default();
    for (i, entry_point) in module.exports.values().copied().enumerate() {
        let func = if let Exportee::Func(func) = entry_point {
            func
        } else {
            unreachable!()
        };
        entry_point.inner_visit_with(&mut GroupSliceCollector {
            module,
            krnl_set,
            func,
            buffer_access: &mut buffer_access,
            lens: &mut entry_lens,
            array_length: &mut array_length,
        });
        let selector = op_spec_constant(
            &cx,
            cx.intern(AttrSetDef {
                attrs: std::iter::once(op_name(&format!("krnl::kernel.{i}"))).collect(),
            }),
            ty_bool,
            [0],
        );
        entry_point_selectors.insert(func, selector);
    }

    // declare spec const for entry point id
    // mask / select len per var
    // swap global var to array with spec / const len

    let one = op_constant(&cx, ty_u32, [1]);
    let mut gv_lens: IndexMap<GlobalVar, Const, FxBuildHasher> =
        entry_lens.keys().map(|x| (x.0, one)).collect();
    let mut gv_types = IndexMap::default();
    for (gv, gv_len) in gv_lens.iter_mut() {
        let gv = *gv;
        let name = variable_name(module, gv).unwrap();
        let var_decl = &mut module.global_vars[gv];
        var_decl.attrs = cx.intern(AttrSetDef {
            attrs: std::iter::once(op_name(&name)).collect(),
        });
        var_decl.addr_space = AddrSpace::SpvStorageClass(StorageClass::Workgroup as u32);
        for ((gv2, func), len2) in entry_lens.iter() {
            if *gv2 != gv {
                continue;
            }
            if *gv_len == one {
                *gv_len = *len2;
            } else {
                // TODO: unsupported by spirt
                todo!();
                *gv_len = op_spec_constant_select(
                    &cx,
                    ty_u32,
                    entry_point_selectors[func],
                    *len2,
                    *gv_len,
                );
            }
        }
        let ptr_ty = {
            let struct_ty = pointee_type(&cx, var_decl.type_of_ptr_to).unwrap();
            let buffer_array_ty = struct_element_type(&cx, struct_ty).unwrap();
            let elem_ty = runtime_array_element_type(&cx, buffer_array_ty).unwrap();
            let array_ty = op_type_array(&cx, AttrSet::default(), elem_ty, *gv_len);
            let struct_ty = op_type_struct(&cx, AttrSet::default(), [array_ty]);
            let ptr_ty = op_type_pointer(&cx, struct_ty, StorageClass::Workgroup);
            ptr_ty
        };
        gv_types.insert(gv, ptr_ty);
        var_decl.type_of_ptr_to = ptr_ty;
    }

    struct GroupSliceTransformer {
        cx: Rc<Context>,
        array_length: IndexMap<DataInst, GlobalVar, FxBuildHasher>,
        gv_lens: IndexMap<GlobalVar, Const, FxBuildHasher>,
        gv_types: IndexMap<GlobalVar, Type, FxBuildHasher>,
    }

    impl Transformer for GroupSliceTransformer {
        fn transform_const_use(&mut self, ct: Const) -> Transformed<Const> {
            let ct_def = &self.cx.clone()[ct];
            match self.transform_const_def(&ct_def) {
                Transformed::Changed(ct_def) => Transformed::Changed(self.cx.intern(ct_def)),
                Transformed::Unchanged => Transformed::Unchanged,
            }
        }
        fn transform_const_def(&mut self, ct_def: &ConstDef) -> Transformed<ConstDef> {
            if let &ConstKind::PtrToGlobalVar(gv) = &ct_def.kind {
                if let Some(ty) = self.gv_types.get(&gv).copied() {
                    return Transformed::Changed(ConstDef {
                        attrs: AttrSet::default(),
                        ty,
                        kind: ConstKind::PtrToGlobalVar(gv),
                    });
                }
            }
            Transformed::Unchanged
        }
        fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
            match *v {
                Value::Const(ct) => {
                    if let Transformed::Changed(ct) = self.transform_const_use(ct) {
                        Transformed::Changed(Value::Const(ct))
                    } else {
                        Transformed::Unchanged
                    }
                }
                Value::DataInstOutput(data_inst) => {
                    if let Some(gv) = self.array_length.get(&data_inst) {
                        if let Some(len) = self.gv_lens.get(gv).copied() {
                            return Transformed::Changed(Value::Const(len));
                        }
                    }
                    Transformed::Unchanged
                }
                _ => Transformed::Unchanged,
            }
        }
        fn in_place_transform_data_inst_def(
            &mut self,
            mut func_at_data_inst: spirt::func_at::FuncAtMut<'_, DataInst>,
        ) {
            let cx = &self.cx;
            let data_inst_def = func_at_data_inst.reborrow().def();
            let data_inst_form_def = &cx[data_inst_def.form];
            if let DataInstKind::SpvInst(inst) = &data_inst_form_def.kind {
                let spec = Spec::get();
                if inst.opcode == spec.well_known.OpAccessChain {
                    let base = data_inst_def.inputs.first().copied().unwrap();
                    if let Value::Const(base) = base {
                        let const_def = &cx[base];
                        if let ConstKind::PtrToGlobalVar(gv) = const_def.kind {
                            if self.gv_types.contains_key(&gv) {
                                let old_ptr_ty = data_inst_form_def.output_type.unwrap();
                                let pointee_ty = pointee_type(cx, old_ptr_ty).unwrap();
                                let ptr_ty =
                                    op_type_pointer(cx, pointee_ty, StorageClass::Workgroup);
                                data_inst_def.form = cx.intern(DataInstFormDef {
                                    kind: data_inst_form_def.kind.clone(),
                                    output_type: Some(ptr_ty),
                                });
                            }
                        }
                    }
                } else if inst.opcode == spec.well_known.OpArrayLength {
                    let base = data_inst_def.inputs.first().copied().unwrap();
                    if let Value::Const(base) = base {
                        let const_def = &cx[base];
                        if let ConstKind::PtrToGlobalVar(gv) = const_def.kind {
                            if let Some(len) = self.gv_lens.get(&gv).copied() {
                                let (opcode, _opname, _opdef) =
                                    Opcode::try_from_u16_with_name_and_def(
                                        spirv_headers::Op::Select as u16,
                                    )
                                    .unwrap();
                                let inst = Inst {
                                    opcode,
                                    imms: SmallVec::default(),
                                };
                                let ty_u32 = op_type_int(cx, 32, false);
                                let form = cx.intern(DataInstFormDef {
                                    kind: DataInstKind::SpvInst(inst),
                                    output_type: Some(ty_u32),
                                });
                                let condition = op_constant_true(&cx);
                                let one = op_constant(cx, ty_u32, [1]);
                                let inputs = [
                                    Value::Const(condition),
                                    Value::Const(len),
                                    Value::Const(one),
                                ]
                                .into_iter()
                                .collect();
                                *data_inst_def = DataInstDef {
                                    attrs: AttrSet::default(),
                                    form,
                                    inputs,
                                };
                            }
                        }
                    }
                }
            }
            for input in data_inst_def.inputs.iter_mut() {
                if let Transformed::Changed(value) = self.transform_value_use(input) {
                    *input = value;
                }
            }
        }
    }

    let mut transformer = GroupSliceTransformer {
        cx,
        array_length,
        gv_lens,
        gv_types,
    };
    for entry_point in module.exports.values().copied() {
        let func = if let Exportee::Func(func) = entry_point {
            func
        } else {
            unreachable!()
        };
        module.funcs[func].inner_in_place_transform_with(&mut transformer);
    }
}

/*
fn type_name(cx: &Context, ty: Type) -> &'static str {
    let ty_def = &cx[ty];
    match &ty_def.kind {
        TypeKind::SpvInst {
            spv_inst,
            type_and_const_inputs,
        } => spv_inst.opcode.name(),
        _ => "",
    }
}
*/
