//use crate::kernel_desc::KernelDesc;
use crate::spirv::{
    assemble, get_element_size, krnl_inst_set, op_constant, op_decorate_block,
    op_member_decorate_offset, op_member_name, op_type_int, op_type_pointer, op_type_struct,
    pointee_type, struct_element_type, validate, variable_name,
};
use camino::{Utf8Path, Utf8PathBuf};
use cargo_metadata::{Metadata, Package};
use clap_cargo::{Manifest, Workspace};
use fxhash::FxBuildHasher;
use indexmap::{
    map::{MutableEntryKey, MutableKeys},
    IndexMap, IndexSet,
};
use krnl_core::__private::__KrnlInst as KrnlInst;
use smallvec::SmallVec;
use spirt::{
    spv::{
        encode_literal_string, extract_literal_string,
        spec::{ExtInstSetDesc, ExtInstSetInstructionDesc, Spec},
        Imm, Inst,
    },
    transform::{InnerInPlaceTransform, Transformer},
    visit::{InnerVisit, Visitor},
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstForm, DataInstFormDef, DataInstKind, DeclDef, ExportKey, Exportee, Func,
    GlobalVar, GlobalVarDecl, GlobalVarDefBody, InternedStr, Module, Type, TypeDef, TypeKind,
    TypeOrConst, Value,
};
use spirv_headers::{Decoration, ExecutionModel, StorageClass};
use spirv_tools::opt::Passes;
use spirv_tools::{
    binary::Binary,
    opt::{Optimizer, Options as OptimizerOptions},
    val::Validator,
    TargetEnv,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

pub fn print_workspace(workspace: &Workspace, manifest: &Manifest, filter: Option<&str>) {
    let metadata = manifest.metadata().exec().unwrap();
    let (selected, _) = workspace.partition_packages(&metadata);
    for package in selected.iter().copied() {
        print_package(package, &metadata, filter);
    }
}

fn print_package(package: &Package, metadata: &Metadata, filter: Option<&str>) {
    let package_dir = package.manifest_path.parent().unwrap();
    let name = &package.name;
    let spirv = std::fs::read(package_dir.join("krnl.spv")).unwrap();
    let output = print_to_string(spirv, filter);
    println!("{output}");
}

pub fn print_to_string(spirv: Vec<u8>, filter: Option<&str>) -> String {
    let context = Rc::new(Context::new());
    context.register_custom_ext_inst_set(KrnlInst::SET_NAME, krnl_inst_set());
    let mut module = Module::lower_from_spv_bytes(context.clone(), spirv).unwrap();
    if let Some(filter) = filter {
        module.exports = module
            .exports
            .into_iter()
            .filter(|(key, _)| {
                if let ExportKey::SpvEntryPoint { imms, .. } = key {
                    let entry_name = extract_literal_string(&imms[1..]).unwrap();
                    entry_name.starts_with(&filter)
                } else {
                    false
                }
            })
            .collect();
    }
    spirt::passes::legalize::structurize_func_cfgs(&mut module);
    let words = assemble(&module).unwrap();
    validate(&words).unwrap();
    let binary = spirv_tools::opt::create(Some(TargetEnv::Vulkan_1_2))
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
    let mut module =
        Module::lower_from_spv_bytes(context.clone(), binary.as_bytes().to_vec()).unwrap();
    spirt::passes::legalize::structurize_func_cfgs(&mut module);
    spirt::print::Plan::for_module(&module)
        .pretty_print()
        .to_string()
}
