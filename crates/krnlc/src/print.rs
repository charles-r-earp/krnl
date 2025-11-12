use crate::{
    reflect::Features,
    spirv::{assemble, krnl_inst_set, validate},
};
use cargo_metadata::Package;
use clap_cargo::{Manifest, Workspace};
use krnl_core::__private::__KrnlInst as KrnlInst;
use spirt::{Context, ExportKey, Module, spv::extract_literal_string};
use spirv_tools::{
    TargetEnv,
    opt::{Optimizer, Options as OptimizerOptions},
};
use std::rc::Rc;

pub fn print_workspace(workspace: &Workspace, manifest: &Manifest, filter: Option<&str>) {
    let metadata = manifest.metadata().exec().unwrap();
    let (selected, _) = workspace.partition_packages(&metadata);
    for package in selected.iter().copied() {
        print_package(package, filter);
    }
}

fn print_package(package: &Package, filter: Option<&str>) {
    let package_dir = package.manifest_path.parent().unwrap();
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
        Features::reflect(&module).write_to_module(&mut module);
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
