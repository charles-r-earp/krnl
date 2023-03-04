use anyhow::{bail, format_err, Result};
use cargo_metadata::{Metadata, Package, PackageId};
use clap::Parser;
use clap_cargo::{Manifest, Workspace};
use spirv_builder::{MetadataPrintout, SpirvBuilder, SpirvMetadata};
use std::{
    collections::HashMap,
    path::PathBuf,
    process::{Command, Stdio},
};
use syn::{visit::Visit, Expr, Item, ItemMod, Lit, Visibility};

#[derive(Parser, Debug)]
struct Cli {
    #[command(flatten)]
    workspace: Workspace,
    #[command(flatten)]
    manifest: Manifest,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let metadata = cli.manifest.metadata().exec()?;
    let (selected, _) = cli.workspace.partition_packages(&metadata);
    for package in selected.iter() {
        let (features, dependencies) = extract_features_dependencies(&metadata, package)?;
        cache(package, &HashMap::new())?;
        let modules = cargo_expand(package, &features)?;
        compile(package, &dependencies, &modules)?;
        todo!();
    }
    Ok(())
}

fn cargo_expand(package: &Package, features: &str) -> Result<HashMap<String, ModuleData>> {
    let mut command = Command::new("cargo");
    command.args([
        "+nightly",
        "rustc",
        "--manifest-path",
        package.manifest_path.as_str(),
    ]);
    if !features.is_empty() {
        command.args(["--features", &features]);
    }
    command
        .args(&["--profile=check", "--", "-Zunpretty=expanded"])
        .stderr(Stdio::inherit());
    let output = command.output()?;
    let expanded = std::str::from_utf8(&output.stdout)?;
    let file: syn::File = syn::parse_str(expanded)?;
    let mut modules = HashMap::new();
    let mut result = Ok(());
    let mut visitor = ModuleVisitor {
        path: String::new(),
        modules: &mut modules,
        result: &mut result,
    };
    visitor.visit_file(&file);
    result?;
    Ok(modules)
}

struct KrnlcMetadata {
    features: String,
    dependencies: String,
}

fn extract_features_dependencies(
    metadata: &Metadata,
    package: &Package,
) -> Result<(String, String)> {
    use std::fmt::Write;

    fn find_krnl_core<'a>(metadata: &'a Metadata, root: &PackageId) -> Option<&'a Package> {
        let node = metadata
            .resolve
            .as_ref()?
            .nodes
            .iter()
            .find(|x| &x.id == root)?;
        let package = metadata.packages.iter().find(|x| x.id == node.id)?;
        if package.name == "krnl-core"
            && package.repository.as_deref() == Some("https://github.com/charles-r-earp/krnl")
        {
            return Some(package);
        }
        for id in node.dependencies.iter() {
            if let Some(package) = find_krnl_core(metadata, id) {
                return Some(package);
            }
        }
        None
    }
    let krnl_core_package = if let Some(package) = find_krnl_core(&metadata, &package.id) {
        package
    } else {
        bail!(
            "krnl-core is not in dependency tree of package {:?}!",
            package.name
        );
    };
    let krnl_core_source = format!(
        " path = {:?}",
        krnl_core_package.manifest_path.parent().unwrap()
    );
    let manifest_path_str = package.manifest_path.as_str();
    let mut features = String::new();
    let mut dependencies = String::new();
    let mut has_krnl_core = false;
    if let Some(krnlc_metadata) = package.metadata.get("krnlc") {
        if let Some(metadata_features) = krnlc_metadata.get("features") {
            if let Some(metadata_features) = metadata_features.as_array() {
                for feature in metadata_features {
                    if let Some(feature) = feature.as_str() {
                        write!(&mut features, "{feature} ").unwrap();
                    } else {
                        bail!("{manifest_path_str:?} [package.metadata.krnlc] features, expected array of strings!");
                    }
                }
            } else {
                bail!("{manifest_path_str:?} [package.metadata.krnlc] features, expected array!");
            }
        }
        if let Some(metadata_dependencies) = krnlc_metadata.get("dependencies") {
            if let Some(metadata_dependencies) = metadata_dependencies.as_object() {
                for (dep, value) in metadata_dependencies.iter() {
                    let (dep_source, dep_default_features, dep_features) = if dep == "krnl-core" {
                        has_krnl_core = true;
                        (krnl_core_source.clone(), true, Vec::new())
                    } else if let Some(dependency) = package
                        .dependencies
                        .iter()
                        .find(|x| x.rename.as_deref().unwrap_or(x.name.as_str()) == dep)
                    {
                        let source = if let Some(path) = dependency.path.as_ref() {
                            let path = path.canonicalize()?;
                            format!("path = {path:?}")
                        } else if let Some(source) = dependency.source.as_ref() {
                            if source == "registry+https://github.com/rust-lang/crates.io-index" {
                                format!("version = \"{}\"", dependency.req)
                            } else if let Some((key, value)) = source.split_once("+") {
                                format!("{key} = {value:?}")
                            } else {
                                bail!("Unsupported source {source:?} for dependency {dep:?}!");
                            }
                        } else {
                            bail!("Source not found for dependency {dep:?}!");
                        };
                        (
                            source,
                            dependency.uses_default_features,
                            dependency.features.clone(),
                        )
                    } else {
                        bail!("{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?} is not a dependency of {:?}!", package.name);
                    };
                    let mut default_features = None;
                    let mut features = Vec::new();
                    if let Some(table) = value.as_object() {
                        dependencies.push_str("{ ");
                        for (key, value) in table.iter() {
                            match key.as_str() {
                                "default-features" => {
                                    if let Some(value) = value.as_bool() {
                                        default_features.replace(value);
                                    }
                                }
                                "features" => {
                                    if let Some(value) = value.as_array() {
                                        for value in value.iter() {
                                            if let Some(value) = value.as_str() {
                                                features.push(value.to_string());
                                            } else {
                                                bail!(
                                                        "{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?} features, expected array of strings!"
                                                    );
                                            }
                                        }
                                    } else {
                                        bail!(
                                                "{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?} features, expected array!"
                                            );
                                    }
                                }
                                _ => {
                                    bail!(
                                            "{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?}, unexpected key {key:?}!"
                                        );
                                }
                            }
                        }
                    } else {
                        bail!(
                                "{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?}, expected table!"
                            );
                    }
                    let default_features = default_features.unwrap_or(dep_default_features);
                    if features.is_empty() {
                        features = dep_features;
                    }
                    let features = itertools::join(features, ", ");
                    writeln!(&mut dependencies, "{dep:?} = {{ {dep_source}, features = [{features:?}], default-features = {default_features} }}").unwrap();
                }
            } else {
                bail!(
                    "{manifest_path_str:?} [package.metadata.krnlc.dependencies], expected table!"
                );
            }
        }
    }
    if !has_krnl_core {
        writeln!(
            &mut dependencies,
            "\"krnl-core\" = {{ {krnl_core_source} }}"
        )
        .unwrap();
    }
    Ok((features, dependencies))
}

#[derive(Debug)]
struct ModuleData {
    path: String,
    source: String,
    name_with_hash: String,
}

impl ModuleData {
    fn new(path: String, source: String) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::default();
        source.hash(&mut hasher);
        let hash = hasher.finish();
        let name = if let Some((_, last)) = path.rsplit_once("::") {
            last
        } else {
            path.as_str()
        };
        let name_with_hash = format!("{name}_{hash:x}");
        Self {
            path,
            source,
            name_with_hash,
        }
    }
}

struct ModuleVisitor<'a> {
    path: String,
    modules: &'a mut HashMap<String, ModuleData>,
    result: &'a mut Result<()>,
}

impl<'a, 'ast> Visit<'ast> for ModuleVisitor<'a> {
    fn visit_item_mod(&mut self, i: &'ast ItemMod) {
        if self.result.is_err() {
            return;
        }
        if !self.path.is_empty() {
            if i.ident == "__krnl_module_data" && i.vis == Visibility::Inherited {
                if let Some((_, items)) = i.content.as_ref() {
                    if let [Item::Const(item_const)] = items.as_slice() {
                        if item_const.ident == "__krnl_module_source" {
                            if let Expr::Lit(expr_lit) = item_const.expr.as_ref() {
                                if let Lit::Str(lit_str) = &expr_lit.lit {
                                    let module =
                                        ModuleData::new(self.path.clone(), lit_str.value());
                                    if let Some(other) = self.modules.get(&module.name_with_hash) {
                                        *self.result = Err(format_err!("Modules are not unique, collision between {} and {}! Try renaming a module.", other.path, module.path));
                                    } else {
                                        self.modules.insert(module.name_with_hash.clone(), module);
                                    }
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
        let path = if self.path.is_empty() {
            i.ident.to_string()
        } else {
            format!("{}::{}", self.path, i.ident)
        };
        let mut visitor = ModuleVisitor {
            path,
            modules: self.modules,
            result: self.result,
        };
        syn::visit::visit_item_mod(&mut visitor, i);
    }
}

fn cache(package: &Package, modules: &HashMap<String, ModuleData>) -> Result<()> {
    use std::fmt::Write;
    let mut cache = r#"macro_rules! __krnl_module {
"#
    .to_string();
    for module in modules.values() {
        writeln!(&mut cache, "({name}) => ();", name = module.name_with_hash).unwrap();
    }
    cache.push_str("($i:ident) => {\n");
    if !modules.is_empty() {
        cache.push_str("compile_error!(\"recompile with krnlc\")\n");
    };
    cache.push_str("};\n}");
    let file = syn::parse_str(&cache)?;
    let mut cache = prettyplease::unparse(&file);
    cache.insert_str(0, "/* Generated by krnlc */\n");
    std::fs::write(
        package
            .manifest_path
            .parent()
            .unwrap()
            .join("krnl-cache.rs"),
        cache,
    )?;
    Ok(())
}

fn compile(
    package: &Package,
    dependencies: &str,
    modules: &HashMap<String, ModuleData>,
) -> Result<()> {
    use std::fmt::Write;
    let manifest_dir = package.manifest_path.parent().unwrap().as_std_path();
    let target_krnl_dir = manifest_dir.join("target/krnl");
    std::fs::create_dir_all(&target_krnl_dir)?;
    {
        // lib
        let lib_dir = target_krnl_dir.join("lib");
        if !lib_dir.exists() {
            std::fs::create_dir(&lib_dir)?;
        }
        for lib in [
            env!("KRNLC_LIBLLVM"),
            env!("KRNLC_LIBRUSTC_DRIVER"),
            env!("KRNLC_LIBSTD"),
        ] {
            let link = lib_dir.join(lib);
            if !link.exists() {
                symlink::symlink_file(
                    &PathBuf::from(env!("KRNLC_TOOLCHAIN_LIB")).join(lib),
                    &link,
                )?;
            }
        }
        let librustc_codegen_spirv = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/../../../librustc_codegen_spirv.so"
        ));
        std::fs::write(
            lib_dir.join("librustc_codegen_spirv.so"),
            librustc_codegen_spirv.as_ref(),
        )?;
        // https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-builder/src/lib.rs
        fn dylib_path_envvar() -> &'static str {
            if cfg!(windows) {
                "PATH"
            } else if cfg!(target_os = "macos") {
                "DYLD_FALLBACK_LIBRARY_PATH"
            } else {
                "LD_LIBRARY_PATH"
            }
        }
        let lib_dir = lib_dir.canonicalize()?;
        let path_var = dylib_path_envvar();
        let path = if let Ok(path) = std::env::var(path_var) {
            std::env::join_paths(std::iter::once(lib_dir).chain(std::env::split_paths(&path)))?
        } else {
            lib_dir.into_os_string()
        };
        std::env::set_var(path_var, path);
    }
    let crate_name = package.name.as_str();
    let device_crate_dir = target_krnl_dir.join(&crate_name);
    {
        // device crate
        std::fs::create_dir_all(&device_crate_dir)?;
        let manifest = format!(
            r#"# generated by krnlc 
[package]
name = {crate_name:?}
version = "0.1.0"
edition = "2021"
publish = false

[workspace]

[lib]
crate-type = ["dylib"]

[dependencies]
{dependencies}
"#
        );
        std::fs::write(device_crate_dir.join("Cargo.toml"), manifest.as_bytes())?;
        let toolchain = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/rust-toolchain.toml"));
        std::fs::write(
            device_crate_dir.join("rust-toolchain.toml"),
            toolchain.as_bytes(),
        )?;
        let src_dir = device_crate_dir.join("src");
        if !src_dir.exists() {
            std::fs::create_dir(&src_dir)?;
        }
        let mut source = r#"#![cfg_attr(target_arch = "spirv",
no_std,
feature(asm_experimental_arch),
)]

extern crate krnl_core;
"#
        .to_string();
        for (name, module) in modules.iter() {
            writeln!(&mut source, "pub mod {name} {{ {} }}", module.source).unwrap();
        }
        let file = syn::parse_str(&source)?;
        let source = prettyplease::unparse(&file);
        std::fs::write(src_dir.join("lib.rs"), source.as_bytes())?;
    }
    {
        // update
        let status = Command::new("cargo")
            .args(&[
                "update",
                "--manifest-path",
                device_crate_dir
                    .join("Cargo.toml")
                    .to_string_lossy()
                    .as_ref(),
            ])
            .status()?;
        if !status.success() {
            bail!("cargo update failed!");
        }
    }
    {
        // run spirv-builder
        let capabilites = {
            use spirv_builder::Capability::*;
            [Int8, Int16, Int64, Float16, Float64]
        };
        let mut builder = SpirvBuilder::new(&device_crate_dir, "spirv-unknown-vulkan1.2")
            .multimodule(true)
            .spirv_metadata(SpirvMetadata::NameVariables)
            .print_metadata(MetadataPrintout::None)
            .preserve_bindings(true)
            .deny_warnings(true)
            .extension("SPV_KHR_non_semantic_info");
        for cap in capabilites {
            builder = builder.capability(cap);
        }
        let output = builder.build()?;
        let spirv_modules = output.module.unwrap_multi();
        for (entry_point, spirv_path) in spirv_modules.iter() {
            let spirv = std::fs::read(spirv_path)?;
            use rspirv::binary::Disassemble;
            let module = rspirv::dr::load_bytes(&spirv).unwrap();
            println!("{}", module.disassemble());
        }
    }
    todo!()
}
