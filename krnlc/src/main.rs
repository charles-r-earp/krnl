#![allow(warnings)]
use anyhow::{bail, format_err, Result};
use cargo_metadata::{Metadata, MetadataCommand, Package, PackageId};
use clap_cargo::{Features, Manifest, Workspace};
use proc_macro2::{Span as Span2, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use rspirv::spirv::Capability;
use spirv_builder::{MetadataPrintout, ModuleResult, SpirvBuilder, SpirvMetadata};
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str::FromStr,
};
use syn::{visit::Visit, Expr, File, ItemConst, ItemMod, Lit, LitInt};
mod metadata {
    include!("../../src/kernel/__private/metadata.rs");
}
use metadata::ModuleData as ModuleData2;

#[derive(clap::Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Subcommand, Debug)]
enum Cmd {
    /// collects and compiles all modules
    ///
    /// - writes to [<manifest-dir>]/.krnl/[<package>]/cache
    Build {
        #[command(flatten)]
        workspace: Workspace,
        #[command(flatten)]
        features: Features,
        #[command(flatten)]
        manifest: Manifest,
    },
    /// removes files created by krnlc
    ///
    ///    - [<manifest-dir>]/.krnl/[<package>]
    ///
    ///    - [<manifest-dir>]/.krnl if all packages are removed
    Clean {
        #[command(flatten)]
        workspace: Workspace,
        #[command(flatten)]
        manifest: Manifest,
    },
    /// collects and compiles all modules to [<manifest-dir>]/krnl.cache
    Build2 {
        #[command(flatten)]
        workspace: Workspace,
        #[command(flatten)]
        manifest: Manifest,
    },
}

fn main() -> Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Build {
            workspace,
            features,
            manifest,
        } => build(workspace, features, manifest),
        Cmd::Clean {
            workspace,
            manifest,
        } => clean(workspace, manifest),
        Cmd::Build2 {
            workspace,
            manifest,
        } => build2(workspace, manifest),
    }
}

fn build(workspace: Workspace, features: Features, manifest: Manifest) -> Result<()> {
    let metadata = manifest.metadata().exec()?;
    let (selected, _) = workspace.partition_packages(&metadata);
    for package in selected {
        let manifest_path = package.manifest_path.as_str();
        let krnl_dir = package
            .manifest_path
            .as_std_path()
            .parent()
            .unwrap()
            .join(".krnl");
        init_krnl_dir(&krnl_dir)?;
        let target_dir = krnl_dir.join("target");
        //cargo_check(&package.name, &features, &target_dir, manifest_path)?;
        cache(&krnl_dir, &package.name, &[], &[])?;
        let module_datas = cargo_expand(&package.name, &features, &target_dir, manifest_path)?;
        fs::write(
            krnl_dir.join(".gitignore"),
            r#"lib
target
modules"#
                .as_bytes(),
        )?;
        let modules_dir = krnl_dir.join("modules");
        if !modules_dir.exists() {
            fs::create_dir(&modules_dir)?;
        }
        let mut kernels = Vec::with_capacity(module_datas.len());
        for module_data in module_datas.iter() {
            kernels.push(compile(&modules_dir, module_data)?);
        }
        cache(&krnl_dir, &package.name, &module_datas, &kernels)?;
    }
    Ok(())
}

fn clean(workspace: Workspace, manifest: Manifest) -> Result<()> {
    let metadata = manifest.metadata().exec()?;
    let (selected, _) = workspace.partition_packages(&metadata);
    for package in selected {
        let krnl_dir = package
            .manifest_path
            .as_std_path()
            .parent()
            .unwrap()
            .join(".krnl");
        if krnl_dir.exists() {
            check_krnl_dir_cashdir_tag(&krnl_dir, false)?;
            fs::remove_dir_all(&krnl_dir)?;
        }
    }
    Ok(())
}

fn build2(workspace: Workspace, manifest: Manifest) -> Result<()> {
    let metadata = manifest.metadata().exec()?;
    let (selected, _) = workspace.partition_packages(&metadata);
    let target_dir = metadata.target_directory.as_std_path();
    let krnlc_dir = target_dir.join("krnlc");
    let lib_dir = krnlc_dir.join("lib");
    fs::create_dir_all(&lib_dir)?;
    let packages_dir = krnlc_dir.join("packages");
    for package in selected {
        let package_dir = packages_dir.join(&package.name);
        fs::create_dir_all(&package_dir)?;
        let (features, dependencies) = extract_features_dependencies(&metadata, package)?;
        cache2(&package, &[], &[])?;
        let module_datas = cargo_expand2(package, &features)?;
        //let mut kernels = Vec::with_capacity(module_datas.len());
        /*for module_data in module_datas.iter() {
            kernels.push(compile(&modules_dir, module_data)?);
        }*/
        //cache2(&package, &module_datas, &kernels)?;
        todo!();
    }
    Ok(())
}

fn add_features_to_command(command: &mut Command, features: &Features) -> Result<()> {
    if features.all_features {
        command.arg("--all-features");
    }
    if features.no_default_features {
        command.arg("--no-default-features");
    }
    match features.features.as_slice() {
        [] => (),
        [feature] => {
            command.args(["--features", feature]);
        }
        [feature, ..] => {
            command.arg("--features");
            let mut features_string = format!("\"{feature}");
            use std::fmt::Write;
            for feature in features.features.iter().skip(1) {
                write!(&mut features_string, " {feature}")?;
            }
            features_string.push('\"');
            command.arg(&features_string);
        }
    }
    Ok(())
}

fn cargo_check<'a>(
    crate_name: &str,
    features: &Features,
    target_dir: &Path,
    manifest_path: &str,
) -> Result<()> {
    let mut command = Command::new("cargo");
    command.args(["+nightly", "check", "--manifest-path", manifest_path]);
    add_features_to_command(&mut command, features)?;
    command.args(&[
        "--target-dir",
        target_dir.to_string_lossy().as_ref(),
        "-p",
        "krnl-core",
    ]);
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(format_err!("cargo check failed!"))
    }
}

fn cargo_expand(
    crate_name: &str,
    features: &Features,
    target_dir: &Path,
    manifest_path: &str,
) -> Result<Vec<ModuleData>> {
    let mut command = Command::new("cargo");
    command.args(["+nightly", "rustc", "--manifest-path", manifest_path]);
    add_features_to_command(&mut command, features)?;
    command.args(&["--target-dir", target_dir.to_string_lossy().as_ref()]);
    command
        .args(&["--profile=check", "--", "-Zunpretty=expanded"])
        .stderr(Stdio::inherit());
    let output = command.output()?;
    let expanded = std::str::from_utf8(&output.stdout)?;
    let file = syn::parse_str(expanded)?;
    let mut modules = Vec::new();
    let mut visitor = Visitor {
        path: crate_name.replace('-', "_"),
        modules: &mut modules,
    };
    visitor.visit_file(&file);
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
    dbg!(krnl_core_package);
    todo!();
    /*let krnl_core_source = if let Some(source) = krnl_core_package.source.as_ref() {
        todo!()
    } else {
        let path = krnl_core_package.manifest_path.parent().unwrap().as_str();
        format!("path = {path:?}")
    };

    let krnl_core_source = if let Some(path) = krnl_core_package.path.as_ref() {
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
    */
    let krnl_core_source = String::new();
    let manifest_dir = package.manifest_path.as_std_path().parent().unwrap();
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
                    writeln!(&mut dependencies, "{dep:?} = {{ {dep_source}, features = [{features:?}], default-features = {default_features} }}");
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
        );
    }
    Ok((features, dependencies))
}

fn cargo_expand2(package: &Package, features: &str) -> Result<Vec<ModuleDataEntry>> {
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
    let mut modules = Vec::new();
    let mut visitor = Visitor2 {
        path: package.name.replace('-', "_"),
        modules: &mut modules,
    };
    visitor.visit_file(&file);
    Ok(modules)
}

struct ModuleDataEntry {
    path: String,
    data: ModuleData2,
}

struct Visitor2<'a> {
    path: String,
    modules: &'a mut Vec<ModuleDataEntry>,
}

impl<'a, 'ast> Visit<'ast> for Visitor2<'a> {
    fn visit_item_mod(&mut self, i: &'ast ItemMod) {
        let mut visitor = Visitor2 {
            path: format!("{}::{}", self.path, i.ident),
            modules: &mut self.modules,
        };
        syn::visit::visit_item_mod(&mut visitor, i);
    }
    fn visit_item_const(&mut self, i: &'ast ItemConst) {
        if let Some(path) = self.path.strip_suffix("::module") {
            if i.ident == "krnlc__krnl_module_data" {
                if let Expr::Array(expr_array) = &*i.expr {
                    let mut bytes = Vec::<u8>::with_capacity(expr_array.elems.len());
                    for elem in expr_array.elems.iter() {
                        if let Expr::Lit(expr_lit) = elem {
                            if let Lit::Int(lit_int) = &expr_lit.lit {
                                if let Ok(val) = lit_int.base10_parse() {
                                    bytes.push(val);
                                } else {
                                    return;
                                }
                            }
                        } else {
                            return;
                        }
                    }
                    if let Ok(data) = bincode::deserialize::<ModuleData2>(&bytes) {
                        self.modules.push(ModuleDataEntry {
                            path: path.to_string(),
                            data,
                        });
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct ModuleData {
    path: String,
    data: HashMap<String, String>,
}

struct Visitor<'a> {
    path: String,
    modules: &'a mut Vec<ModuleData>,
}

impl<'a, 'ast> Visit<'ast> for Visitor<'a> {
    fn visit_item_mod(&mut self, i: &'ast ItemMod) {
        let mut visitor = Visitor {
            path: format!("{}::{}", self.path, i.ident),
            modules: &mut self.modules,
        };
        syn::visit::visit_item_mod(&mut visitor, i);
    }
    fn visit_item_const(&mut self, i: &'ast ItemConst) {
        if let Some(path) = self.path.strip_suffix("::module") {
            if i.ident == "krnlc__krnl_module_data" {
                if let Expr::Array(expr_array) = &*i.expr {
                    let mut bytes = Vec::<u8>::with_capacity(expr_array.elems.len());
                    for elem in expr_array.elems.iter() {
                        if let Expr::Lit(expr_lit) = elem {
                            if let Lit::Int(lit_int) = &expr_lit.lit {
                                if let Ok(val) = lit_int.base10_parse() {
                                    bytes.push(val);
                                } else {
                                    return;
                                }
                            }
                        } else {
                            return;
                        }
                    }
                    if let Ok(data) = bincode::deserialize::<HashMap<String, String>>(&bytes) {
                        if data.contains_key("krnl_module_tokens") {
                            let data = ModuleData {
                                path: path.to_string(),
                                data,
                            };
                            self.modules.push(data);
                        }
                    }
                }
            }
        }
    }
}

fn check_krnl_dir_cashdir_tag(krnl_dir: &Path, create: bool) -> Result<()> {
    let cachedir_tag_path = krnl_dir.join("CACHEDIR.TAG");
    let cachedir_tag = concat!(
        "Signature: 8a477f597d28d172789f06886806bc55",
        "\n# This file is a cache directory tag created by krnlc.",
        "\n# For information about cache directory tags see https://bford.info/cachedir/"
    );
    let exists = cachedir_tag_path.exists();
    if create && !exists {
        fs::write(&cachedir_tag_path, cachedir_tag.as_bytes())?;
    } else if exists {
        let tag = fs::read_to_string(&cachedir_tag_path)?;
        if tag != cachedir_tag {
            let path = cachedir_tag_path.to_string_lossy();
            bail!("A CACHEDIR.TAG already exists at {path:?} for another app!");
        }
    }
    Ok(())
}

fn init_krnl_dir(krnl_dir: &Path) -> Result<()> {
    if !krnl_dir.exists() {
        fs::create_dir(&krnl_dir)?;
    }
    check_krnl_dir_cashdir_tag(krnl_dir, true)?;
    fs::write(
        krnl_dir.join(".gitignore"),
        "lib/\nmodules/\ntarget/".as_bytes(),
    )?;
    let lib_dir = krnl_dir.join("lib");
    if !lib_dir.exists() {
        fs::create_dir(&lib_dir)?;
    }
    for lib in [
        env!("KRNLC_LIBLLVM"),
        env!("KRNLC_LIBRUSTC_DRIVER"),
        env!("KRNLC_LIBSTD"),
    ] {
        let link = lib_dir.join(lib);
        if !link.exists() {
            symlink::symlink_file(&PathBuf::from(env!("KRNLC_TOOLCHAIN_LIB")).join(lib), &link)?;
        }
    }
    let librustc_codegen_spirv = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/../../../librustc_codegen_spirv.so"
    ));
    fs::write(
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
    Ok(())
}

fn compile(
    modules_dir: &Path,
    module_data: &ModuleData, /*target_dir: &str*/
) -> Result<BTreeMap<String, PathBuf>> {
    let crate_name = module_data.path.replace("::", "_");
    let crate_dir = modules_dir.join(&crate_name);
    if !crate_dir.exists() {
        fs::create_dir(&crate_dir)?;
    }
    let dependencies = module_data.data.get("dependencies").unwrap();
    let mut manifest = format!(
        r#"[package]
name = {crate_name:?}
version = "0.1.0"
edition = "2021"
publish = false

[workspace]

[lib]
crate-type = ["dylib"]

[dependencies]
{dependencies}

[patch.crates-io]
libm = {{ git = "https://github.com/rust-lang/libm", tag = "0.2.5" }}
"#
    );
    fs::write(crate_dir.join("Cargo.toml"), manifest.as_bytes())?;
    let toolchain = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/rust-toolchain.toml"));
    fs::write(crate_dir.join("rust-toolchain.toml"), toolchain.as_bytes())?;
    let src_dir = crate_dir.join("src");
    if !src_dir.exists() {
        fs::create_dir(&src_dir)?;
    }
    let tokens = module_data.data.get("krnl_module_tokens").unwrap();
    let file = syn::parse_str(tokens)?;
    let src = prettyplease::unparse(&file);
    let src = format!(
        r#"#![cfg_attr(target_arch = "spirv",
no_std,
feature(asm_experimental_arch),
)]
{src}
extern crate krnl_core; "#
    );
    fs::write(src_dir.join("lib.rs"), src.as_bytes())?;
    let status = Command::new("cargo")
        .args(&[
            "update",
            "--manifest-path",
            crate_dir.join("Cargo.toml").to_string_lossy().as_ref(),
        ])
        .status()?;
    if !status.success() {
        bail!("cargo update failed!");
    }
    let capabilites = {
        use spirv_builder::Capability::*;
        [Int8, Int16, Int64, Float16, Float64]
    };
    //let krnl_dir = modules_dir.parent().unwrap();
    /*let config = format!(r#"[build]
    rustflags = ["--manifest-path {crate_name}/Cargo.toml", "--target-dir my-target"]

    [target.'cfg(target_arch = "spirv")']
    rustflags = ["--target-dir spirv-target"]
        "#);
        fs::write(
            krnl_dir.join(".cargo").join("config.toml"),
            config.as_bytes()
        )?;*/

    let mut builder = SpirvBuilder::new(crate_dir, "spirv-unknown-vulkan1.2")
        .multimodule(true)
        .spirv_metadata(SpirvMetadata::NameVariables)
        .print_metadata(MetadataPrintout::None)
        .preserve_bindings(true)
        .deny_warnings(true)
        .extension("SPV_KHR_non_semantic_info");
    for cap in capabilites {
        builder = builder.capability(cap);
    }
    let module = builder.build()?.module;
    if let ModuleResult::MultiModule(map) = module {
        Ok(map)
    } else {
        Err(format_err!("Expected multimodule!"))
    }
}

fn cache(
    krnl_dir: &Path,
    package_name: &str,
    module_datas: &[ModuleData],
    kernels: &[BTreeMap<String, PathBuf>],
) -> Result<()> {
    fn create_lit_int<T: std::fmt::Display>(x: T) -> LitInt {
        LitInt::new(&x.to_string(), Span2::call_site())
    }
    let mut module_arms = Vec::with_capacity(module_datas.len());
    let mut kernel_arms = Vec::with_capacity(kernels.iter().map(|x| x.len()).sum());
    for (module_data, kernels) in module_datas.iter().zip(kernels) {
        let module_path = &module_data.path;
        let dependencies = module_data.data.get("dependencies").unwrap();
        let module_tokens = module_data.data.get("krnl_module_tokens").unwrap();
        let module_src = format!("(dependencies({dependencies:?})) => ({module_tokens})");
        let module_path_indices = (0..module_path.len())
            .into_iter()
            .map(create_lit_int)
            .collect::<Vec<_>>();
        module_arms.push(quote! {
            {
                let module_path = #module_path.as_bytes();
                if path.len() == module_path.len() + "::module".len() {
                    let success = #(path[#module_path_indices] == module_path[#module_path_indices])&&*;
                    if success {
                        return Some(#module_src);
                    }
                }
            }
        });
        for (entry_point, spirv_path) in kernels.iter() {
            let spirv_bytes = fs::read(spirv_path)?;
            let (kernel, kernel_hash, spirv_words, capabilities) =
                process_kernel(module_path, entry_point, &spirv_bytes)?;
            let spirv_words = spirv_words
                .into_iter()
                .map(create_lit_int)
                .collect::<Vec<_>>();
            let kernel_path = format!("{module_path}::{kernel}");
            let kernel_path_indices = (0..kernel_path.len())
                .into_iter()
                .map(create_lit_int)
                .collect::<Vec<_>>();
            let mut features = TokenStream2::new();
            {
                use Capability::*;
                for (feature, cap) in [
                    ("shader_int8", Int8),
                    ("shader_int16", Int16),
                    ("shader_int64", Int64),
                    ("shader_float16", Float16),
                    ("shader_float64", Float64),
                ] {
                    let ident = format_ident!("with_{feature}");
                    if capabilities.contains(&cap) {
                        features.extend(quote! {
                            .#ident(true)
                        });
                    }
                }
            }
            kernel_arms.push(quote! {
                {
                    let kernel_path = #kernel_path.as_bytes();
                    let kernel_spirv = &[#(#spirv_words),*];
                    let features = Features::new()
                        #features;
                    if path.len() == kernel_path.len() {
                        let success = #(path[#kernel_path_indices] == kernel_path[#kernel_path_indices])&&*;
                        if success {
                            return Some((#kernel_hash, kernel_spirv, features));
                        }
                    }
                }
            });
        }
    }
    let cache = quote! {
        /* generated by krnlc */

        const fn __module(path: &'static str) -> Option<&'static str> {
            #[allow(unused_variables)]
            let path = path.as_bytes();
            #(#module_arms)*
            None
        }
        pub(super) const fn __kernel(path: &'static str) -> Option<(u64, &'static [u32], Features)> {
            #[allow(unused_variables)]
            let path = path.as_bytes();
            #(#kernel_arms)*
            None
        }
    };
    fn tokens_to_string(tokens: TokenStream2) -> String {
        use proc_macro2::TokenTree as TokenTree2;
        use std::fmt::Write;
        let mut string = String::new();
        let mut iter = tokens.into_iter().peekable();
        while let Some(token) = iter.next() {
            use TokenTree2::*;
            match token {
                Group(group) => {
                    use proc_macro2::Delimiter;
                    let (pfx, sfx) = match group.delimiter() {
                        Delimiter::Parenthesis => (Some('('), Some(')')),
                        Delimiter::Brace => (Some('{'), Some('}')),
                        Delimiter::Bracket => (Some('['), Some(']')),
                        Delimiter::None => (None, None),
                    };
                    if let Some(pfx) = pfx {
                        string.push(pfx);
                    }
                    string.push_str(&tokens_to_string(group.stream()));
                    if let Some(sfx) = sfx {
                        string.push(sfx);
                    }
                }
                Ident(ident) => {
                    string.push_str(&ident.to_string());
                    if let Some(next) = iter.peek() {
                        if let TokenTree2::Ident(_) = next {
                            string.push(' ');
                        }
                    }
                }
                Punct(punct) => {
                    string.push(punct.as_char());
                }
                Literal(literal) => {
                    string.push_str(&literal.to_string());
                }
            }
        }
        string
    }
    let cache = tokens_to_string(cache);
    fs::write(krnl_dir.join("cache"), cache.as_bytes())?;
    Ok(())
}

fn cache2(
    package: &Package,
    module_datas: &[ModuleData],
    kernels: &[BTreeMap<String, PathBuf>],
) -> Result<()> {
    let manifest_dir = package.manifest_path.as_std_path().parent().unwrap();
    let header = r#"/* generated by krnlc */
/* For more information see https://github.com/charles-r-earp/krnl */
"#;
    let krnlc_cache_path = manifest_dir.join("krnlc.cache");
    if krnlc_cache_path.exists() {
        let krnlc_cache = fs::read_to_string(&krnlc_cache_path)?;
        if !krnlc_cache.starts_with(&header) {
            bail!("{krnlc_cache_path:?} not recognized as being generated by krnlc!");
        }
    }

    fn create_lit_int<T: std::fmt::Display>(x: T) -> LitInt {
        LitInt::new(&x.to_string(), Span2::call_site())
    }
    /*let mut module_arms = Vec::with_capacity(module_datas.len());
    let mut kernel_arms = Vec::with_capacity(kernels.iter().map(|x| x.len()).sum());
    for (module_data, kernels) in module_datas.iter().zip(kernels) {
        let module_path = &module_data.path;
        let dependencies = module_data.data.get("dependencies").unwrap();
        let module_tokens = module_data.data.get("krnl_module_tokens").unwrap();
        let module_src = format!("(dependencies({dependencies:?})) => ({module_tokens})");
        let module_path_indices = (0..module_path.len())
            .into_iter()
            .map(create_lit_int)
            .collect::<Vec<_>>();
        module_arms.push(quote! {
            {
                let module_path = #module_path.as_bytes();
                if path.len() == module_path.len() + "::module".len() {
                    let success = #(path[#module_path_indices] == module_path[#module_path_indices])&&*;
                    if success {
                        return Some(#module_src);
                    }
                }
            }
        });
        for (entry_point, spirv_path) in kernels.iter() {
            let spirv_bytes = fs::read(spirv_path)?;
            let (kernel, kernel_hash, spirv_words, capabilities) =
                process_kernel(module_path, entry_point, &spirv_bytes)?;
            let spirv_words = spirv_words
                .into_iter()
                .map(create_lit_int)
                .collect::<Vec<_>>();
            let kernel_path = format!("{module_path}::{kernel}");
            let kernel_path_indices = (0..kernel_path.len())
                .into_iter()
                .map(create_lit_int)
                .collect::<Vec<_>>();
            let mut features = TokenStream2::new();
            {
                use Capability::*;
                for (feature, cap) in [
                    ("shader_int8", Int8),
                    ("shader_int16", Int16),
                    ("shader_int64", Int64),
                    ("shader_float16", Float16),
                    ("shader_float64", Float64),
                ] {
                    let ident = format_ident!("with_{feature}");
                    if capabilities.contains(&cap) {
                        features.extend(quote! {
                            .#ident(true)
                        });
                    }
                }
            }
            kernel_arms.push(quote! {
                {
                    let kernel_path = #kernel_path.as_bytes();
                    let kernel_spirv = &[#(#spirv_words),*];
                    let features = Features::new()
                        #features;
                    if path.len() == kernel_path.len() {
                        let success = #(path[#kernel_path_indices] == kernel_path[#kernel_path_indices])&&*;
                        if success {
                            return Some((#kernel_hash, kernel_spirv, features));
                        }
                    }
                }
            });
        }
    }*/

    let span = Span2::call_site();
    let version_major = LitInt::new(env!("CARGO_PKG_VERSION_MAJOR"), span);
    let version_minor = LitInt::new(env!("CARGO_PKG_VERSION_MINOR"), span);
    let version_patch = LitInt::new(env!("CARGO_PKG_VERSION_PATCH"), span);
    let cache = quote! {
        pub(super) const KRNLC_VERSION_MAJOR: u32 = #version_major;
        pub(super) const KRNLC_VERSION_MINOR: u32 = #version_minor;
        pub(super) const KRNLC_VERSION_PATCH: u32 = #version_patch;
        pub(super) const fn module(path: &'static str) -> Option<&'static ::krnl::kernel::__private::ModuleDesc> {
            None
        }
    };
    fn tokens_to_string(tokens: TokenStream2) -> String {
        use proc_macro2::TokenTree as TokenTree2;
        use std::fmt::Write;
        let mut string = String::new();
        let mut iter = tokens.into_iter().peekable();
        while let Some(token) = iter.next() {
            use TokenTree2::*;
            match token {
                Group(group) => {
                    use proc_macro2::Delimiter;
                    let (pfx, sfx) = match group.delimiter() {
                        Delimiter::Parenthesis => (Some('('), Some(')')),
                        Delimiter::Brace => (Some('{'), Some('}')),
                        Delimiter::Bracket => (Some('['), Some(']')),
                        Delimiter::None => (None, None),
                    };
                    if let Some(pfx) = pfx {
                        string.push(pfx);
                    }
                    string.push_str(&tokens_to_string(group.stream()));
                    if let Some(sfx) = sfx {
                        string.push(sfx);
                    }
                }
                Ident(ident) => {
                    string.push_str(&ident.to_string());
                    if let Some(next) = iter.peek() {
                        if let TokenTree2::Ident(_) = next {
                            string.push(' ');
                        }
                    }
                }
                Punct(punct) => {
                    string.push(punct.as_char());
                }
                Literal(literal) => {
                    string.push_str(&literal.to_string());
                }
            }
        }
        string
    }
    let cache = tokens_to_string(cache);
    let krnlc_cache: Vec<u8> = header.bytes().chain(cache.bytes()).collect();
    fs::write(&krnlc_cache_path, &krnlc_cache)?;
    Ok(())
}

fn process_kernel(
    module_path: &str,
    entry_point: &str,
    spirv_bytes: &[u8],
) -> Result<(String, u64, Vec<u32>, HashSet<Capability>)> {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Instruction, Operand},
        spirv::{BuiltIn, Decoration, ExecutionMode, ExecutionModel, Op, StorageClass},
    };
    use spirv_tools::{opt::Optimizer, val::Validator, TargetEnv};
    let prefix = "__krnl_";
    if !entry_point.starts_with(prefix) {
        return Err(format_err!("Found unexpected entry point `{entry_point}`!"));
    }
    let entry_point = entry_point.split_at(prefix.len()).1;
    let (hash, entry_point) = entry_point.split_once('_').unwrap();
    let hash = u64::from_str(hash)?;
    let (push_consts_size, entry_point) = entry_point.split_once('_').unwrap();
    let push_consts_size = u32::from_str(push_consts_size)?;
    let kernel_name = format!("{module_path}::{entry_point}");
    if push_consts_size % 4 != 0 {
        bail!("Kernel `{kernel_name}` push constant size not multiple of 4 bytes!");
    }
    let mut module = rspirv::dr::load_bytes(&spirv_bytes).map_err(|e| format_err!("{e}"))?;
    /*{
        use rspirv::binary::Disassemble;
        eprintln!("{}", module.disassemble());
    }*/
    let entry_id = if let [entry_point] = module.entry_points.as_mut_slice() {
        if let [Operand::ExecutionModel(exec_model), Operand::IdRef(entry_id), Operand::LiteralString(entry), ..] =
            entry_point.operands.as_mut_slice()
        {
            if *exec_model != ExecutionModel::GLCompute {
                bail!("Kernel `{kernel_name}` execution model should be GLCompute, found {exec_model:?}!");
            }
            *entry = "main".to_string();
            *entry_id
        } else {
            bail!("Kernel `{kernel_name}` unable to parse entry point!");
        }
    } else {
        bail!("Kernel `{kernel_name}` should have 1 entry point!");
    };
    let entry_fn_index = module
        .functions
        .iter()
        .position(|x| x.def_id() == Some(entry_id))
        .unwrap();
    let mut names = HashMap::<u32, &str>::new();
    for inst in module.debug_names.iter() {
        let op = inst.class.opcode;
        if op == Op::Name {
            if let [Operand::IdRef(id), Operand::LiteralString(name)] = inst.operands.as_slice() {
                if !name.is_empty() {
                    names.insert(*id, name);
                }
            }
        }
    }
    let mut bindings = HashMap::<u32, u32>::new();
    for inst in module.annotations.iter() {
        let op = inst.class.opcode;
        let operands = inst.operands.as_slice();
        if op == Op::Decorate {
            match operands {
                [Operand::IdRef(id), Operand::Decoration(Decoration::DescriptorSet), Operand::LiteralInt32(set)] => {
                    if *set != 0 {
                        bail!("Kernel `{kernel_name}` descriptor set mut be 0!");
                    }
                }
                [Operand::IdRef(id), Operand::Decoration(Decoration::Binding), Operand::LiteralInt32(binding)] =>
                {
                    bindings.insert(*id, *binding);
                }
                _ => (),
            }
        }
    }
    let mut non_writable = HashSet::<u32>::new();
    for (id, binding) in bindings.iter() {
        if let Some(name) = names.get(id) {
            if name.ends_with("_r") {
                non_writable.insert(*id);
            }
        } else {
            bail!("Kernel `{kernel_name}` buffer with binding `{binding}` requires a name!");
        }
    }
    let mut push_consts_id = None;
    let mut push_consts_ptr = None;
    for inst in module.types_global_values.iter() {
        let op = inst.class.opcode;
        if op == Op::Variable {
            if let [Operand::StorageClass(StorageClass::PushConstant), ..] =
                inst.operands.as_slice()
            {
                if push_consts_id.is_none() {
                    push_consts_id = inst.result_id;
                    push_consts_ptr = inst.result_type;
                } else {
                    bail!("Kernel `{kernel_name}` expected at most 1 push constant variable!");
                }
            }
        }
    }
    let mut builder = Builder::new_from_module(module);
    for id in non_writable.iter() {
        builder.decorate(*id, Decoration::NonWritable, []);
    }
    let type_u32 = builder.type_int(32, 0);
    let type_i32 = builder.type_int(32, 1);
    let type_ptr_u32 = builder.type_pointer(None, StorageClass::PushConstant, type_u32);
    let (push_consts_id, push_consts_struct, dyn_member_start) =
        if let Some((push_consts_id, push_consts_ptr)) = push_consts_id.zip(push_consts_ptr) {
            let push_consts_struct = builder
                .module_ref()
                .types_global_values
                .iter()
                .find(|x| x.result_id == Some(push_consts_ptr))
                .unwrap()
                .operands[1]
                .unwrap_id_ref();
            let push_consts_struct_inst = builder
                .module_mut()
                .types_global_values
                .iter_mut()
                .find(|x| x.result_id == Some(push_consts_struct))
                .unwrap();
            let dyn_member_start = push_consts_struct_inst.operands.len() as u32;
            push_consts_struct_inst
                .operands
                .extend(std::iter::repeat(Operand::IdRef(type_u32)).take(bindings.len() * 2));
            (push_consts_id, push_consts_struct, dyn_member_start)
        } else {
            // push consts have been optimized away
            let push_consts_struct = builder.id();
            builder.type_struct_id(
                Some(push_consts_struct),
                std::iter::repeat(type_u32).take(bindings.len() * 2),
            );
            builder.decorate(push_consts_struct, Decoration::Block, []);
            let push_consts_ptr =
                builder.type_pointer(None, StorageClass::PushConstant, push_consts_struct);
            let push_consts_id =
                builder.variable(push_consts_ptr, None, StorageClass::PushConstant, None);
            builder.module_mut().entry_points[0]
                .operands
                .push(Operand::IdRef(push_consts_id));
            let dyn_member_start = 0;
            (push_consts_id, push_consts_struct, dyn_member_start)
        };
    let dyn_offset_start = push_consts_size
        .checked_sub(2 * bindings.len() as u32 * 4)
        .unwrap();
    for i in 0..bindings.len() as u32 * 2 {
        let member = dyn_member_start + i;
        builder.member_decorate(
            push_consts_struct,
            member,
            Decoration::Offset,
            [Operand::LiteralInt32(dyn_offset_start + i * 4)],
        );
    }
    // dynamic offset and len
    {
        let mut b = 0;
        let mut i = 0;
        loop {
            #[derive(Debug)]
            enum DynOp {
                Offset { binding: u32, index: u32 },
                Len { binding: u32 },
            }
            let mut dyn_op = None;
            'block: for block in builder.module_mut().functions[entry_fn_index]
                .blocks
                .iter_mut()
                .skip(b)
            {
                for inst in block.instructions.iter_mut().skip(i) {
                    let op = inst.class.opcode;
                    if op == Op::AccessChain {
                        let id = inst.operands[0].unwrap_id_ref();
                        if let Some(binding) = bindings.get(&id).copied() {
                            if let Some(index) = inst.operands.get(2).map(|x| x.unwrap_id_ref()) {
                                dyn_op.replace(DynOp::Offset { binding, index });
                                break 'block;
                            }
                        }
                    } else if op == Op::ArrayLength {
                        let id = inst.operands[0].unwrap_id_ref();
                        if let Some(binding) = bindings.get(&id).copied() {
                            dyn_op.replace(DynOp::Len { binding });
                            break 'block;
                        }
                    }
                    i += 1;
                }
                b += 1;
                i = 0;
            }
            if let Some(dyn_op) = dyn_op.take() {
                builder.select_function(Some(entry_fn_index)).unwrap();
                builder.select_block(Some(b)).unwrap();
                match dyn_op {
                    DynOp::Offset { binding, index } => {
                        let member = builder.constant_u32(type_i32, 2 * binding + dyn_member_start);
                        let offset_ptr = builder.insert_access_chain(
                            InsertPoint::FromBegin(i),
                            type_ptr_u32,
                            None,
                            push_consts_id,
                            [member],
                        )?;
                        i += 1;
                        let offset = builder.insert_load(
                            InsertPoint::FromBegin(i),
                            type_u32,
                            None,
                            offset_ptr,
                            None,
                            [],
                        )?;
                        i += 1;
                        let index = builder.insert_i_add(
                            InsertPoint::FromBegin(i),
                            type_u32,
                            None,
                            index,
                            offset,
                        )?;
                        i += 1;
                        let inst = &mut builder.module_mut().functions[entry_fn_index].blocks[b]
                            .instructions[i];
                        inst.operands[2] = Operand::IdRef(index);
                        i += 1;
                    }
                    DynOp::Len { binding } => {
                        let inst = builder.module_mut().functions[entry_fn_index].blocks[b]
                            .instructions
                            .remove(i);
                        let member =
                            builder.constant_u32(type_i32, 2 * binding + 1 + dyn_member_start);
                        let len_ptr = builder.insert_access_chain(
                            InsertPoint::FromBegin(i),
                            type_ptr_u32,
                            None,
                            push_consts_id,
                            [member],
                        )?;
                        i += 1;
                        builder.insert_load(
                            InsertPoint::FromBegin(i),
                            type_u32,
                            inst.result_id,
                            len_ptr,
                            None,
                            [],
                        )?;
                        i += 1;
                    }
                }
            } else {
                break;
            }
        }
    }
    let mut module = builder.module();
    let mut capabilities = HashSet::with_capacity(module.capabilities.len());
    let mut extensions = HashSet::with_capacity(module.extensions.len());
    for inst in module.types_global_values.iter() {
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
    module.capabilities.retain(|inst| {
        if let [Operand::Capability(cap)] = inst.operands.as_slice() {
            matches!(cap, Capability::Shader | Capability::VulkanMemoryModel)
                || capabilities.contains(cap)
        } else {
            false
        }
    });
    for inst in module.ext_inst_imports.iter() {
        if let [Operand::LiteralString(ext_inst_import)] = inst.operands.as_slice() {
            if ext_inst_import.starts_with("NonSemantic") {
                extensions.insert("SPV_KHR_non_semantic_info");
            }
        }
    }
    module.extensions.retain(|inst| {
        if let [Operand::LiteralString(ext)] = inst.operands.as_slice() {
            ext.as_str() == "SPV_KHR_vulkan_memory_model" || extensions.contains(ext.as_str())
        } else {
            false
        }
    });
    let spirv_words = module.assemble();
    let target_env = spirv_tools::TargetEnv::Vulkan_1_2;
    let validator = spirv_tools::val::create(Some(target_env));
    validator.validate(&spirv_words, None)?;
    let mut optimizer = spirv_tools::opt::create(Some(target_env));
    optimizer
        .register_performance_passes()
        .register_size_passes();
    let spirv_words = optimizer
        .optimize(&spirv_words, &mut |_| (), None)?
        .as_words()
        .to_vec();
    /*{
        use rspirv::binary::Disassemble;
        let module = rspirv::dr::load_words(&spirv_words).unwrap();
        eprintln!("{}", module.disassemble());
        panic!();
    }*/
    Ok((entry_point.to_string(), hash, spirv_words, capabilities))
}
