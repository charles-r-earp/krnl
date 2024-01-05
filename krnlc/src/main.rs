#![forbid(unsafe_code)]

use anyhow::{bail, Error, Result};
use cargo_metadata::{Metadata, Package, PackageId};
use clap::Parser;
use clap_cargo::{Manifest, Workspace};
use fxhash::{FxHashMap, FxHashSet};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use semver::{Version, VersionReq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use spirv_builder::{MetadataPrintout, SpirvBuilder, SpirvMetadata};
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str::FromStr,
};
use syn::{visit::Visit, Expr, Item, ItemMod, Lit, Visibility};

#[derive(Parser, Debug)]
#[command(
    name = "krnlc",
    version,
    about = "Compiler for krnl.",
    long_about = "Compiler for krnl.\n\nCollects `#[modules]`s and compiles them, creates \"krnl-cache.rs\"."
)]
struct Cli {
    #[command(flatten)]
    workspace: Workspace,
    #[command(flatten)]
    manifest: Manifest,
    /// Directory for all generated artifacts
    #[arg(long = "target-dir")]
    target_dir: Option<PathBuf>,
    /// Check mode.
    #[arg(long = "check")]
    check: bool,
    #[arg(long = "debug-printf", hide = true)]
    debug_printf: bool,
    /// Use verbose output.
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let metadata = cli.manifest.metadata().exec()?;
    let (selected, _) = cli.workspace.partition_packages(&metadata);
    let target_dir = cli
        .target_dir
        .as_ref()
        .map(|x| x.to_string_lossy())
        .unwrap_or(metadata.target_directory.as_str().into());
    for package in selected.iter().copied() {
        let krnlc_metadata = KrnlcMetadata::new(&metadata, package)?;
        let module_sources = cargo_expand(package, &target_dir, &krnlc_metadata, cli.verbose)?;
        if module_sources.is_empty() {
            continue;
        }
        let modules = compile(
            package,
            &target_dir,
            &krnlc_metadata.dependencies,
            module_sources,
            cli.debug_printf,
            cli.verbose,
        )?;
        cache(package, modules, cli.check)?;
    }
    Ok(())
}

fn cargo_expand(
    package: &Package,
    target_dir: &str,
    krnlc_metadata: &KrnlcMetadata,
    verbose: bool,
) -> Result<FxHashMap<String, String>> {
    use std::env::var;
    let mut command = Command::new("cargo");
    if let Ok("stable" | "beta") | Err(_) = var("RUSTUP_TOOLCHAIN").as_deref() {
        command.arg("+nightly");
    }
    command.args([
        "rustc",
        "--manifest-path",
        package.manifest_path.as_str(),
        "--target-dir",
        target_dir,
    ]);
    if verbose {
        command.arg("-v");
    }
    if !krnlc_metadata.default_features {
        command.arg("--no-default-features");
    }
    if !krnlc_metadata.features.is_empty() {
        command.args(["--features", &krnlc_metadata.features]);
    }
    command
        .args([
            "--profile=check",
            "--",
            "--cfg=krnlc",
            "-Zunpretty=expanded",
        ])
        .stderr(Stdio::inherit());
    let output = command.output()?;
    if !output.status.success() {
        bail!("expansion failed!");
    }
    let expanded = std::str::from_utf8(&output.stdout)?;
    let file: syn::File = syn::parse_str(expanded)?;
    let mut modules = FxHashMap::default();
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

// Should match krnl_macros
fn krnlc_version_compatible(krnlc_version: &str, version: &str) -> bool {
    let krnlc_version = Version::parse(krnlc_version).unwrap();
    let (version, req) = (
        Version::parse(version).unwrap(),
        VersionReq::parse(version).unwrap(),
    );
    if !req.matches(&krnlc_version) {
        return false;
    }
    if !krnlc_version.pre.is_empty() && krnlc_version < version {
        return false;
    }
    if !version.pre.is_empty() && version != krnlc_version {
        return false;
    }
    true
}

fn pretty_fmt(input: &str) -> Result<String> {
    let output = prettyplease::unparse(&syn::parse_file(input)?);
    Ok(output)
}

struct KrnlcMetadata {
    default_features: bool,
    features: String,
    dependencies: String,
}

impl KrnlcMetadata {
    fn new(metadata: &Metadata, package: &Package) -> Result<Self> {
        use std::collections::HashSet;
        use std::fmt::Write;

        fn find_krnl_core<'a>(
            metadata: &'a Metadata,
            root: &'a PackageId,
            searched: &mut HashSet<&'a PackageId>,
        ) -> Option<&'a Package> {
            searched.insert(root);
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
                if !searched.contains(id) {
                    if let Some(package) = find_krnl_core(metadata, id, searched) {
                        return Some(package);
                    }
                }
            }
            None
        }
        let mut searched = HashSet::new();
        let krnl_core_package =
            if let Some(package) = find_krnl_core(metadata, &package.id, &mut searched) {
                package
            } else {
                bail!(
                    "krnl-core is not in dependency tree of package {:?}!",
                    package.name
                );
            };
        if !krnlc_version_compatible(
            env!("CARGO_PKG_VERSION"),
            &krnl_core_package.version.to_string(),
        ) {
            bail!("krnlc version is not compatible!");
        }
        let krnl_core_source = format!(
            " path = {:?}",
            krnl_core_package.manifest_path.parent().unwrap()
        );
        let manifest_path_str = package.manifest_path.as_str();
        let manifest_dir = package.manifest_path.parent().unwrap();
        let mut default_features = true;
        let mut features = String::new();
        let mut dependencies = String::new();
        let mut has_krnl_core = false;
        if let Some(krnlc_metadata) = package.metadata.get("krnlc") {
            if let Some(metadata_default_features) = krnlc_metadata.get("default-features") {
                if let Some(metadata_default_features) = metadata_default_features.as_bool() {
                    default_features = metadata_default_features;
                } else {
                    bail!("{manifest_path_str:?} [package.metadata.krnlc] default-features, expected bool!");
                }
            }
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
                    bail!(
                        "{manifest_path_str:?} [package.metadata.krnlc] features, expected array!"
                    );
                }
            }
            if let Some(metadata_dependencies) = krnlc_metadata.get("dependencies") {
                if let Some(metadata_dependencies) = metadata_dependencies.as_object() {
                    for (dep, value) in metadata_dependencies.iter() {
                        let (mut dep_source, dep_default_features, dep_features) = if dep
                            == "krnl-core"
                        {
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
                                if source == "registry+https://github.com/rust-lang/crates.io-index"
                                {
                                    format!("version = \"{}\"", dependency.req)
                                } else if let Some((key, value)) = source.split_once('+') {
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
                            let source = String::new();
                            let dep_default_features = false;
                            let features = Vec::new();
                            (source, dep_default_features, features)
                        };
                        let mut inherit_from_host_dep = true;
                        let mut default_features = None;
                        let mut features = Vec::new();
                        if let Some(table) = value.as_object() {
                            for (key, value) in table.iter() {
                                match key.as_str() {
                                    "path" => {
                                        if let Some(value) = value.as_str() {
                                            let mut path = PathBuf::from(value);
                                            if path.is_relative() {
                                                path = manifest_dir
                                                    .as_std_path()
                                                    .join(&path)
                                                    .canonicalize()?;
                                            }
                                            if !path.exists() {
                                                bail!(
                                                    "{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?} path = {value:?} does not exist!"
                                                );
                                            }
                                            dep_source = format!("path = {path:?}");
                                            inherit_from_host_dep = false;
                                        } else {
                                            bail!(
                                                "{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?} path, expected string!"
                                            );
                                        }
                                    }
                                    "default-features" => {
                                        if let Some(value) = value.as_bool() {
                                            default_features.replace(value);
                                        } else {
                                            bail!(
                                                "{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?} default-features, expected bool!"
                                            );
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
                        if features.is_empty() && inherit_from_host_dep {
                            features = dep_features;
                        }
                        let mut features = itertools::join(features, ", ");
                        if !features.is_empty() && inherit_from_host_dep {
                            features = format!("{features:?}");
                        }
                        if dep_source.is_empty() {
                            bail!("{manifest_path_str:?} [package.metadata.krnlc.dependencies] {dep:?} is not a dependency of {:?}!", package.name);
                        }
                        writeln!(&mut dependencies, "{dep:?} = {{ {dep_source}, features = [{features}], default-features = {default_features} }}").unwrap();
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
        Ok(Self {
            default_features,
            features,
            dependencies,
        })
    }
}

struct ModuleVisitor<'a> {
    path: String,
    modules: &'a mut FxHashMap<String, String>,
    result: &'a mut Result<()>,
}

impl<'a, 'ast> Visit<'ast> for ModuleVisitor<'a> {
    fn visit_item_mod(&mut self, i: &'ast ItemMod) {
        if self.result.is_err() {
            return;
        }
        if !self.path.is_empty()
            && i.ident == "__krnl_module_data"
            && i.vis == Visibility::Inherited
        {
            if let Some((_, items)) = i.content.as_ref() {
                if let [Item::Const(item_const)] = items.as_slice() {
                    if item_const.ident == "__krnl_module_source" {
                        if let Expr::Lit(expr_lit) = item_const.expr.as_ref() {
                            if let Lit::Str(lit_str) = &expr_lit.lit {
                                self.modules.insert(self.path.clone(), lit_str.value());
                                return;
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

fn cache(package: &Package, kernels: Vec<KernelDesc>, check: bool) -> Result<()> {
    use flate2::{write::GzEncoder, Compression};

    let version = env!("CARGO_PKG_VERSION");
    let cache = KrnlcCache {
        version: version.to_string(),
        kernels,
    };
    let mut bytes = Vec::new();
    let encoder = GzEncoder::new(&mut bytes, Compression::fast());
    bincode2::serialize_into(encoder, &cache)?;
    let cache = itertools::join(bytes.chunks(40).map(hex::encode), ",\nx");
    let manifest_dir = package.manifest_path.parent().unwrap();
    let cache_path = manifest_dir.join("krnl-cache.rs");
    let cache = format!(
        r#"/* generated by krnlc {version} */
    #[doc(hidden)]
    macro_rules! __krnl_module {{
        ($m:ident, $k:ident) => {{
            __krnl_cache!($m, $k, x{cache})
        }};
    }}"#
    );
    if check {
        let prev = std::fs::read_to_string(&cache_path)?;
        for (i, (prev, cache)) in prev.lines().zip(cache.lines()).enumerate() {
            if prev != cache {
                eprintln!("{i}: {prev}");
                eprintln!("{i}: {cache}");
                bail!("{cache_path:?} check failed!");
            }
        }
    } else {
        std::fs::write(cache_path, cache.as_bytes())?;
    }
    Ok(())
}

fn compile(
    package: &Package,
    target_dir: &str,
    dependencies: &str,
    module_sources: FxHashMap<String, String>,
    debug_printf: bool,
    verbose: bool,
) -> Result<Vec<KernelDesc>> {
    use std::{
        env::consts::{DLL_PREFIX, DLL_SUFFIX},
        sync::Once,
    };
    let target_krnl_dir = PathBuf::from(target_dir).join("krnlc");

    static INIT_LIB_DIR: Once = Once::new();
    if !INIT_LIB_DIR.is_completed() {
        std::fs::create_dir_all(&target_krnl_dir)?;
        let lib_dir = target_krnl_dir.join("lib");
        if !lib_dir.exists() {
            std::fs::create_dir(&lib_dir)?;
        }
        for lib in [
            option_env!("KRNLC_LIBLLVM"),
            option_env!("KRNLC_LIBRUSTC_DRIVER"),
            option_env!("KRNLC_LIBSTD"),
        ]
        .into_iter()
        .flatten()
        {
            let link = lib_dir.join(lib);
            if !link.exists() {
                symlink::symlink_file(
                    &PathBuf::from(env!("KRNLC_TOOLCHAIN_LIB")).join(lib),
                    &link,
                )?;
            }
        }
        let rustc_codegen_spirv_lib = format!("{DLL_PREFIX}rustc_codegen_spirv{DLL_SUFFIX}");
        let librustc_codegen_spirv = include_bytes!(env!("KRNLC_LIBRUSTC_CODEGEN_SPIRV"));
        std::fs::write(
            lib_dir.join(&rustc_codegen_spirv_lib),
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
        INIT_LIB_DIR.call_once(|| {});
    }
    let crate_name = package.name.as_str();
    let device_crate_dir = target_krnl_dir.join("crates").join(crate_name);
    let device_crate_manifest_path = device_crate_dir.join("Cargo.toml");
    let mut update = false;
    {
        // device crate
        std::fs::create_dir_all(&device_crate_dir)?;
        let config_dir = device_crate_dir.join(".cargo");
        std::fs::create_dir_all(&config_dir)?;
        let config = format!(
            r#"[build]
target-dir = {target_dir:?}

[term]
verbose = {verbose}
"#
        );
        std::fs::write(config_dir.join("config.toml"), config.as_bytes())?;
        let build_script = r#"fn main() {
            println!("cargo:rustc-cfg=krnlc");
        }
                "#;
        std::fs::write(device_crate_dir.join("build.rs"), build_script.as_bytes())?;
        let manifest = format!(
            r#"# generated by krnlc 
[package]
name = {crate_name:?}
version = "0.0.0"
edition = "2021"
publish = false

[workspace]

[lib]
crate-type = ["dylib"]

[dependencies]
{dependencies}
"#
        );
        if let Ok(old_manifest) = std::fs::read_to_string(&device_crate_manifest_path) {
            if manifest != old_manifest {
                update = true;
            }
        }
        std::fs::write(&device_crate_manifest_path, manifest.as_bytes())?;
        let toolchain = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/rust-toolchain.toml"));
        std::fs::write(
            device_crate_dir.join("rust-toolchain.toml"),
            toolchain.as_bytes(),
        )?;
        let src_dir = device_crate_dir.join("src");
        if !src_dir.exists() {
            std::fs::create_dir(&src_dir)?;
        }
        write_device_source(&src_dir, &module_sources)?;
    }
    if update {
        let status = Command::new("cargo")
            .args([
                "update",
                "--manifest-path",
                device_crate_manifest_path.to_string_lossy().as_ref(),
            ])
            .status()?;
        if !status.success() {
            bail!("cargo update failed!");
        }
    }
    let crate_name_ident = crate_name.replace('-', "_");
    let kernels_dir = device_crate_dir.join("kernels");
    std::fs::create_dir_all(&kernels_dir)?;
    let mut builder = SpirvBuilder::new(&device_crate_dir, "spirv-unknown-vulkan1.2")
        .spirv_metadata(SpirvMetadata::NameVariables)
        .print_metadata(MetadataPrintout::None)
        .deny_warnings(true);
    if debug_printf {
        builder = builder
            .extension("SPV_KHR_non_semantic_info")
            .relax_logical_pointer(true);
    }
    let capabilites = {
        use spirv_builder::Capability::*;
        [
            Int8,
            Int16,
            Int64,
            Float16,
            Float64,
            GroupNonUniform,
            GroupNonUniformArithmetic,
        ]
    };
    for cap in capabilites {
        builder = builder.capability(cap);
    }
    let output = builder.build()?;
    let spirv_path = output.module.unwrap_single();
    let spirv_module = rspirv::dr::load_bytes(std::fs::read(spirv_path)?)
        .map_err(|e| Error::msg(e.to_string()))?;
    let entry_fns: FxHashSet<u32> = spirv_module
        .entry_points
        .iter()
        .map(|inst| inst.operands[1].unwrap_id_ref())
        .collect();
    spirv_module
        .entry_points
        .par_iter()
        .map(|entry_point| {
            kernel_post_process(
                &kernels_dir,
                &crate_name_ident,
                entry_point,
                &spirv_module,
                &entry_fns,
            )
        })
        .collect()
}

fn write_device_source(src_dir: &Path, module_sources: &FxHashMap<String, String>) -> Result<()> {
    if module_sources.is_empty() {
        return Ok(());
    }
    let mut tree = FxHashMap::<&str, FxHashSet<&str>>::default();
    for module in module_sources.keys() {
        let mut parent = "";
        for child in module.split("::") {
            let child = if parent.is_empty() {
                child
            } else {
                &module[..parent.len() + "::".len() + child.len()]
            };
            tree.entry(parent).or_default().insert(child);
            parent = child;
        }
    }
    let mut files = FxHashSet::default();
    fn visit_module(
        dir: &Path,
        module: &str,
        tree: &FxHashMap<&str, FxHashSet<&str>>,
        module_sources: &FxHashMap<String, String>,
        files: &mut FxHashSet<PathBuf>,
    ) -> Result<()> {
        use std::fmt::Write;

        let module_name = module.rsplit_once("::").map_or(module, |x| x.1);
        let file_name = if module_name.is_empty() {
            "lib"
        } else {
            module_name
        };
        let mut source = String::new();
        if module_name.is_empty() {
            source = r#"#![cfg_attr(target_arch = "spirv",
no_std,
feature(asm_experimental_arch),
)]

extern crate krnl_core;

"#
            .to_string();
        }
        let file_path = dir.join(file_name).with_extension("rs");
        if let Some(children) = tree.get(module) {
            for child in children {
                let name = child.rsplit_once("::").map_or(*child, |x| x.1);
                writeln!(source, "pub mod {name};").unwrap();
            }
            let current = std::fs::read_to_string(&file_path).unwrap_or_default();
            if source != current {
                std::fs::write(&file_path, &source)?;
            }
            files.insert(file_path);
            for child in children {
                let child_dir = dir.join(module_name);
                std::fs::create_dir_all(&child_dir)?;
                files.insert(child_dir.clone());
                visit_module(&child_dir, child, tree, module_sources, files)?;
            }
        } else {
            let source = pretty_fmt(&module_sources[module])?;
            let current = std::fs::read_to_string(&file_path).unwrap_or_default();
            if source != current {
                std::fs::write(&file_path, source)?;
            }
            files.insert(file_path);
        }
        Ok(())
    }
    visit_module(src_dir, "", &tree, module_sources, &mut files)?;
    fn cleanup_files(dir: &Path, keep: &FxHashSet<PathBuf>) -> Result<()> {
        for entry in walkdir::WalkDir::new(dir) {
            let entry = entry?;
            if !keep.contains(entry.path()) {
                if entry.file_type().is_dir() {
                    std::fs::remove_dir(entry.path())?;
                } else {
                    std::fs::remove_file(entry.path())?;
                }
            }
        }
        Ok(())
    }
    cleanup_files(src_dir, &files)
}

fn kernel_post_process(
    kernels_dir: &Path,
    crate_name_ident: &str,
    entry_point: &rspirv::dr::Instruction,
    spirv_module: &rspirv::dr::Module,
    entry_fns: &FxHashSet<u32>,
) -> Result<KernelDesc> {
    use rspirv::{
        binary::Assemble,
        dr::{Instruction, Module, Operand},
        spirv::{BuiltIn, Decoration, Op, StorageClass},
    };
    let entry_id = entry_point.operands[1].unwrap_id_ref();
    let kernel_name = entry_point.operands[2].unwrap_literal_string();
    let execution_mode = spirv_module
        .execution_modes
        .iter()
        .find(|inst| inst.operands.first().unwrap().unwrap_id_ref() == entry_id)
        .unwrap();
    let functions = spirv_module
        .functions
        .iter()
        .filter(|f| {
            let id = f.def.as_ref().unwrap().result_id.unwrap();
            id == entry_id || !entry_fns.contains(&id)
        })
        .cloned()
        .collect();
    let kernel_desc = (|| -> Result<KernelDesc> {
        let spirv_module = Module {
            entry_points: vec![entry_point.clone()],
            execution_modes: vec![execution_mode.clone()],
            functions,
            ..spirv_module.clone()
        };
        let spirv = spirv_module.assemble();
        let spirv = spirv_opt(&spirv, SpirvOptKind::DeadCodeElimination)?;
        let mut spirv_module = rspirv::dr::load_words(&spirv).map_err(|e| Error::msg(e.to_string())).unwrap();
        let mut kernel_desc: KernelDesc = {
            let mut kernel_data_var = None;
            for inst in spirv_module.annotations.iter() {
                let op = inst.class.opcode;
                if op == Op::Decorate {
                    if let [Operand::IdRef(id), Operand::Decoration(Decoration::DescriptorSet), Operand::LiteralInt32(1)] =
                        inst.operands.as_slice()
                    {
                        kernel_data_var.replace(*id);
                        break;
                    }
                }
            }
            let kernel_data_var = if let Some(var) = kernel_data_var {
                var
            } else {
                bail!("Unable to decode kernel {kernel_name}!");
            };
            spirv_module.annotations.retain(|inst| {
                let op = inst.class.opcode;
                !(op == Op::Decorate && inst.operands.first() == Some(&Operand::IdRef(kernel_data_var)))
            });
            spirv_module.entry_points[0].operands.retain(|x| {
                if let Operand::IdRef(id) = x {
                    *id != kernel_data_var
                } else {
                    true
                }
            });
            add_spec_constant_ops(&mut spirv_module);
            let mut constants = FxHashMap::default();
            for inst in spirv_module.types_global_values.iter() {
                if let Some(result_id) = inst.result_id {
                    let op = inst.class.opcode;
                    let operands = inst.operands.as_slice();
                    if let (Op::Constant, [Operand::LiteralInt32(value)]) = (op, operands) {
                        constants.insert(result_id, *value);
                    }
                }
            }
            let mut kernel_data_ptrs = FxHashMap::default();
            let mut kernel_data_stores = FxHashMap::default();
            for function in spirv_module.functions.iter_mut() {
                for block in function.blocks.iter_mut() {
                    block.instructions.retain(|inst| {
                        let op = inst.class.opcode;
                        let operands = inst.operands.as_slice();
                        match (op, operands) {
                            (Op::AccessChain, [Operand::IdRef(var), _, Operand::IdRef(index)]) => {
                                if *var == kernel_data_var {
                                    kernel_data_ptrs.insert(inst.result_id.unwrap(), *index);
                                    return false;
                                }
                            }
                            (Op::AccessChain, [Operand::IdRef(var), ..]) => {
                                if *var == kernel_data_var {
                                    return false;
                                }
                            }
                            (Op::Store, [Operand::IdRef(ptr), Operand::IdRef(value)]) => {
                                if let Some(index) = kernel_data_ptrs.get(ptr) {
                                    if let Some(id) = constants.get(index) {
                                        kernel_data_stores.insert(*id, *value);
                                    }
                                    return false;
                                }
                            }
                            _ => {}
                        }
                        true
                    });
                }
            }
            let mut kernel_data = None;
            let mut array_ids = FxHashMap::with_capacity_and_hasher(
                kernel_data_stores.len().checked_sub(1).unwrap_or_default(),
                Default::default(),
            );
            spirv_module.debug_names.retain_mut(|inst| {
                let op = inst.class.opcode;
                let operands = inst.operands.as_mut_slice();
                if let (Op::Name, [Operand::IdRef(var), Operand::LiteralString(name)]) = (op, operands)
                {
                    if *var == kernel_data_var {
                        kernel_data.replace(std::mem::take(name));
                        return false;
                    } else if name.starts_with("__krnl_group_array_")
                        || name.starts_with("__krnl_subgroup_array_")
                    {
                        if let Some(id) = name.rsplit_once('_').and_then(|x| x.1.parse().ok()) {
                            array_ids.insert(*var, id);
                        }
                    }
                }
                true
            });
            if !array_ids.is_empty() {
                let mut array_types = FxHashMap::default();
                let mut pointer_types = FxHashMap::default();
                let mut pointer_lens = FxHashMap::default();
                for inst in spirv_module.types_global_values.iter() {
                    if let Some(result_id) = inst.result_id {
                        let op = inst.class.opcode;
                        let operands = inst.operands.as_slice();
                        match (op, operands) {
                            (Op::Constant, [Operand::LiteralInt32(value)]) => {
                                constants.insert(result_id, *value);
                            }
                            (Op::TypeArray, [Operand::IdRef(ty), Operand::IdRef(_)]) => {
                                array_types.insert(result_id, *ty);
                            }
                            (
                                Op::TypePointer,
                                [Operand::StorageClass(storage_class), Operand::IdRef(pointee)],
                            ) => {
                                if *storage_class == StorageClass::Workgroup
                                    || *storage_class == StorageClass::Private
                                {
                                    if let Some(ty) = array_types.get(pointee) {
                                        pointer_types.insert(result_id, *ty);
                                    }
                                }
                            }
                            (Op::Variable, _) => {
                                if let Some((result_type, id)) =
                                    inst.result_type.zip(array_ids.get(&result_id))
                                {
                                    if let Some(len) = kernel_data_stores.get(id) {
                                        pointer_lens.insert(result_type, *len);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let mut id_counter = spirv_module.header.as_ref().unwrap().bound;
                let mut scalars = Vec::new();
                let mut constants = Vec::new();
                let mut types_global_values =
                    Vec::with_capacity(spirv_module.types_global_values.len() + array_ids.len());
                for mut inst in spirv_module.types_global_values {
                    if inst.result_id == Some(kernel_data_var) {
                        continue;
                    } else if let Some(result_id) = inst.result_id {
                        let op = inst.class.opcode;
                        let operands = inst.operands.as_mut_slice();
                        match (op, operands) {
                            (Op::TypeInt | Op::TypeFloat | Op::TypeBool, _) => {
                                scalars.push(inst);
                                continue;
                            }
                            (
                                Op::Constant
                                | Op::ConstantTrue
                                | Op::ConstantFalse
                                | Op::ConstantNull
                                | Op::SpecConstant
                                | Op::SpecConstantTrue
                                | Op::SpecConstantFalse
                                | Op::SpecConstantOp,
                                _,
                            ) => {
                                if scalars.iter().any(|x| x.result_id == inst.result_type) {
                                    constants.push(inst);
                                }
                                continue;
                            }
                            (Op::TypePointer, [Operand::StorageClass(_), Operand::IdRef(pointee)]) => {
                                if let Some((ty, len)) = pointer_types
                                    .get(&result_id)
                                    .zip(pointer_lens.get(&result_id))
                                {
                                    *pointee = id_counter;
                                    id_counter += 1;
                                    types_global_values.push(Instruction::new(
                                        Op::TypeArray,
                                        None,
                                        Some(*pointee),
                                        vec![Operand::IdRef(*ty), Operand::IdRef(*len)],
                                    ));
                                }
                            }
                            (Op::Variable, [Operand::StorageClass(storage_class)]) => {
                                if *storage_class == StorageClass::Private {
                                    if let Some(result_type) = inst.result_type {
                                        if let Some(ty) = pointer_types.get(&result_type) {
                                            let null = id_counter;
                                            id_counter += 1;
                                            types_global_values.push(Instruction::new(
                                                Op::ConstantNull,
                                                Some(*ty),
                                                Some(null),
                                                Vec::new(),
                                            ));
                                            inst.operands.push(Operand::IdRef(null));
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    types_global_values.push(inst);
                }
                spirv_module.types_global_values = scalars
                    .into_iter()
                    .chain(constants)
                    .chain(types_global_values)
                    .collect();
                spirv_module.header.as_mut().unwrap().bound = id_counter;
            } else {
                for (i, inst) in spirv_module.types_global_values.iter().enumerate() {
                    if inst.result_id == Some(kernel_data_var) {
                        spirv_module.types_global_values.remove(i);
                        break;
                    }
                }
            }
            let kernel_data = if let Some(kernel_data) = kernel_data
                .as_ref()
                .and_then(|x| x.strip_prefix("__krnl_kernel_data_"))
            {
                kernel_data
            } else {
                bail!("Unable to decode kernel {kernel_name}, found {kernel_data:?}!");
            };
            let bytes = hex::decode(kernel_data).unwrap();
            bincode2::deserialize(&bytes).unwrap()
        };
        {
            let mut builder = rspirv::dr::Builder::new_from_module(std::mem::take(&mut spirv_module));
            let uint = builder.type_int(32, 0);
            let one = builder.constant_u32(uint, 1);
            let threads = builder.spec_constant_u32(uint, 1);
            let spec_id = kernel_desc.spec_descs.len() as u32;
            builder.decorate(
                threads,
                Decoration::SpecId,
                [Operand::LiteralInt32(spec_id)],
            );
            let uvec3 = builder.type_vector(uint, 3);
            let workgroup_size = builder.spec_constant_composite(uvec3, [threads, one, one]);
            builder.decorate(workgroup_size, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupSize)]);
            spirv_module = builder.module();
        }
        spirv_module.entry_points.first_mut().unwrap().operands[2] =
            Operand::LiteralString("main".to_string());
        kernel_desc.name = format!("{crate_name_ident}::{kernel_name}");
        let mut features = Features::default();
        for inst in spirv_module.types_global_values.iter() {
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
        spirv_module.capabilities.retain(|inst| {
            use rspirv::spirv::Capability::*;
            match inst.operands.first().unwrap().unwrap_capability() {
                Int8 => features.shader_int8,
                Int16 => features.shader_int16,
                Int64 => features.shader_int64,
                Float16 => features.shader_float16,
                Float64 => features.shader_float64,
                _ => true,
            }
        });
        let spirv = spirv_module.assemble();
        let spirv = spirv_opt(&spirv, SpirvOptKind::Performance)?;
        {
            let path = kernels_dir.join(kernel_desc.name.replace("::", "/"));
            std::fs::create_dir_all(path.parent().unwrap())?;
            let string = serde_json::to_string_pretty(&kernel_desc)?;
            std::fs::write(path.with_extension("json"), string.as_bytes())?;
            std::fs::write(
                path.with_extension("spv"),
                bytemuck::cast_slice(spirv.as_slice()),
            )?;
        }
        kernel_desc.spirv = spirv;
        Ok(kernel_desc)
    })().map_err(|e| {
        e.context(kernel_name.to_string())
    })?;
    Ok(kernel_desc)
}

#[derive(Clone, Copy, Debug)]
enum SpirvOptKind {
    DeadCodeElimination,
    Performance,
}

fn spirv_opt(spirv: &[u32], kind: SpirvOptKind) -> Result<Vec<u32>> {
    use spirv_tools::{
        opt::{Optimizer, Passes},
        val::Validator,
        TargetEnv,
    };
    let target_env = TargetEnv::Vulkan_1_2;
    let validator = spirv_tools::val::create(Some(target_env));
    validator.validate(spirv, None)?;
    let mut optimizer = spirv_tools::opt::create(Some(target_env));
    match kind {
        SpirvOptKind::DeadCodeElimination => {
            let passes = {
                use Passes::*;
                [
                    EliminateDeadFunctions,
                    DeadVariableElimination,
                    EliminateDeadConstant,
                    CombineAccessChains,
                    //CFGCleanup,
                    CompactIds,
                ]
            };
            for pass in passes {
                optimizer.register_pass(pass);
            }
        }
        SpirvOptKind::Performance => {
            /*let passes = {
                use Passes::*;
                [
                    DeadBranchElim,
                    MergeReturn,
                    PrivateToLocal,
                    LocalMultiStoreElim,
                    ConditionalConstantPropagation,
                    DeadBranchElim,
                    Simplification,
                    LocalSingleStoreElim,
                    IfConversion,
                    Simplification,
                    AggressiveDCE,
                    DeadBranchElim,
                    BlockMerge,
                    LocalAccessChainConvert,
                    LocalSingleBlockLoadStoreElim,
                    AggressiveDCE,
                    CopyPropagateArrays,
                    VectorDCE,
                    DeadInsertElim,
                    EliminateDeadMembers,
                    LocalSingleStoreElim,
                    BlockMerge,
                    LocalMultiStoreElim,
                    RedundancyElimination,
                    Simplification,
                    AggressiveDCE,
                    CFGCleanup,
                    CompactIds,
                ]
            };
            for pass in passes {
                optimizer.register_pass(pass);
            }*/
            optimizer.register_performance_passes();
            //optimizer.register_pass(Passes::LoopPeeling);
        }
    }
    //optimizer.register_performance_passes();
    let spirv = optimizer
        .optimize(spirv, &mut |_| (), None)?
        .as_words()
        .to_vec();
    Ok(spirv)
}

fn add_spec_constant_ops(module: &mut rspirv::dr::Module) {
    use rspirv::{
        dr::{Instruction, Operand},
        spirv::Op,
    };
    use std::collections::HashSet;
    let mut constants = HashSet::new();
    for inst in module.types_global_values.iter() {
        if matches!(
            inst.class.opcode,
            Op::Constant
                | Op::ConstantTrue
                | Op::ConstantFalse
                | Op::ConstantNull
                | Op::ConstantComposite
                | Op::SpecConstant
                | Op::SpecConstantTrue
                | Op::SpecConstantFalse
                | Op::SpecConstantComposite
                | Op::SpecConstantOp
        ) {
            if let Some(result_id) = inst.result_id {
                constants.insert(result_id);
            }
        }
    }
    for function in module.functions.iter_mut() {
        for block in function.blocks.iter_mut() {
            block.instructions.retain(|inst| {
                if matches!(
                    inst.class.opcode,
                    Op::SConvert
                        | Op::UConvert
                        | Op::FConvert
                        | Op::SNegate
                        | Op::Not
                        | Op::IAdd
                        | Op::ISub
                        | Op::IMul
                        | Op::UDiv
                        | Op::SDiv
                        | Op::UMod
                        | Op::SRem
                        | Op::SMod
                        | Op::ShiftRightLogical
                        | Op::ShiftRightArithmetic
                        | Op::ShiftLeftLogical
                        | Op::BitwiseOr
                        | Op::BitwiseAnd
                        | Op::VectorShuffle
                        | Op::CompositeExtract
                        | Op::CompositeInsert
                        | Op::LogicalOr
                        | Op::LogicalAnd
                        | Op::LogicalNot
                        | Op::LogicalEqual
                        | Op::LogicalNotEqual
                        | Op::Select
                        | Op::IEqual
                        | Op::INotEqual
                        | Op::ULessThan
                        | Op::SLessThan
                        | Op::UGreaterThan
                        | Op::SGreaterThan
                        | Op::ULessThanEqual
                        | Op::SLessThanEqual
                        | Op::UGreaterThanEqual
                        | Op::SGreaterThanEqual
                        | Op::QuantizeToF16
                ) {
                    if let Some(result_id) = inst.result_id {
                        let mut used_constants = HashSet::new();
                        for operand in inst.operands.iter() {
                            if let Operand::IdRef(id) = operand {
                                if !constants.contains(id) {
                                    return true;
                                }
                                used_constants.insert(*id);
                            }
                        }
                        for (i, global_inst) in module.types_global_values.iter().enumerate().rev()
                        {
                            if let Some(global_result_id) = global_inst.result_id {
                                if inst.result_type == global_inst.result_id
                                    || used_constants.contains(&global_result_id)
                                {
                                    module.types_global_values.insert(
                                        i + 1,
                                        Instruction::new(
                                            Op::SpecConstantOp,
                                            inst.result_type,
                                            inst.result_id,
                                            [Operand::LiteralInt32(inst.class.opcode as u32)]
                                                .into_iter()
                                                .chain(inst.operands.clone())
                                                .collect(),
                                        ),
                                    );
                                    constants.insert(result_id);
                                    return false;
                                }
                            }
                        }
                    }
                }
                true
            });
        }
    }
}
/*
fn unroll_loops(module: &mut rspirv::dr::Module) {
    use rspirv::{
        dr::Operand,
        spirv::{LoopControl, Op},
    };
    for func in module.functions.iter_mut() {
        for block in func.blocks.iter_mut() {
            for inst in block.instructions.iter_mut() {
                if inst.class.opcode == Op::LoopMerge {
                    let loop_control = inst.operands[2].unwrap_loop_control();
                    if loop_control == LoopControl::NONE {
                        inst.operands[2] = Operand::LoopControl(LoopControl::UNROLL);
                    }
                }
            }
        }
    }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScalarType {
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl ScalarType {
    fn iter() -> impl Iterator<Item = Self> {
        use ScalarType::*;
        [U8, I8, U16, I16, F16, BF16, U32, I32, F32, U64, I64, F64].into_iter()
    }
    fn name(&self) -> &'static str {
        use ScalarType::*;
        match self {
            U8 => "u8",
            I8 => "i8",
            U16 => "u16",
            I16 => "i16",
            F16 => "f16",
            BF16 => "bf16",
            U32 => "u32",
            I32 => "i32",
            F32 => "f32",
            U64 => "u64",
            I64 => "i64",
            F64 => "F64",
        }
    }
    fn as_str(&self) -> &'static str {
        use ScalarType::*;
        match self {
            U8 => "U8",
            I8 => "I8",
            U16 => "U16",
            I16 => "I16",
            F16 => "F16",
            BF16 => "BF16",
            U32 => "U32",
            I32 => "I32",
            F32 => "F32",
            U64 => "U64",
            I64 => "I64",
            F64 => "F64",
        }
    }
    /*fn size(&self) -> usize {
        use ScalarType::*;
        match self {
            U8 | I8 => 1,
            U16 | I16 | F16 | BF16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }
    fn signed(&self) -> bool {
        use ScalarType::*;
        matches!(self, I8 | I16 | I32 | I64)
    }*/
}

impl FromStr for ScalarType {
    type Err = ();
    fn from_str(input: &str) -> Result<Self, ()> {
        Self::iter()
            .find(|x| x.as_str() == input || x.name() == input)
            .ok_or(())
    }
}

impl Serialize for ScalarType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for ScalarType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Visitor;

        struct ScalarTypeVisitor;

        impl Visitor<'_> for ScalarTypeVisitor {
            type Value = ScalarType;
            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(formatter, "a scalar type")
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if let Ok(scalar_type) = ScalarType::from_str(v) {
                    Ok(scalar_type)
                } else {
                    Err(E::custom(format!("unknown ScalarType {v}")))
                }
            }
        }
        deserializer.deserialize_str(ScalarTypeVisitor)
    }
}

// must match krnl_macros defs!

#[derive(Serialize, Deserialize, Debug)]
struct KernelDesc {
    name: String,
    //hash: u64,
    spirv: Vec<u32>,
    features: Features,
    safe: bool,
    spec_descs: Vec<SpecDesc>,
    slice_descs: Vec<SliceDesc>,
    push_descs: Vec<PushDesc>,
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct Features {
    shader_int8: bool,
    shader_int16: bool,
    shader_int64: bool,
    shader_float16: bool,
    shader_float64: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct SpecDesc {
    name: String,
    scalar_type: ScalarType,
}

#[derive(Serialize, Deserialize, Debug)]
struct SliceDesc {
    name: String,
    scalar_type: ScalarType,
    mutable: bool,
    item: bool,
}

/*
#[derive(Serialize, Deserialize, Debug)]
struct ArrayDesc {
    name: String,
    storage_class: ArrayStorageClass,
    mutable: bool,
    len: usize,
    spec_id: Option<u32>,
}

pub enum ArrayStorageClass {
    Group,
    Subgroup,
    Thread,
}
*/

#[derive(Serialize, Deserialize, Debug)]
struct PushDesc {
    name: String,
    scalar_type: ScalarType,
}

#[derive(Serialize, serde::Deserialize, Debug)]
struct KrnlcCache {
    version: String,
    kernels: Vec<KernelDesc>,
}
