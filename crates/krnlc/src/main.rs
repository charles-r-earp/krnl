use anyhow::{bail, format_err, Result};
use cargo_metadata::{Metadata, Package, PackageId};
use clap::Parser;
use clap_cargo::{Manifest, Workspace};
use parking_lot::Mutex;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use spirv_builder::{MetadataPrintout, SpirvBuilder, SpirvMetadata};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str::FromStr,
    sync::Arc,
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
    let cache_manifest_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
    {
        let cache_manifest_path = cache_manifest_path.clone();
        ctrlc::set_handler(move || {
            if let Some(path) = cache_manifest_path.lock().take() {
                let result = cache(&path, Some(Default::default()));
                if let Err(e) = result {
                    eprintln!("{e:?}");
                }
            }
        })?;
    }

    let cli = Cli::parse();
    let metadata = cli.manifest.metadata().exec()?;
    let (selected, _) = cli.workspace.partition_packages(&metadata);
    for package in selected {
        let (features, dependencies) = extract_features_dependencies(&metadata, &package)?;
        let manifest_path = &package.manifest_path;
        let result = (|| -> Result<()> {
            {
                let mut cache_manifest_path = cache_manifest_path.lock();
                cache_manifest_path.replace(manifest_path.clone().into());
                cache(manifest_path.as_std_path(), None)?;
            }
            cargo_check(&package, &features)?;
            let module_datas = cargo_expand(&package, &features)?;
            let modules = compile(&package, &dependencies, module_datas)?;
            cache(manifest_path.as_std_path(), Some(modules))?;
            cache_manifest_path.lock().take();
            cargo_check(&package, &features)?;
            Ok(())
        })();
        if let Err(e) = result {
            cache(manifest_path.as_std_path(), Some(Default::default()))?;
            return Err(e.into());
        }
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

fn cargo_check(package: &Package, features: &str) -> Result<()> {
    let mut command = Command::new("cargo");
    command.args(["check", "--manifest-path", package.manifest_path.as_str()]);
    if !features.is_empty() {
        command.args(["--features", &features]);
    }
    command.stderr(Stdio::inherit());
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(format_err!("cargo check failed!"))
    }
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

fn cache(
    manifest_path: &Path,
    modules: Option<HashMap<String, HashMap<String, KernelDesc>>>,
) -> Result<()> {
    let version = env!("CARGO_PKG_VERSION");
    let cache = KrnlcCache {
        version: version.to_string(),
        modules,
    };
    let cache = bincode::serialize(&cache)?;
    let cache = format!(
        r#"macro_rules! __krnl_cache {{
    ($m:ident) => {{ 
        __krnl::krnl_macros::__krnl_module!($m, {cache:?});
    }};
}}"#
    );
    let mut cache = prettyplease::unparse(&syn::parse_str(&cache)?);
    cache.insert_str(0, &format!("/* generated by krnlc {version} */\n"));
    std::fs::write(manifest_path.parent().unwrap().join("krnl-cache.rs"), cache)?;
    Ok(())
}

fn compile(
    package: &Package,
    dependencies: &str,
    module_datas: HashMap<String, ModuleData>,
) -> Result<HashMap<String, HashMap<String, KernelDesc>>> {
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
    let device_crate_manifest_path = device_crate_dir.join("Cargo.toml");
    let mut update = false;
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
        let mut source = r#"#![cfg_attr(target_arch = "spirv",
no_std,
feature(asm_experimental_arch),
)]

extern crate krnl_core;
"#
        .to_string();
        for (name, module) in module_datas.iter() {
            writeln!(&mut source, "pub mod {name} {{ {} }}", module.source).unwrap();
        }
        let file = syn::parse_str(&source)?;
        let source = prettyplease::unparse(&file);
        std::fs::write(src_dir.join("lib.rs"), source.as_bytes())?;
    }
    if update {
        let status = Command::new("cargo")
            .args(&[
                "update",
                "--manifest-path",
                device_crate_manifest_path.to_string_lossy().as_ref(),
            ])
            .status()?;
        if !status.success() {
            bail!("cargo update failed!");
        }
    }
    let crate_name_ident = crate_name.replace("-", "_");
    let mut modules =
        HashMap::<String, HashMap<String, KernelDesc>>::with_capacity(module_datas.len());
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
            //.preserve_bindings(true)
            .deny_warnings(true)
            .extension("SPV_KHR_non_semantic_info");
        for cap in capabilites {
            builder = builder.capability(cap);
        }
        let output = builder.build()?;
        let spirv_modules = output.module.unwrap_multi();
        for (entry_point, spirv_path) in spirv_modules.iter() {
            use rspirv::{
                binary::Assemble,
                dr::{Instruction, Operand},
                spirv::{Decoration, Op},
            };
            let spirv = std::fs::read(spirv_path)?;
            let mut spirv_module = rspirv::dr::load_bytes(&spirv).unwrap();
            /*{
                use rspirv::binary::Disassemble;
                eprintln!("{}", spirv_module.disassemble());
            }*/
            let (module_name_with_hash, kernel_name) = entry_point.split_once("::").unwrap();
            let module_data = module_datas.get(module_name_with_hash).unwrap();
            let module_path = &module_data.path;
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
                    bail!("Unable to decode kernel {module_path}::{kernel_name}!");
                };
                /*spirv_module.annotations.retain(|inst| {
                    let op = inst.class.opcode;
                    !(op == Op::Decorate
                        && inst.operands.first() == Some(&Operand::IdRef(kernel_data_var)))
                });*/
                /*let mut kernel_data_ptrs = std::collections::HashSet::new();
                for function in spirv_module.functions.iter_mut() {
                    for block in function.blocks.iter_mut() {
                        for inst in block.instructions.iter_mut() {
                            let op = inst.class.opcode;
                            if op == Op::AccessChain
                                && inst.operands.first().unwrap().unwrap_id_ref() == kernel_data_var
                            {
                                kernel_data_ptrs.insert(inst.result_id.unwrap());
                                *inst = Instruction::new(Op::Nop, None, None, Vec::new());
                            }
                        }
                    }
                }
                for function in spirv_module.functions.iter_mut() {
                    for block in function.blocks.iter_mut() {
                        for inst in block.instructions.iter_mut() {
                            let op = inst.class.opcode;
                            if op == Op::Store {
                                if kernel_data_ptrs
                                    .contains(&inst.operands.first().unwrap().unwrap_id_ref())
                                {
                                    *inst = Instruction::new(Op::Nop, None, None, Vec::new());
                                }
                            }
                        }
                    }
                }*/
                let mut kernel_data = None;
                for (i, inst) in spirv_module.debug_names.iter().enumerate() {
                    let op = inst.class.opcode;
                    if op == Op::Name
                        && inst.operands.first().unwrap().unwrap_id_ref() == kernel_data_var
                    {
                        kernel_data.replace(inst.operands[1].unwrap_literal_string().to_string());
                        //spirv_module.debug_names.remove(i);
                        break;
                    }
                }
                let kernel_data = kernel_data.unwrap();
                /*for (i, inst) in spirv_module.types_global_values.iter().enumerate() {
                    if inst.result_id == Some(kernel_data_var) {
                        spirv_module.types_global_values.remove(i);
                        break;
                    }
                }*/
                let kernel_data =
                    if let Some(kernel_data) = kernel_data.strip_prefix("__krnl_kernel_data_") {
                        kernel_data
                    } else {
                        todo!()
                    };
                let mut bytes = Vec::with_capacity(kernel_data.len() / 2);
                let mut iter = kernel_data.chars();
                while let Some((a, b)) = iter.next().zip(iter.next()) {
                    let byte = a
                        .to_digit(16)
                        .unwrap()
                        .checked_mul(16)
                        .unwrap()
                        .checked_add(b.to_digit(16).unwrap())
                        .unwrap()
                        .try_into()
                        .unwrap();
                    bytes.push(byte);
                }
                bincode::deserialize(&bytes)?
            };
            spirv_module.entry_points.first_mut().unwrap().operands[2] =
                Operand::LiteralString("main".to_string());
            kernel_desc.name = format!("{crate_name_ident}::{module_path}::{kernel_name}");
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
            /*{
                use rspirv::binary::Disassemble;
                eprintln!("{}", spirv_module.disassemble());
            }*/
            let spirv = spirv_module.assemble();
            kernel_desc.spirv = {
                use spirv_tools::{
                    opt::{Optimizer, Passes},
                    val::Validator,
                    TargetEnv,
                };
                let target_env = TargetEnv::Vulkan_1_2;
                let validator = spirv_tools::val::create(Some(target_env));
                validator.validate(&spirv, None)?;
                let mut optimizer = spirv_tools::opt::create(Some(target_env));
                optimizer
                    .register_pass(Passes::StripNonSemanticInfo)
                    .register_pass(Passes::StripDebugInfo)
                    .register_pass(Passes::RemoveUnusedInterfaceVariables)
                    .register_performance_passes();
                optimizer
                    .optimize(&spirv, &mut |_| (), None)?
                    .as_words()
                    .to_vec()
            };
            {
                use rspirv::binary::Disassemble;
                /*eprintln!(
                    "{}",
                    rspirv::dr::load_words(&kernel_desc.spirv)
                        .unwrap()
                        .disassemble()
                );*/
            }
            modules
                .entry(module_data.name_with_hash.clone())
                .or_default()
                .insert(kernel_name.to_string(), kernel_desc);
        }
    }
    Ok(modules)
}

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
            U16 => "I16",
            I16 => "U16",
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
    spirv: Vec<u32>,
    features: Features,
    threads: Vec<u32>,
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
    thread_dim: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SliceDesc {
    name: String,
    scalar_type: ScalarType,
    mutable: bool,
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

#[derive(Serialize, Debug)]
struct KrnlcCache {
    version: String,
    modules: Option<HashMap<String, HashMap<String, KernelDesc>>>,
}
