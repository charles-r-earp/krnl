use anyhow::{format_err, bail};
use krnl_types::{
    version::Version,
    kernel::Module,
    __private::raw_module::{RawModule, Spirv},
};
use spirv::Capability;
use spirv_builder::{MetadataPrintout, SpirvBuilder};
use std::{
    collections::{hash_map::DefaultHasher, HashSet},
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

#[doc(inline)]
pub use krnl_types::kernel;
#[doc(inline)]
pub use krnl_types::version;

type Result<T, E = anyhow::Error> = std::result::Result<T, E>;

fn target_from_version(vulkan_version: Version) -> String {
    let Version {
        major,
        minor,
        patch
    } = vulkan_version;
    if patch == 0 {
        format!("spirv-unknown-vulkan{major}.{minor}")
    } else {
        format!("spirv-unknown-vulkan{major}.{minor}.{patch}")
    }
}

pub struct ModuleBuilder {
    crate_path: PathBuf,
    vulkan_version: Option<Version>,
}

impl ModuleBuilder {
    pub fn new(crate_path: impl AsRef<Path>) -> Self {
        let crate_path = crate_path.as_ref().to_owned();
        ModuleBuilder {
            crate_path,
            vulkan_version: None,
        }
    }
    pub fn vulkan(mut self, vulkan_version: Version) -> Self {
        self.vulkan_version.replace(vulkan_version);
        self
    }
    pub fn build(self) -> Result<Module> {
        let crate_path = self.crate_path.canonicalize()?;
        let vulkan_version = self.vulkan_version.ok_or_else(|| {
            format_err!("No vulkan version specified! Use `.vulkan()` to set the version of vulkan to use, for example `.vulkan(Version::from_major_minor(1, 1))`.")
        })?;
        let target = target_from_version(vulkan_version);
        let crate_path_hash = {
            let mut h = DefaultHasher::new();
            crate_path.hash(&mut h);
            h.finish()
        };
        let name = crate_path
            .file_stem()
            .ok_or_else(|| format_err!("`crate_path` is empty!"))?
            .to_string_lossy()
            .into_owned();
        let name_with_hash = format!("{name}{crate_path_hash}");
        let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
        let krnl_dir = PathBuf::from(manifest_dir).join("target").join(".krnl");
        let target_dir = krnl_dir.join("target").join(&target).join(&name_with_hash);
        fs::create_dir_all(&target_dir)?;
        let modules_dir = krnl_dir.join("modules");
        fs::create_dir_all(&modules_dir)?;
        let module_path = modules_dir.join(&name_with_hash).with_extension("bincode");
        let saved_modules_dir = krnl_dir.join("saved-modules");
        fs::create_dir_all(&saved_modules_dir)?;
        let saved_module_path = saved_modules_dir.join(&name_with_hash).with_extension("bincode");
        let raw_module = RawModule {
            name,
            vulkan_version,
            kernels: Default::default(),
        };
        let saved_module: Option<RawModule> = if saved_module_path.exists() {
            let bytes = fs::read(&saved_module_path)?;
            Some(bincode::deserialize(&bytes)?)
        } else {
            None
        };
        {
            let bytes = bincode::serialize(&raw_module)?;
            fs::write(&module_path, &bytes)?;
        }
        let status = Command::new("cargo")
            .args(&[
                "check",
                "--manifest-path",
                &*crate_path.join("Cargo.toml").to_string_lossy(),
                "--target-dir",
                &*target_dir.to_string_lossy(),
            ])
            .env("KRNL_MODULE_PATH", &*module_path.to_string_lossy())
            .status()?;
        if !status.success() {
            bail!("cargo check failed!");
        }
        let bytes = fs::read(&module_path)?;
        let mut raw_module: RawModule = bincode::deserialize(&bytes)?;
        if !raw_module.kernels.is_empty() {
            {
                let bytes = bincode::serialize(&raw_module)?;
                fs::write(&saved_module_path, &bytes)?;
            }
            let mut compile_options_set = HashSet::with_capacity(raw_module.kernels.len());
            for kernel in raw_module.kernels.values() {
                let compile_options = CompileOptions {
                    vulkan_version: kernel.vulkan_version,
                    capabilities: kernel.capabilities.clone(),
                    extensions: kernel.extensions.clone(),
                };
                compile_options_set.insert(compile_options);
            }
            for options in compile_options_set.iter() {
                let target = target_from_version(options.vulkan_version);
                let mut builder =
                    SpirvBuilder::new(&crate_path, &target)
                        .multimodule(true)
                        .print_metadata(MetadataPrintout::None);
                for cap in options.capabilities.iter().copied() {
                    builder = builder.capability(cap);
                }
                for ext in options.extensions.iter().cloned() {
                    builder = builder.extension(ext);
                }
                for (entry, spv_path) in builder.build()?.module.unwrap_multi() {
                    let spv = fs::read(spv_path)?;
                    if let Some(mut kernel) = raw_module.kernels.get_mut(entry) {
                        let kernel = Arc::get_mut(&mut kernel).unwrap();
                        kernel.spirv.replace(Spirv {
                            words: bytemuck::cast_slice(&spv).to_vec(),
                        });
                    } else {
                        bail!("Found unexpected entry_point {entry:?}!");
                    }
                }
            }
            Ok(Module::__from_raw(Arc::new(raw_module)))
        } else {
            Ok(Module::__from_raw(Arc::new(saved_module.unwrap_or(raw_module))))
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct CompileOptions {
    vulkan_version: Version,
    capabilities: Vec<Capability>,
    extensions: Vec<String>,
}
