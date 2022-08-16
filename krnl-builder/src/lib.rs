use krnl_core::__private::{
    KRNL_MODULE_PATH,
    serde::Serialize,
    bincode,
    raw_module::{RawModule, Target}};
use std::{path::{Path, PathBuf}, process::Command, fs, hash_map::DefaultHasher};
use anyhow::{anyhow, bail};
use spirv_builder::SpirvBuilder;

type Result<T, E = anhow::error> = Result<T, E>;

pub mod builder {
    pub struct ModuleBuilder {
        pub(super) crate_path: PathBuf,
        pub(super) target: Option<Target>,
    }
}
use builder::ModuleBuilder;

pub struct Module {
    raw: RawModule,
}

impl Module {
    pub fn builder(crate_path: impl AsRef<Path>) -> ModuleBuilder {
        ModuleBuilder {
            crate_path: crate_path.as_ref().to_owned(),
            target: None,
        }
    }
}

impl ModuleBuilder {
    pub fn vulkan(mut self, version: (u32, u32)) -> Self {
        self.target.replace(Target::Vulkan(version.0, version.1));
    }
    pub fn build(self) -> Result<Module> {
        use std::env;
        let ModuleBuilder {
            crate_path,
            target,
        } = self.raw;
        let target = target.unwrap_or_else(|| anyhow!("No target specified! Hint: Use .vulkan() to set the vulkan version."))?;
        if env::var(KRNL_MODULE_PATH).is_ok() {
            bail!("Cannot build while building another module!");
        }
        let source = fs::read(manifest_dir.join("src").join("lib.rs"))?;
        let source_hash = {
            let mut h = DefaultHasher::new();
            source.hash(&mut h);
            h.finish()
        };
        let name = crate_path.file_stem().ok_or_else(|| anyhow!("`crate_path` is empty!"))?
            .to_string_lossy()
            .into_owned();
        let name_with_hash = format!("{name}{source_hash}");
        let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
        let target_dir = PathBuf::from(manifest_dir).join("target");
        let krnl_dir = target_dir.join(".krnl");
        let krnl_module_path = krnl_dir.join(name_with_hash).with_extension(".bincode");
        let krnl_module_path_string = krnl_module_path.to_string_lossy();
        fs::create_dir_all(&krnl_dir);
        let raw_module = RawModule {
            source,
            name,
            target,
            kernels: Default::default(),
        };
        dbg!(krnl_module_path);
        panic!();
        fs::write(krnl_module_path, bincode::serialize(&raw_module)?)?;
        let status = Command::new("cargo check")
            .current_dir(&crate_path)
            .env(KRNL_MODULE_PATH, &krnl_module_path_string)
            .status()?;
        if !status.is_success() {
            bail!("cargo check failed!");
        }
        todo!()
    }
}
