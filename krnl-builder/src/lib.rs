use anyhow::{bail, format_err};
use krnl_types::{
    kernel::{CompileOptions, KernelInfoInner, Module, ModuleInner, Spirv},
    version::Version,
};
use spirv_builder::{MetadataPrintout, SpirvBuilder};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
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
        patch,
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
        let saved_module_path = saved_modules_dir
            .join(&name_with_hash)
            .with_extension("bincode");
        let module = ModuleInner {
            name,
            kernels: Default::default(),
        };
        let saved_module: Option<ModuleInner> = if saved_module_path.exists() {
            let bytes = fs::read(&saved_module_path)?;
            Some(bincode::deserialize(&bytes)?)
        } else {
            None
        };
        {
            let bytes = bincode::serialize(&module)?;
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
            .env("KRNL_VULKAN_VERSION", &vulkan_version.to_string())
            .status()?;
        if !status.success() {
            bail!("cargo check failed!");
        }
        let bytes = fs::read(&module_path)?;
        let mut module: ModuleInner = bincode::deserialize(&bytes)?;
        if !module.kernels.is_empty() {
            {
                let bytes = bincode::serialize(&module)?;
                fs::write(&saved_module_path, &bytes)?;
            }
            let mut compile_options_map =
                HashMap::<CompileOptions, Vec<String>>::with_capacity(module.kernels.len());
            for kernel_info in module.kernels.values() {
                compile_options_map
                    .entry(kernel_info.compile_options())
                    .or_default()
                    .push(kernel_info.name.clone());
            }
            for (options, kernel_names) in compile_options_map.iter() {
                let target = target_from_version(options.vulkan_version);
                let mut builder = SpirvBuilder::new(&crate_path, &target)
                    .multimodule(true)
                    .print_metadata(MetadataPrintout::None);
                for cap in options.capabilities.iter().copied() {
                    builder = builder.capability(cap);
                }
                for ext in options.extensions.iter().cloned() {
                    builder = builder.extension(ext);
                }

                let module_result = {
                    let mut kernels = String::new();
                    for name in kernel_names {
                        kernels.push_str(name);
                        kernels.push(',');
                    }
                    std::env::set_var("KRNL_KERNELS", kernels);
                    let result = std::panic::catch_unwind(|| builder.build());
                    std::env::remove_var("KRNL_KERNELS");
                    match result {
                        Ok(x) => x?,
                        Err(e) => std::panic::resume_unwind(e),
                    }
                };
                let spv_paths = module_result.module.unwrap_multi();
                for name in kernel_names.iter() {
                    if let Some(spv_path) = spv_paths.get(name) {
                        let kernel_info =
                            Arc::get_mut(module.kernels.get_mut(name).unwrap()).unwrap();
                        let data = fs::read(spv_path)?;
                        process_spirv(&data, kernel_info)?;
                    } else {
                        dbg!(module.kernels.get(name));
                        dbg!(&options);
                        dbg!(&spv_paths);
                        bail!(
                            "Expected spv_path for kernel {name:?} in module {:?}!",
                            module.name
                        );
                    }
                }
            }
            Ok(module.into_module())
        } else {
            Ok(saved_module.unwrap_or(module).into_module())
        }
    }
}

fn process_spirv(data: &[u8], kernel_info: &mut KernelInfoInner) -> Result<()> {
    use rspirv::{
        binary::{Assemble, Parser},
        dr::{Builder, Loader},
    };
    use spirv_tools::val::Validator;
    let mut loader = Loader::new();
    Parser::new(data, &mut loader)
        .parse()
        .map_err(|x| anyhow::Error::msg(x.to_string()))?;
    let mut builder = Builder::new_from_module(loader.module());
    replace_array_length(&mut builder, kernel_info)?;
    let module = builder.module();
    let words = module.assemble();
    spirv_tools::val::create(None).validate(&words, None)?;
    kernel_info.spirv.replace(Spirv { words });
    Ok(())
}

fn replace_array_length(
    builder: &mut rspirv::dr::Builder,
    kernel_info: &mut KernelInfoInner,
) -> Result<()> {
    use rspirv::dr::{Instruction, Operand::*};
    use spirv::{Op, StorageClass, Word};
    if kernel_info.slice_infos.is_empty() {
        return Ok(());
    }
    let module = builder.module_ref();
    let mut func_block_inst_indices =
        Vec::<(usize, usize, usize)>::with_capacity(kernel_info.slice_infos.len());
    for (f, func) in module.functions.iter().enumerate() {
        for (b, block) in func.blocks.iter().enumerate() {
            for (i, inst) in block.instructions.iter().enumerate() {
                if inst.class.opcode == Op::ArrayLength {
                    func_block_inst_indices.push((f, b, i));
                }
            }
        }
    }
    if func_block_inst_indices.is_empty() {
        return Ok(());
    }
    let mut buffer_ids = vec![None; kernel_info.slice_infos.len()];
    for inst in module.annotations.iter() {
        if let [IdRef(id), Decoration(spirv::Decoration::Binding), LiteralInt32(binding)] =
            inst.operands.as_slice()
        {
            buffer_ids[*binding as usize].replace(*id);
        }
    }
    let mut push_consts_id: Option<Word> = None;
    for inst in module.types_global_values.iter() {
        if inst.class.opcode == Op::Variable {
            if let [StorageClass(spirv::StorageClass::PushConstant)] = inst.operands.as_slice() {
                push_consts_id.replace(inst.result_id.unwrap());
                break;
            }
        }
    }
    let ty_int = builder.type_int(32, 0);
    let zero = builder.constant_u32(ty_int, 0);
    let ty_int_ptr = builder.type_pointer(None, StorageClass::PushConstant, ty_int);
    let mut push_offset_ids: Vec<Option<Word>> = vec![None; buffer_ids.len()];
    for (slice_info, offset_id) in kernel_info
        .slice_infos
        .iter()
        .zip(push_offset_ids.iter_mut())
    {
        let push_info = kernel_info
            .push_infos
            .iter()
            .find(|x| x.name.starts_with("__krnl_len_") && x.name.ends_with(&slice_info.name))
            .unwrap();
        offset_id.replace(builder.constant_u32(ty_int, push_info.offset));
    }
    let push_consts_id = if let Some(push_consts_id) = push_consts_id {
        push_consts_id
    } else {
        todo!()
    };
    for (f, b, i) in func_block_inst_indices.iter().copied() {
        let access_id = builder.id();
        let instructions = &mut builder.module_mut().functions[f].blocks[b].instructions;
        let inst = &mut instructions[i];
        let id = if let Some(IdRef(id)) = inst.operands.get(0) {
            id
        } else {
            unreachable!("{:?}", inst)
        };
        let index = buffer_ids
            .iter()
            .position(|x| x.as_ref() == Some(id))
            .unwrap();
        let offset_id = push_offset_ids[index].unwrap();
        let access_chain = Instruction::new(
            Op::AccessChain,
            Some(ty_int_ptr),
            Some(access_id),
            vec![IdRef(push_consts_id), IdRef(zero), IdRef(offset_id)],
        );
        let load = Instruction::new(
            Op::Load,
            Some(ty_int),
            inst.result_id,
            vec![IdRef(access_id)],
        );
        *inst = load;
        instructions.insert(i, access_chain);
    }
    Ok(())
}
