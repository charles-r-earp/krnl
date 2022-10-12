use std::{env::args, path::PathBuf, fs, collections::HashMap};
use spirv_builder::{SpirvBuilder, MetadataPrintout};

mod kernel_info;
use kernel_info::{CompileOptions, KernelInfo, Spirv};

fn main() {
    let mut args = args().skip(1);
    let crate_dir = PathBuf::from(args.next().unwrap());
    let kernel_infos_path = crate_dir.join("kernel_infos.bincode");
    let kernel_infos: HashMap<CompileOptions, Vec<KernelInfo>> = bincode::deserialize(&fs::read(&kernel_infos_path).unwrap()).unwrap();
    let spirvs_path = crate_dir.join(format!("spirvs.bincode"));
    let mut spirvs = HashMap::with_capacity(kernel_infos.values().map(|x| x.len()).sum());
    for (compile_options, kernel_infos) in kernel_infos.iter() {
        let vulkan = compile_options.vulkan;
        let target = format!("spirv-unknown-vulkan{}.{}", vulkan.0, vulkan.1);
        let mut builder = SpirvBuilder::new(&crate_dir, &target)
            .multimodule(true)
            .print_metadata(MetadataPrintout::DependencyOnly)
            .deny_warnings(true)
            .preserve_bindings(true);
        for cap in compile_options.capabilities.iter().copied() {
            builder = builder.capability(cap);
        }
        for ext in compile_options.extensions.iter().cloned() {
            builder = builder.extension(ext);
        }
        let module_result = builder.build().unwrap().module;
        let spirv_paths = module_result.unwrap_multi();
        for kernel_info in kernel_infos.iter() {
            let spirv_path = spirv_paths.get(&kernel_info.name).unwrap();
            let spirv_bytes = fs::read(spirv_path).unwrap();
            let words = process_spirv(&spirv_bytes, kernel_info);
            spirvs.insert(kernel_info.name.clone(), Spirv { words });
        }
    }
    fs::write(&spirvs_path, bincode::serialize(&spirvs).unwrap()).unwrap();
}

fn process_spirv(spirv: &[u8], kernel_info: &KernelInfo) -> Vec<u32> {
    use rspirv::{
        binary::{Assemble, Parser},
        dr::{Builder, Loader},
    };
    use spirv_tools::val::Validator;
    let mut loader = Loader::new();
    Parser::new(spirv, &mut loader)
        .parse()
        .unwrap();
    let mut builder = Builder::new_from_module(loader.module());
    replace_array_length(&mut builder, kernel_info);
    let module = builder.module();
    let words = module.assemble();
    spirv_tools::val::create(None).validate(&words, None).unwrap();
    words
}

fn replace_array_length(
    builder: &mut rspirv::dr::Builder,
    kernel_info: &KernelInfo,
) {
    use rspirv::dr::{Instruction, Operand::*};
    use spirv::{Op, StorageClass, Word};
    let num_slices = kernel_info.slice_infos().count();
    if num_slices == 0 {
        return;
    }
    let module = builder.module_ref();
    let mut func_block_inst_indices =
        Vec::<(usize, usize, usize)>::with_capacity(num_slices);
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
        return;
    }
    let mut buffer_ids = vec![None; num_slices];
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
    let mut push_offset_ids: Vec<Option<Word>> = vec![None; num_slices];
    for (slice_info, offset_id) in kernel_info
        .slice_infos()
        .zip(push_offset_ids.iter_mut())
    {
        let push_info = kernel_info
            .push_infos()
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
}
