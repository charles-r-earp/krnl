use crate::reflect::{Features, KernelDesc};
use crate::spirv::{
    assemble, get_element_size, krnl_inst_set, op_constant, op_decorate_block,
    op_member_decorate_offset, op_member_name, op_type_int, op_type_pointer, op_type_struct,
    pointee_type, struct_element_type, validate, variable_name,
};
use camino::{Utf8Path, Utf8PathBuf};
use fxhash::{FxBuildHasher, FxHashSet};
use indexmap::{
    map::{MutableEntryKey, MutableKeys},
    IndexMap, IndexSet,
};
use krnl_core::__private::__KrnlInst as KrnlInst;
use num_traits::FromPrimitive;
use smallvec::SmallVec;
use spirt::{
    print::Plan,
    spv::{
        encode_literal_string, extract_literal_string,
        spec::{
            raw::OperandKind, ExtInstSetDesc, ExtInstSetInstructionDesc, RestOperandsUnit, Spec,
        },
        Imm, Inst,
    },
    transform::{InnerInPlaceTransform, Transformer},
    visit::{InnerVisit, Visit, Visitor},
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstForm, DataInstFormDef, DataInstKind, DeclDef, ExportKey, Exportee, Func,
    GlobalVar, GlobalVarDecl, GlobalVarDefBody, InternedStr, Module, ModuleDialect, Type, TypeDef,
    TypeKind, TypeOrConst, Value,
};
use spirv_headers::{Capability, Decoration, ExecutionMode, ExecutionModel, StorageClass};
use spirv_tools::{
    binary::Binary,
    opt::{Optimizer, Options as OptimizerOptions, Passes},
    val::Validator,
    TargetEnv,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

pub struct CompiledKernel {
    pub spirv: Vec<u32>,
    pub desc: KernelDesc,
    pub features: Features,
}

pub fn compile(spirv: Vec<u8>, spec_constants: &BTreeMap<String, [u32; 2]>) -> CompiledKernel {
    let context = Rc::new(Context::new());
    context.register_custom_ext_inst_set(KrnlInst::SET_NAME, krnl_inst_set());
    let mut module = Module::lower_from_spv_bytes(context.clone(), spirv).unwrap();
    let func = module
        .exports
        .values()
        .nth(0)
        .map(|x| {
            if let Exportee::Func(x) = x {
                *x
            } else {
                unreachable!()
            }
        })
        .unwrap();
    let desc = KernelDesc::reflect(&module, func);
    specialize(&mut module, spec_constants);
    spirt::passes::legalize::structurize_func_cfgs(&mut module);
    let words = assemble(&module).unwrap();
    validate(&words).unwrap();
    let binary = spirv_tools::opt::create(Some(TargetEnv::Vulkan_1_2))
        .register_pass(Passes::StripDebugInfo)
        .register_pass(Passes::StripNonSemanticInfo)
        .register_performance_passes()
        .optimize(
            &words,
            &mut |_| (),
            Some(OptimizerOptions {
                preserve_bindings: true,
                ..OptimizerOptions::default()
            }),
        )
        .unwrap();
    let spirv = if let Binary::OwnedU8(bytes) = binary {
        bytes
    } else {
        binary.as_bytes().to_vec()
    };
    let mut module = Module::lower_from_spv_bytes(context.clone(), spirv).unwrap();
    let features = Features::reflect(&module);
    features.write_to_module(&mut module);
    spirt::passes::legalize::structurize_func_cfgs(&mut module);
    let spirv = assemble(&module).unwrap();
    validate(&spirv).unwrap();
    CompiledKernel {
        spirv,
        desc,
        features,
    }
}

fn op_execution_mode_local_size(local_size: [u32; 3]) -> Inst {
    let spec = Spec::get();
    let opcode = spec.well_known.OpExecutionMode;
    let literal_integer = spec.operand_kinds.lookup("LiteralInteger").unwrap();
    let operand_kinds = opcode
        .def()
        .all_operands()
        .map(|(_, x)| x)
        .skip(1)
        .chain([literal_integer; 3]);
    let imms = operand_kinds
        .zip(std::iter::once(ExecutionMode::LocalSize as u32).chain(local_size))
        .map(|(k, x)| Imm::Short(k, x))
        .collect();
    Inst { opcode, imms }
}

fn set_threads(module: &mut Module, threads: u32) {
    let (key, value) = module.exports.first().unwrap();
    let func = if let Exportee::Func(func) = value {
        *func
    } else {
        unreachable!()
    };
    let cx = module.cx();
    let func_decl = &mut module.funcs[func];
    let mut attrs = cx[func_decl.attrs].attrs.clone();
    let spec = Spec::get();
    attrs.retain(|attr| {
        if let Attr::SpvAnnotation(inst) = attr {
            if inst.opcode == spec.well_known.OpExecutionMode
                || inst.opcode == spec.well_known.OpExecutionModeId
            {
                return false;
            }
        }
        true
    });
    let inst = op_execution_mode_local_size([threads, 1, 1]);
    attrs.insert(Attr::SpvAnnotation(inst));
    func_decl.attrs = cx.intern(AttrSetDef { attrs });
}

fn specialize(module: &mut Module, spec_constants: &BTreeMap<String, [u32; 2]>) {
    set_threads(module, spec_constants["krnl::threads"][0]);
}
