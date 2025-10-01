use crate::scalar::ScalarType;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::LitInt;

pub(crate) struct KernelDesc {
    pub(crate) name: String,
    pub(crate) generics: Vec<GenericInput>,
    pub(crate) inputs: Vec<KernelInput>,
}

impl KernelDesc {
    pub(crate) fn emit(&self, path: &str, spirv: &[u32]) -> String {
        todo!()
    }
}

pub(crate) struct GenericInput {
    pub(crate) name: String,
    pub(crate) scalar_type: ScalarType,
}

pub(crate) struct KernelInput {
    pub(crate) name: String,
    pub(crate) kind: KernelInputKind,
}

pub(crate) enum KernelInputKind {
    Spec(ScalarType),
    Slice {
        item: bool,
        mutable: bool,
        elem: ScalarType,
    },
    Push(ScalarType),
}
