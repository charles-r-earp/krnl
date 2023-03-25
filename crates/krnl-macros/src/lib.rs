#![allow(unused_imports, dead_code)]
use derive_syn_parse::Parse;
use proc_macro::TokenStream;
use proc_macro2::{Span as Span2, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    str::FromStr,
};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    token::{
        And, Brace, Bracket, Colon, Colon2, Comma, Const, Eq as SynEq, Fn, Gt, Lt, Mod, Mut, Paren,
        Pound, Pub, Semi, Unsafe,
    },
    Attribute, Block, Error, Expr, ExprParen, FnArg, Ident, ItemFn, ItemMod, Lit, LitBool, LitInt,
    LitStr, Stmt, Type, Visibility,
};

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Parse, Debug)]
struct InsideParen<T> {
    #[paren]
    paren: Paren,
    #[inside(paren)]
    value: T,
}

#[derive(Parse, Debug)]
struct InsideBracket<T> {
    #[allow(unused)]
    #[bracket]
    bracket: Bracket,
    #[inside(bracket)]
    value: T,
}

#[derive(Parse, Debug)]
struct InsideBrace<T> {
    #[brace]
    brace: Brace,
    #[inside(brace)]
    value: T,
}

impl<T: ToTokens> ToTokens for InsideBrace<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.brace
            .surround(tokens, |tokens| self.value.to_tokens(tokens));
    }
}

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return Error::new_spanned(&TokenStream2::from(attr), "unexpected tokens")
            .into_compile_error()
            .into();
    }
    let mut item = parse_macro_input!(item as ModuleItem);
    let mut build = true;
    let mut krnl = parse_quote! { ::krnl };
    let new_attr = Vec::with_capacity(item.attr.len());
    for attr in std::mem::replace(&mut item.attr, new_attr) {
        if attr.path.segments.len() == 1
            && attr
                .path
                .segments
                .first()
                .map_or(false, |x| x.ident == "krnl")
        {
            let tokens = attr.tokens.clone().into();
            let args = syn::parse_macro_input!(tokens as ModuleKrnlArgs);
            for arg in args.args.iter() {
                if let Some(krnl_crate) = arg.krnl_crate.as_ref() {
                    krnl = krnl_crate.clone();
                } else if let Some(ident) = &arg.ident {
                    if ident == "no_build" {
                        build = false;
                    } else {
                        return Error::new_spanned(
                            &ident,
                            format!("unknown krnl arg `{ident}`, expected `crate` or `no_build`"),
                        )
                        .into_compile_error()
                        .into();
                    }
                }
            }
        } else {
            item.attr.push(attr);
        }
    }
    item.tokens.extend(quote! {
        #[cfg(not(target_arch = "spirv"))]
        use #krnl as __krnl;
    });
    if build {
        let source = item.tokens.to_string();
        let mut hasher = std::collections::hash_map::DefaultHasher::default();
        source.hash(&mut hasher);
        let hash = hasher.finish();
        let name_with_hash = format_ident!("{ident}_{hash:x}", ident = item.ident);
        let tokens = item.tokens;
        item.tokens = quote! {
            mod __krnl_module_data {
                #[allow(non_upper_case_globals)]
                const __krnl_module_source: &'static str = #source;
            }
            include!(concat!(env!("CARGO_MANIFEST_DIR"), "/krnl-cache.rs"));
            __krnl_cache!(#name_with_hash);
            #tokens
        };
    } else {
    }

    item.into_token_stream().into()
}

#[derive(Parse, Debug)]
struct ModuleKrnlArgs {
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_terminated)]
    args: Punctuated<ModuleKrnlArg, Comma>,
}

#[derive(Parse, Debug)]
struct ModuleKrnlArg {
    crate_token: Option<syn::token::Crate>,
    #[parse_if(crate_token.is_some())]
    eq: Option<SynEq>,
    #[parse_if(crate_token.is_some())]
    krnl_crate: Option<syn::Path>,
    #[parse_if(crate_token.is_none())]
    ident: Option<Ident>,
}

#[derive(Parse, Debug)]
struct ModuleItem {
    #[call(Attribute::parse_outer)]
    attr: Vec<Attribute>,
    vis: Visibility,
    mod_token: Mod,
    ident: Ident,
    #[brace]
    brace: Brace,
    #[inside(brace)]
    tokens: TokenStream2,
}

impl ToTokens for ModuleItem {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        for attr in self.attr.iter() {
            attr.to_tokens(tokens);
        }
        self.vis.to_tokens(tokens);
        self.mod_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.brace
            .surround(tokens, |tokens| self.tokens.to_tokens(tokens));
    }
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as KernelAttr);
    let item = parse_macro_input!(item as KernelItem);
    match kernel_impl(attr, item) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

#[derive(Parse, Debug)]
struct KernelAttr {
    #[call(Punctuated::parse_terminated)]
    args: Punctuated<KernelAttrArg, Comma>,
}

impl KernelAttr {
    fn meta(&self) -> Result<KernelAttrMeta> {
        let mut meta = KernelAttrMeta::default();
        for arg in self.args.iter() {
            if let Some(threads) = arg.threads.as_ref() {
                let threads = &threads.value.inner;
                if !meta.threads.is_empty() {
                    return Err(Error::new_spanned(
                        &arg.ident,
                        "`threads` already specified",
                    ));
                }
                if threads.len() > 3 {
                    return Err(Error::new_spanned(
                        &arg.ident,
                        "expected 1, 2, or 3 dimensional `threads`",
                    ));
                }
                for dim in threads.iter() {
                    let dim = if let Some(lit) = dim.lit.as_ref() {
                        if let Ok(dim) = lit.base10_parse() {
                            dim
                        } else {
                            return Err(Error::new_spanned(lit, "expected u32"));
                        }
                    } else {
                        1
                    };
                    meta.threads.push(dim);
                }
                meta.thread_dims = threads.clone();
            } else {
                return Err(Error::new_spanned(&arg.ident, "unknown arg"));
            }
        }
        if meta.threads.is_empty() {
            return Err(Error::new(Span2::call_site(), "expected `threads`"));
        }
        Ok(meta)
    }
}

#[derive(Default, Debug)]
struct KernelAttrMeta {
    thread_dims: Punctuated<KernelThreadDim, Comma>,
    threads: Vec<u32>,
}

#[derive(Parse, Debug)]
struct KernelAttrArg {
    ident: Ident,
    #[parse_if(ident == "threads")]
    threads: Option<InsideParen<KrnlAttrThreads>>,
}

#[derive(Parse, Debug)]
struct KrnlAttrThreads {
    #[call(Punctuated::parse_separated_nonempty)]
    inner: Punctuated<KernelThreadDim, Comma>,
}

#[derive(Parse, Clone, Debug)]
struct KernelThreadDim {
    #[peek(Ident)]
    ident: Option<Ident>,
    #[parse_if(ident.is_none())]
    lit: Option<LitInt>,
}

impl ToTokens for KernelThreadDim {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        if let Some(ident) = self.ident.as_ref() {
            ident.to_tokens(tokens);
        } else if let Some(lit) = self.lit.as_ref() {
            lit.to_tokens(tokens);
        }
    }
}

/*
#[derive(Parse, Debug, Clone)]
struct IdentOrLiteral {
    ident: Option<Ident>,
    #[parse_if(ident.is_none())]
    lit: Option<LitInt>,
}

impl ToTokens for IdentOrLiteral {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        if let Some(ident) = self.ident.as_ref() {
            ident.to_tokens(tokens);
        } else if let Some(lit) = self.lit.as_ref() {
            lit.to_tokens(tokens);
        }
    }
}*/

#[derive(Parse, Debug)]
struct KernelItem {
    #[call(Attribute::parse_outer)]
    attrs: Vec<Attribute>,
    vis: Visibility,
    unsafe_token: Option<Unsafe>,
    #[allow(unused)]
    fn_token: Fn,
    ident: Ident,
    #[peek(Lt)]
    generics: Option<KernelGenerics>,
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_terminated)]
    args: Punctuated<KernelArg, Comma>,
    block: Block,
}

impl KernelItem {
    fn meta(&self, attr_meta: KernelAttrMeta) -> Result<KernelMeta> {
        let mut meta = KernelMeta {
            attr_meta,
            spec_metas: Vec::new(),
            unsafe_token: self.unsafe_token,
            ident: self.ident.clone(),
            arg_metas: Vec::with_capacity(self.args.len()),
            block: self.block.clone(),
            itemwise: false,
            arrays: HashMap::new(),
        };
        let mut spec_id = 0;
        if let Some(generics) = self.generics.as_ref() {
            meta.spec_metas = generics
                .specs
                .iter()
                .map(|x| {
                    let meta = KernelSpecMeta {
                        ident: x.ident.clone(),
                        ty: x.ty.clone(),
                        id: spec_id,
                        thread_dim: None,
                    };
                    spec_id += 1;
                    meta
                })
                .collect();
        }
        let mut binding = 0;
        for arg in self.args.iter() {
            let mut arg_meta = arg.meta()?;
            if arg_meta.kind.is_global() || arg_meta.kind.is_item() {
                arg_meta.binding.replace(binding);
                binding += 1;
            }
            meta.itemwise |= arg_meta.kind.is_item();
            if let Some(len) = arg_meta.len.as_ref() {
                meta.arrays
                    .entry(arg_meta.scalar_ty.scalar_type)
                    .or_default()
                    .push((arg.ident.clone(), len.clone()));
            }
            meta.arg_metas.push(arg_meta);
        }
        Ok(meta)
    }
}

#[derive(Parse, Debug)]
struct KernelGenerics {
    lt: Lt,
    #[call(Punctuated::parse_separated_nonempty)]
    specs: Punctuated<KernelSpec, Comma>,
    gt: Gt,
}

#[derive(Parse, Debug)]
struct KernelSpec {
    #[allow(unused)]
    pound: Pound,
    spec: InsideBracket<Ident>,
    #[allow(unused)]
    const_token: Const,
    ident: Ident,
    #[allow(unused)]
    colon: Colon,
    ty: KernelTypeScalar,
}

#[derive(Debug)]
struct KernelSpecMeta {
    ident: Ident,
    ty: KernelTypeScalar,
    id: u32,
    thread_dim: Option<usize>,
}

impl KernelSpecMeta {
    fn declare(&self) -> TokenStream2 {
        use ScalarType::*;
        let scalar_type = self.ty.scalar_type;
        let bits = scalar_type.size() * 8;
        let signed = if matches!(scalar_type, I8 | I16 | I32 | I64) {
            1
        } else {
            0
        };
        let float = matches!(scalar_type, F32 | F64);
        let ty_string = if float {
            format!("%ty = OpTypeFloat {bits}")
        } else {
            format!("%ty = OpTypeInt {bits} {signed}")
        };
        let values = if bits == 64 {
            "0 0"
        } else if float {
            "0"
        } else {
            "1"
        };
        let spec_string = format!("%spec = OpSpecConstant %ty {values}");
        let spec_id_string = format!("OpDecorate %spec SpecId {}", self.id);
        let ident = &self.ident;
        quote! {
            #[allow(non_snake_case)]
            let #ident = (|| unsafe {
                ::core::arch::asm! {
                    #ty_string,
                    #spec_string,
                    #spec_id_string,
                    "OpReturnValue %spec",
                    options(noreturn),
                }
            })();
        }
    }
}

#[derive(Clone, Debug)]
struct KernelTypeScalar {
    ident: Ident,
    scalar_type: ScalarType,
}

/*
impl ScalarType {
    fn size(&self) -> usize {
        match self.name {
            "u8" | "i8" => 1,
            "u16" | "i16" | "f16" | "bf16" => 2,
            "u32" | "i32" | "f32" => 4,
            "u64" | "i64" | "f64" => 8,
            _ => unreachable!(),
        }
    }
}*/

impl Parse for KernelTypeScalar {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        use ScalarType::*;
        let ident = input.parse()?;
        if let Some(scalar_type) = ScalarType::iter().find(|x| ident == x.name()) {
            Ok(Self { ident, scalar_type })
        } else {
            Err(Error::new(ident.span(), "expected scalar"))
        }
    }
}

#[derive(Parse, Debug)]
struct KernelArg {
    kind: KernelArgKind,
    ident: Ident,
    #[allow(unused)]
    colon: Colon,
    #[parse_if(kind.is_global())]
    slice_ty: Option<KernelTypeSlice>,
    #[parse_if(kind.is_item())]
    item_ty: Option<KernelTypeItem>,
    #[parse_if(kind.is_group())]
    array_ty: Option<KernelTypeArray>,
    #[parse_if(kind.is_push())]
    push_ty: Option<KernelTypeScalar>,
}

impl KernelArg {
    fn meta(&self) -> Result<KernelArgMeta> {
        let kind = self.kind;
        let (scalar_ty, mutable, len) = if let Some(slice_ty) = self.slice_ty.as_ref() {
            let slice_ty_ident = &slice_ty.ty;
            let mutable = if slice_ty.ty == "Slice" {
                false
            } else if slice_ty.ty == "UnsafeSlice" {
                true
            } else if slice_ty.ty == "SliceMut" {
                return Err(Error::new_spanned(&slice_ty_ident, "try `UnsafeSlice`"));
            } else {
                return Err(Error::new_spanned(
                    &slice_ty_ident,
                    "expected `Slice` or `UnsafeSlice`",
                ));
            };
            (slice_ty.scalar_ty.clone(), mutable, None)
        } else if let Some(array_ty) = self.array_ty.as_ref() {
            let len = array_ty.len.to_token_stream();
            (array_ty.scalar_ty.clone(), true, Some(len))
        } else if let Some(item_ty) = self.item_ty.as_ref() {
            (item_ty.scalar_ty.clone(), item_ty.mut_token.is_some(), None)
        } else if let Some(push_ty) = self.push_ty.as_ref() {
            (push_ty.clone(), false, None)
        } else {
            unreachable!("KernelArg::meta expected type!")
        };
        let meta = KernelArgMeta {
            kind,
            ident: self.ident.clone(),
            scalar_ty,
            mutable,
            binding: None,
            len,
        };
        Ok(meta)
    }
}

#[derive(Debug)]
struct KernelArgMeta {
    kind: KernelArgKind,
    ident: Ident,
    scalar_ty: KernelTypeScalar,
    mutable: bool,
    binding: Option<u32>,
    len: Option<TokenStream2>,
}

impl KernelArgMeta {
    fn compute_def_tokens(&self) -> Option<TokenStream2> {
        let ident = &self.ident;
        let ty = &self.scalar_ty.ident;
        if let Some(binding) = self.binding.as_ref() {
            let set = LitInt::new("0", Span2::call_site());
            let binding = LitInt::new(&binding.to_string(), Span2::call_site());
            let mut_token = if self.mutable {
                Some(Mut::default())
            } else {
                None
            };
            Some(quote! {
                #[spirv(storage_buffer, descriptor_set = #set, binding = #binding)] #ident: &#mut_token [#ty; 1]
            })
        } else {
            None
        }
    }
    fn device_fn_def_tokens(&self) -> TokenStream2 {
        let ident = &self.ident;
        let ty = &self.scalar_ty.ident;
        let mutable = self.mutable;
        use KernelArgKind::*;
        match self.kind {
            Global => {
                if mutable {
                    quote! {
                        #ident: ::krnl_core::buffer::UnsafeSlice<#ty>
                    }
                } else {
                    quote! {
                        #ident: ::krnl_core::buffer::Slice<#ty>
                    }
                }
            }
            Item => {
                if mutable {
                    quote! {
                        #ident: &mut #ty
                    }
                } else {
                    quote! {
                        #ident: #ty
                    }
                }
            }
            Group => quote! {
                #ident: ::krnl_core::buffer::UnsafeSlice<#ty>
            },
            Push => quote! {
                #ident: #ty
            },
        }
    }
    fn device_slices(&self) -> TokenStream2 {
        let ident = &self.ident;
        let mutable = self.mutable;
        use KernelArgKind::*;
        match self.kind {
            Global | Item => {
                let offset = format_ident!("__krnl_offset_{ident}");
                let len = format_ident!("__krnl_len_{ident}");
                let slice_fn = if mutable {
                    quote! {
                        ::krnl_core::buffer::UnsafeSlice::from_unsafe_raw_parts
                    }
                } else {
                    quote! {
                        ::krnl_core::buffer::Slice::from_raw_parts
                    }
                };
                quote! {
                    let #ident = unsafe {
                        #slice_fn(#ident, __krnl_push_consts.#offset as usize, __krnl_push_consts.#len as usize)
                    };
                }
            }
            Group => {
                let offset = format_ident!("__krnl_offset_{ident}");
                let len = format_ident!("__krnl_len_{ident}");
                let scalar_name = self.scalar_ty.scalar_type.name();
                let array = format_ident!("__krnl_group_array_{scalar_name}");
                quote! {
                    let #ident = {
                        unsafe {
                            ::krnl_core::buffer::UnsafeSlice::from_unsafe_raw_parts(#array, #offset, #len)
                        }
                    };
                }
            }
            Push => TokenStream2::new(),
        }
    }
    fn device_fn_call_tokens(&self) -> TokenStream2 {
        let ident = &self.ident;
        let mutable = self.mutable;
        use KernelArgKind::*;
        match self.kind {
            Global | Group => quote! {
                #ident
            },
            Item => {
                if mutable {
                    quote! {
                        unsafe {
                            use ::krnl_core::buffer::UnsafeIndex;
                            #ident.unsafe_index_mut(kernel.item_id() as usize)
                        },
                    }
                } else {
                    quote! {
                        #ident[kernel.item_id() as usize]
                    }
                }
            }
            Push => quote! {
                __krnl_push_consts.#ident
            },
        }
    }
}

#[derive(Parse, Debug)]
struct KernelArgAttr {
    pound: Option<Pound>,
    #[parse_if(pound.is_some())]
    ident: Option<InsideBracket<Ident>>,
}

impl KernelArgAttr {
    fn kind(&self) -> Result<KernelArgKind> {
        use KernelArgKind::*;
        let ident = if let Some(ident) = self.ident.as_ref() {
            &ident.value
        } else {
            return Ok(Push);
        };
        let kind = if ident == "global" {
            Global
        } else if ident == "item" {
            Item
        } else if ident == "group" {
            Group
        } else {
            return Err(Error::new_spanned(
                ident,
                "expected `global`, `item`, or `group`",
            ));
        };
        Ok(kind)
    }
}

#[derive(Clone, Copy, derive_more::IsVariant, PartialEq, Eq, Hash, Debug)]
enum KernelArgKind {
    Global,
    Item,
    Group,
    Push,
}

impl Parse for KernelArgKind {
    fn parse(input: ParseStream) -> Result<Self> {
        KernelArgAttr::parse(input)?.kind()
    }
}

#[derive(Parse, Debug)]
struct KernelTypeItem {
    and: Option<And>,
    #[parse_if(and.is_some())]
    mut_token: Option<Mut>,
    scalar_ty: KernelTypeScalar,
}

#[derive(Parse, Debug)]
struct KernelTypeSlice {
    ty: Ident,
    lt: Lt,
    scalar_ty: KernelTypeScalar,
    gt: Gt,
}

#[derive(Parse, Debug)]
struct KernelTypeArray {
    ty: Ident,
    lt: Lt,
    scalar_ty: KernelTypeScalar,
    comma: Comma,
    len: KernelArrayLength,
    gt: Gt,
}

#[derive(Debug)]
struct KernelArrayLength {
    block: Option<Block>,
    ident: Option<Ident>,
    lit: Option<LitInt>,
}

impl Parse for KernelArrayLength {
    fn parse(input: &syn::parse::ParseBuffer) -> Result<Self> {
        if input.peek(Brace) {
            Ok(Self {
                block: Some(input.parse()?),
                ident: None,
                lit: None,
            })
        } else if input.peek(Ident) {
            Ok(Self {
                block: None,
                ident: Some(input.parse()?),
                lit: None,
            })
        } else {
            Ok(Self {
                block: None,
                ident: None,
                lit: Some(input.parse()?),
            })
        }
    }
}

impl ToTokens for KernelArrayLength {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        if let Some(block) = self.block.as_ref() {
            for stmt in block.stmts.iter() {
                stmt.to_tokens(tokens);
            }
        } else if let Some(ident) = self.ident.as_ref() {
            ident.to_tokens(tokens);
        } else if let Some(lit) = self.lit.as_ref() {
            lit.to_tokens(tokens);
        }
    }
}

#[derive(Debug)]
struct KernelMeta {
    attr_meta: KernelAttrMeta,
    spec_metas: Vec<KernelSpecMeta>,
    ident: Ident,
    unsafe_token: Option<Unsafe>,
    arg_metas: Vec<KernelArgMeta>,
    itemwise: bool,
    block: Block,
    arrays: HashMap<ScalarType, Vec<(Ident, TokenStream2)>>,
}

impl KernelMeta {
    fn desc(&self) -> Result<KernelDesc> {
        let mut kernel_desc = KernelDesc::default();
        kernel_desc.name = self.ident.to_string();
        kernel_desc.safe = self.unsafe_token.is_none();
        kernel_desc.threads = self.attr_meta.threads.clone();
        for spec in self.spec_metas.iter() {
            let thread_dim = self.attr_meta.thread_dims.iter().position(|dim| {
                if let Some(ident) = dim.ident.as_ref() {
                    ident == &spec.ident
                } else {
                    false
                }
            });
            kernel_desc.spec_descs.push(SpecDesc {
                name: spec.ident.to_string(),
                scalar_type: spec.ty.scalar_type,
                thread_dim,
            })
        }
        for arg_meta in self.arg_metas.iter() {
            let kind = arg_meta.kind;
            let scalar_type = arg_meta.scalar_ty.scalar_type;
            use KernelArgKind::*;
            match kind {
                Global | Item => {
                    kernel_desc.slice_descs.push(SliceDesc {
                        name: arg_meta.ident.to_string(),
                        scalar_type,
                        mutable: arg_meta.mutable,
                        item: kind.is_item(),
                    });
                }
                Group => (),
                Push => {
                    kernel_desc.push_descs.push(PushDesc {
                        name: arg_meta.ident.to_string(),
                        scalar_type,
                    });
                }
            }
        }
        kernel_desc
            .push_descs
            .sort_by_key(|x| x.scalar_type.size() as i32);
        Ok(kernel_desc)
    }
    fn compute_def_args(&self) -> Punctuated<TokenStream2, Comma> {
        let mut id = 1;
        let arrays = self.arrays.iter().map(|(scalar_type, _arrays)| {
            let scalar_name = scalar_type.name();
            let ident = format_ident!("__krnl_group_array_{scalar_name}_{id}");
            let ty = format_ident!("{scalar_name}");
            id += 1;
            quote! {
                #[spirv(workgroup)] #ident: &mut [#ty; 1]
            }
        });
        self.arg_metas
            .iter()
            .filter_map(|arg| arg.compute_def_tokens())
            .chain(arrays)
            .collect()
    }
    fn compute_threads(&self) -> Punctuated<LitInt, Comma> {
        self.attr_meta
            .threads
            .iter()
            .map(|dim| LitInt::new(&dim.to_string(), Span2::call_site()))
            .collect()
    }
    fn threads(&self) -> Punctuated<TokenStream2, Comma> {
        self.attr_meta
            .thread_dims
            .iter()
            .map(|dim| dim.to_token_stream())
            .collect()
    }
    fn threads3(&self) -> Punctuated<TokenStream2, Comma> {
        self.threads()
            .into_iter()
            .chain([
                quote! {
                    1
                },
                quote! {
                    1
                },
            ])
            .take(3)
            .collect()
    }
    fn declare_specs(&self) -> TokenStream2 {
        self.spec_metas
            .iter()
            .flat_map(|spec| spec.declare())
            .collect()
    }
    fn spec_def_args(&self) -> Punctuated<TokenStream2, Comma> {
        self.spec_metas
            .iter()
            .map(|spec| {
                let ident = &spec.ident;
                let ty = &spec.ty.ident;
                quote! {
                    #[allow(non_snake_case)]
                    #ident: #ty
                }
            })
            .collect()
    }
    fn spec_args(&self) -> Vec<Ident> {
        self.spec_metas
            .iter()
            .map(|spec| spec.ident.clone())
            .collect()
    }
    fn device_arrays(&self) -> TokenStream2 {
        let spec_def_args: Punctuated<_, Comma> = self
            .spec_def_args()
            .into_iter()
            .map(|arg| {
                quote! {
                    #[allow(unused)] #arg
                }
            })
            .collect();
        let spec_args: Punctuated<_, Comma> = self.spec_args().into_iter().collect();
        let group_barrier = if self.arg_metas.iter().any(|arg| arg.kind.is_global()) {
            quote! {
                unsafe {
                    ::krnl_core::spirv_std::arch::workgroup_memory_barrier();
                }
            }
        } else {
            TokenStream2::new()
        };
        let mut id = 1;
        self.arrays
            .iter()
            .flat_map(|(scalar_type, arrays)| {
                let scalar_name = scalar_type.name();
                let ident = format_ident!("__krnl_group_array_{scalar_name}");
                let ident_with_id = format_ident!("{ident}_{id}");
                let id_lit = LitInt::new(&id.to_string(), Span2::call_site());
                id += 1;
                let len = format_ident!("{ident}_len");
                let offset = format_ident!("{ident}_offset");
                let array_offsets_lens: TokenStream2 = arrays
                    .iter()
                    .map(|(array, len_expr)| {
                        let array_offset = format_ident!("__krnl_offset_{array}");
                        let array_len = format_ident!("__krnl_len_{array}");
                        quote! {
                            let #array_offset = #offset;
                            let #array_len = {
                                const fn #array_len(#spec_def_args) -> usize {
                                    #len_expr
                                }
                                #array_len(#spec_args)
                            };
                            #offset += #array_len;
                        }
                    })
                    .collect();
                let group_init = quote! {
                    {
                        use ::krnl_core::spirv_std::arch::IndexUnchecked;
                        let mut __krnl_i = kernel.thread_index() as usize;
                        let __krnl_group_stride = (__krnl_threads.x * __krnl_threads.y * __krnl_threads.z) as usize;
                        while __krnl_i < #len {
                            unsafe {
                                *#ident.index_unchecked_mut(__krnl_i) = Default::default();
                            }
                            __krnl_i += __krnl_group_stride;
                        }
                    }
                };
                quote! {
                    let #ident = #ident_with_id;
                    let mut #offset = 0usize;
                    #array_offsets_lens
                    let #len = #offset;
                    unsafe {
                        use ::krnl_core::spirv_std::arch::IndexUnchecked;
                        *__krnl_kernel_data.index_unchecked_mut(#id_lit) = #len as u32;
                    }
                    #group_init
                }
            })
            .chain(group_barrier)
            .collect()
    }
    fn host_array_length_checks(&self) -> TokenStream2 {
        let spec_def_args = self.spec_def_args();
        self.arg_metas
            .iter()
            .flat_map(|arg| {
                if let Some(len) = arg.len.as_ref() {
                    quote! {
                        const _: () = {
                            const fn __krnl_array_len(#spec_def_args) -> usize {
                                #len
                            }
                            let _ = __krnl_array_len;
                        };
                    }
                } else {
                    TokenStream2::new()
                }
            })
            .collect()
    }
    fn device_slices(&self) -> TokenStream2 {
        self.arg_metas
            .iter()
            .map(|arg| arg.device_slices())
            .collect()
    }
    fn device_items(&self) -> TokenStream2 {
        let mut items = self
            .arg_metas
            .iter()
            .filter(|arg| arg.kind.is_item())
            .map(|arg| &arg.ident);
        if let Some(first) = items.next() {
            let ident = format_ident!("__krnl_len_{first}");
            quote! {
                __krnl_push_consts.#ident
            }
            .into_iter()
            .chain(items.flat_map(|item| {
                let ident = format_ident!("__krnl_len_{item}");
                quote! {
                    .max(__krnl_push_consts.#ident)
                }
            }))
            .chain(quote! {
                as u32
            })
            .chain([])
            .collect()
        } else {
            quote! {
                0
            }
        }
    }
    fn device_fn_def_args(&self) -> Punctuated<TokenStream2, Comma> {
        self.spec_metas
            .iter()
            .map(|x| {
                let ident = &x.ident;
                let ty = &x.ty.ident;
                let allow_unused = x.thread_dim.map(|_| {
                    quote! {
                        #[allow(unused)]
                    }
                });
                quote! {
                    #allow_unused
                    #[allow(non_snake_case)]
                    #ident: #ty
                }
            })
            .chain(self.arg_metas.iter().map(|arg| arg.device_fn_def_tokens()))
            .collect()
    }
    fn device_fn_call_args(&self) -> Punctuated<TokenStream2, Comma> {
        self.spec_metas
            .iter()
            .map(|spec| spec.ident.to_token_stream())
            .chain(self.arg_metas.iter().map(|arg| arg.device_fn_call_tokens()))
            .collect()
    }
    fn dispatch_args(&self) -> TokenStream2 {
        let mut tokens = TokenStream2::new();
        for arg in self.arg_metas.iter() {
            let ident = &arg.ident;
            let ty = &arg.scalar_ty.ident;
            if arg.binding.is_some() {
                let slice_ty = if arg.mutable {
                    format_ident!("SliceMut")
                } else {
                    format_ident!("Slice")
                };
                tokens.extend(quote! {
                    #ident: #slice_ty<#ty>,
                });
            } else if arg.kind.is_push() {
                tokens.extend(quote! {
                    #ident: #ty,
                });
            }
        }
        tokens
    }
    fn dispatch_slice_args(&self) -> TokenStream2 {
        let mut tokens = TokenStream2::new();
        for arg in self.arg_metas.iter() {
            let ident = &arg.ident;
            if arg.binding.is_some() {
                tokens.extend(quote! {
                    #ident.into(),
                });
            }
        }
        tokens
    }
    fn dispatch_push_args(&self) -> TokenStream2 {
        let mut tokens = TokenStream2::new();
        for arg in self.arg_metas.iter() {
            let ident = &arg.ident;
            if arg.kind.is_push() {
                tokens.extend(quote! {
                    #ident.into(),
                })
            }
        }
        tokens
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
            F64 => "f64",
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
    fn size(&self) -> usize {
        use ScalarType::*;
        match self {
            U8 | I8 => 1,
            U16 | I16 | F16 | BF16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }
}

impl FromStr for ScalarType {
    type Err = ();
    fn from_str(input: &str) -> Result<Self, ()> {
        use ScalarType::*;
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
        serializer.serialize_str(self.as_str())
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

#[derive(Default, Serialize, Deserialize)]
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

impl KernelDesc {
    fn encode(&self) -> Result<String> {
        let mut entry_point = "__krnl_kernel_data_".to_string();
        let bytes = bincode::serialize(self).map_err(|e| Error::new(Span2::call_site(), e))?;
        for byte in bytes.iter().copied() {
            let a = byte / 16;
            let b = byte % 16;
            entry_point.push(char::from_digit(a as _, 16).unwrap());
            entry_point.push(char::from_digit(b as _, 16).unwrap());
        }
        Ok(entry_point)
    }
    fn push_const_fields(&self) -> TokenStream2 {
        let mut tokens = TokenStream2::new();
        let mut bytes = 0;
        for push_desc in self.push_descs.iter() {
            let ident = format_ident!("{}", push_desc.name);
            let ty = format_ident!("{}", push_desc.scalar_type.name());
            tokens.extend(quote! {
               #ident: #ty,
            });
            bytes += push_desc.scalar_type.size();
        }
        let mut pad_bytes = 0;
        while bytes % 4 != 0 {
            let ident = format_ident!("__krnl_pad{pad_bytes}");
            tokens.extend(quote! {
               #ident: u8,
            });
            pad_bytes += 1;
            bytes += 1;
        }
        for slice_desc in self.slice_descs.iter() {
            let offset_ident = format_ident!("__krnl_offset_{}", slice_desc.name);
            let len_ident = format_ident!("__krnl_len_{}", slice_desc.name);
            tokens.extend(quote! {
                #offset_ident: u32,
                #len_ident: u32,
            });
        }
        tokens
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
struct Features {
    shader_int8: bool,
    shader_int16: bool,
    shader_int64: bool,
    shader_float16: bool,
    shader_float64: bool,
}

#[derive(Serialize, Deserialize)]
struct SpecDesc {
    name: String,
    scalar_type: ScalarType,
    thread_dim: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct SliceDesc {
    name: String,
    scalar_type: ScalarType,
    mutable: bool,
    item: bool,
}

#[derive(Serialize, Deserialize)]
struct PushDesc {
    name: String,
    scalar_type: ScalarType,
}

fn kernel_impl(attr: KernelAttr, item: KernelItem) -> Result<TokenStream2> {
    let kernel_meta = item.meta(attr.meta()?)?;
    let kernel_desc = kernel_meta.desc()?;
    let unsafe_token = kernel_meta.unsafe_token;
    let ident = &kernel_meta.ident;
    let dimensionality = kernel_desc.threads.len();
    let device_tokens = {
        let kernel_data = format_ident!("{}", kernel_desc.encode()?);
        let block = &kernel_meta.block;
        let compute_threads = kernel_meta.compute_threads();
        let compute_def_args = kernel_meta.compute_def_args();
        let declare_specs = kernel_meta.declare_specs();
        let threads3 = kernel_meta.threads3();
        let items = kernel_meta.device_items();
        let device_arrays = kernel_meta.device_arrays();
        let device_slices = kernel_meta.device_slices();
        let device_fn_def_args = kernel_meta.device_fn_def_args();
        let device_fn_call_args = kernel_meta.device_fn_call_args();
        let push_consts_ident = format_ident!("__krnl_{ident}PushConsts");
        let push_const_fields = kernel_desc.push_const_fields();
        let kernel_dim = if dimensionality == 1 {
            quote! { u32 }
        } else if dimensionality == 2 {
            quote! {
                ::krnl_core::glam::UVec2
            }
        } else {
            quote! { ::krnl_core::glam::UVec3 }
        };
        let push_struct_tokens = quote! {
            #[cfg(target_arch = "spirv")]
            #[automatically_derived]
            #[repr(C)]
            pub struct #push_consts_ident {
                #push_const_fields
            }
        };
        let mut device_fn_call = quote! {
            #unsafe_token {
                #ident (
                    &kernel,
                    #device_fn_call_args
                );
            }
        };
        if kernel_meta.itemwise {
            device_fn_call = quote! {
                while kernel.item_id() < kernel.items() {
                    #device_fn_call
                    kernel.next_item();
                }
            }
        }
        quote! {
            #push_struct_tokens
            #[cfg(target_arch = "spirv")]
            #[::krnl_core::spirv_std::spirv(compute(threads(#compute_threads)))]
            #[allow(unused)]
            pub fn #ident(
                #[allow(unused)]
                #[spirv(push_constant)]
                __krnl_push_consts: &#push_consts_ident,
                #[allow(unused)]
                #[spirv(global_invocation_id)]
                __krnl_global_id: ::krnl_core::glam::UVec3,
                #[allow(unused)]
                #[spirv(num_workgroups)]
                __krnl_groups: ::krnl_core::glam::UVec3,
                #[allow(unused)]
                #[spirv(workgroup_id)]
                __krnl_group_id: ::krnl_core::glam::UVec3,
                #[allow(unused)]
                #[spirv(num_subgroups)]
                __krnl_subgroups: u32,
                #[allow(unused)]
                #[spirv(subgroup_id)]
                __krnl_subgroup_id: u32,
                #[allow(unused)]
                #[spirv(subgroup_size)]
                __krnl_subgroup_threads: u32,
                #[allow(unused)]
                #[spirv(subgroup_local_invocation_id)]
                __krnl_subgroup_thread_id: u32,
                #[allow(unused)]
                #[spirv(local_invocation_id)]
                __krnl_thread_id: ::krnl_core::glam::UVec3,
                #[allow(unused)]
                #[spirv(storage_buffer, descriptor_set = 1, binding = 0)]
                #kernel_data: &mut [u32],
                #compute_def_args
            ) {
                #unsafe_token fn #ident(
                    #[allow(unused)]
                    kernel: &::krnl_core::kernel::Kernel<#kernel_dim>,
                    #device_fn_def_args
                ) #block
                {
                    let __krnl_kernel_data = #kernel_data;
                    unsafe {
                        use ::krnl_core::spirv_std::arch::IndexUnchecked as _;
                        *__krnl_kernel_data.index_unchecked_mut(0) = 1;
                    }
                    #declare_specs
                    let __krnl_threads = ::krnl_core::glam::UVec3::new(#threads3);
                    /*unsafe {
                        ::core::arch::asm!(
                            "%uint = OpTypeInt 32 0",
                            "%uvec3 = OpTypeVector %uint 3",
                            "%spec = OpSpecConstant %uint 1",
                            "OpDecorate %spec SpecId 100",
                            "%threads = OpSpecConstantComposite %uvec3 %spec %spec %spec",
                            "OpDecorate %threads BuiltIn WorkgroupSize",
                            /*threads_x = in(reg) __krnl_threads.x,
                            threads_y = in(reg) __krnl_threads.y,
                            threads_z = in(reg) __krnl_threads.z,*/
                        );
                    }*/
                    let mut kernel = ::krnl_core::kernel::Kernel::<#kernel_dim>::from(::krnl_core::kernel::KernelArgs {
                        groups: __krnl_groups,
                        group_id: __krnl_group_id,
                        subgroups: __krnl_subgroups,
                        subgroup_id: __krnl_subgroup_id,
                        subgroup_threads: __krnl_subgroup_threads,
                        subgroup_thread_id: __krnl_subgroup_thread_id,
                        threads: __krnl_threads,
                        thread_id: __krnl_thread_id,
                        items: #items,
                    });
                    #device_arrays
                    #device_slices
                    #device_fn_call
                }
            }
        }
    };
    let host_tokens = {
        let dispatch_args = kernel_meta.dispatch_args();
        let dispatch_slice_args = kernel_meta.dispatch_slice_args();
        let dispatch_push_args = kernel_meta.dispatch_push_args();
        let dimensionality = LitInt::new(&dimensionality.to_string(), Span2::call_site());
        let safe = unsafe_token.is_none();
        let host_array_length_checks = kernel_meta.host_array_length_checks();
        let kernel_builder_specialize_fn = if !kernel_desc.spec_descs.is_empty() {
            let spec_def_args = kernel_meta.spec_def_args();
            let spec_args = kernel_meta.spec_args();
            quote! {
                pub fn specialize(mut self, #spec_def_args) -> Result<Self> {
                    let inner = self.inner.specialize(&[#(#spec_args.into()),*])?;
                    Ok(Self {
                        inner,
                    })
                }
            }
        } else {
            TokenStream2::new()
        };
        quote! {
            #[cfg(not(target_arch = "spirv"))]
            #[automatically_derived]
            pub mod #ident {
                use super::__krnl::{
                    anyhow::{self, Result},
                    krnl_core::half::{f16, bf16},
                    buffer::{Slice, SliceMut},
                    device::{Device, Kernel as KernelBase, KernelBuilder as KernelBuilderBase},
                    once_cell::sync::Lazy,
                };

                #host_array_length_checks

                pub struct KernelBuilder {
                    inner: KernelBuilderBase,
                }

                pub fn builder() -> Result<KernelBuilder> {
                    static BUILDER: Lazy<Result<KernelBuilderBase, String>> = Lazy::new(|| {
                        let bytes = __krnl_kernel!(#ident);
                        let inner = KernelBuilderBase::from_bytes(bytes)
                            .map_err(|e| e.to_string())?;
                        assert_eq!(inner.safe(), #safe);
                        Ok(inner)
                    });
                    match &*BUILDER {
                        Ok(inner) => {
                            Ok(KernelBuilder { inner: inner.clone() })
                        }
                        Err(e) => Err(anyhow::Error::msg(e)),
                    }
                }

                impl KernelBuilder {
                    #kernel_builder_specialize_fn
                    pub fn build(&self, device: Device) -> Result<Kernel> {
                        let inner = self.inner.build(device)?;
                        Ok(Kernel { inner })
                    }
                }

                pub struct Kernel {
                    inner: KernelBase,
                }

                impl Kernel {
                    pub fn with_global_threads(mut self, global_threads: [u32; #dimensionality]) -> Self {
                        self.inner = self.inner.global_threads(&global_threads);
                        self
                    }
                    pub fn with_groups(mut self, groups: [u32; #dimensionality]) -> Self {
                        self.inner = self.inner.groups(&groups);
                        self
                    }
                    pub #unsafe_token fn dispatch(&self, #dispatch_args) -> Result<()> {
                        unsafe { self.inner.dispatch(&[#dispatch_slice_args], &[#dispatch_push_args]) }
                    }
                }
            }
        }
    };
    let tokens = quote! {
        #host_tokens
        #device_tokens
    };
    /*{
        let file = syn::parse2(tokens.clone()).map_err(|e| {
            let mut err = Error::new_spanned(
                &tokens,
                &format!("unable to parse module tokens:\n{tokens}"),
            );
            err.combine(e);
            err
        })?;
        let source = prettyplease::unparse(&file);
        eprintln!("{source}");
    }*/
    Ok(tokens)
}

#[proc_macro]
pub fn __krnl_module(input: TokenStream) -> TokenStream {
    let version = env!("CARGO_PKG_VERSION");
    let input = parse_macro_input!(input as KrnlcCacheInput);
    let cache_bytes: Vec<u8> = input
        .bytes
        .iter()
        .map(|x| x.base10_parse().unwrap())
        .collect();
    let krnlc_version: String =
        bincode::deserialize(&cache_bytes).expect("Expected krnlc version!");
    let mut error = None;
    if krnlc_version != version {
        error.replace(format!(
            "Cached krnlc version {krnlc_version} is not compatible!"
        ));
    }
    let mut macro_arms = TokenStream2::new();
    if error.is_none() {
        match bincode::deserialize::<KrnlcCache>(&cache_bytes) {
            Ok(cache) => {
                if let Some(modules) = cache.modules.as_ref() {
                    if let Some(module) = modules.get(&input.module.to_string()) {
                        for (kernel, kernel_desc) in module {
                            let ident = format_ident!("{kernel}");
                            let bytes = bincode::serialize(kernel_desc).unwrap();
                            let bytes: Punctuated<LitInt, Comma> = bytes
                                .iter()
                                .map(|x| LitInt::new(&x.to_string(), Span2::call_site()))
                                .collect();
                            macro_arms.extend(quote! {
                                (#ident) => {
                                    &[#bytes]
                                };
                            });
                        }
                    } else {
                        error.replace("Cache modified, recompile with krnlc".to_string());
                    }
                }
            }
            Err(e) => {
                error.replace(format!(
                    "Unable to deserialize cache: {e:?}, try recompiling with krnlc"
                ));
            }
        }
    }
    let error = if let Some(error) = error {
        quote! {
            compile_error!(#error);
        }
    } else {
        TokenStream2::new()
    };
    quote! {
        macro_rules! __krnl_kernel {
            #macro_arms
            ($i:ident) => { &[] };
        }
        #error
    }
    .into()
}

#[derive(Parse, Debug)]
struct KrnlcCacheInput {
    module: Ident,
    comma: Comma,
    #[bracket]
    bracket: Bracket,
    #[inside(bracket)]
    #[call(Punctuated::parse_terminated)]
    bytes: Punctuated<LitInt, Comma>,
}

#[derive(Deserialize)]
struct KrnlcCache {
    version: String,
    modules: Option<HashMap<String, HashMap<String, KernelDesc>>>,
}
