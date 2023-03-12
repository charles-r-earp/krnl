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
    Attribute, Block, Error, Expr, FnArg, Ident, ItemFn, ItemMod, Lit, LitBool, LitInt, LitStr,
    Stmt, Type, Visibility,
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

#[proc_macro_attribute]
pub fn module(_: TokenStream, item: TokenStream) -> TokenStream {
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
        let file = syn::parse2(item.tokens.clone()).expect("unable to parse module tokens");
        // TODO: filter out items with `#[cfg(not(target_arch = "spirv"))]`?
        let source = prettyplease::unparse(&file);
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
                meta.threads = threads.clone();
            } else if arg.ident == "for_each" {
                meta.for_each = true;
            } else {
                return Err(Error::new_spanned(&arg.ident, "unknown arg"));
            }
        }
        if meta.threads.is_empty() {
            return Err(Error::new(Span2::call_site(), "expected `threads`"));
        }
        if meta.for_each && meta.threads.len() != 1 {
            return Err(Error::new(
                Span2::call_site(),
                "expected 1 dimensional `threads` for `for_each` kernel",
            ));
        }
        Ok(meta)
    }
}

#[derive(Default, Debug)]
struct KernelAttrMeta {
    for_each: bool,
    threads: Punctuated<IdentOrLiteral, Comma>,
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
    inner: Punctuated<IdentOrLiteral, Comma>,
}

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
}

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
        };
        if let Some(generics) = self.generics.as_ref() {
            let mut id = 0;
            meta.spec_metas = generics
                .specs
                .iter()
                .map(|x| {
                    let meta = KernelSpecMeta {
                        ident: x.ident.clone(),
                        ty: x.ty.clone(),
                        id,
                    };
                    id += 1;
                    meta
                })
                .collect();
        }
        let mut binding = 0;
        for arg in self.args.iter() {
            let mut arg_meta = arg.meta()?;
            if let Some(attr) = arg_meta.attr.as_ref() {
                if attr == "global" {
                    arg_meta.binding.replace(binding);
                    binding += 1;
                }
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
        let values = if bits == 64 { "0 0" } else { "0" };
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
    #[peek(Pound)]
    attr: Option<KernelArgAttr>,
    ident: Ident,
    #[allow(unused)]
    colon: Colon,
    #[parse_if(attr.eq("global"))]
    and: Option<And>,
    mut_token: Option<Mut>,
    #[parse_if(attr.eq("global"))]
    slice_ty: Option<KernelTypeSlice>,
    #[parse_if(attr.is_none())]
    scalar_ty: Option<KernelTypeScalar>,
}

impl KernelArg {
    fn meta(&self) -> Result<KernelArgMeta> {
        let scalar_ty = self
            .slice_ty
            .as_ref()
            .map(|x| &x.ty)
            .or(self.scalar_ty.as_ref())
            .expect("KernelArg::meta expected scalar_ty")
            .clone();
        let meta = KernelArgMeta {
            attr: self.attr.as_ref().map(|x| x.ident.value.clone()),
            ident: self.ident.clone(),
            scalar_ty,
            mutable: self.mut_token.is_some(),
            len: None,
            binding: None,
        };
        Ok(meta)
    }
}

#[derive(Debug)]
struct KernelArgMeta {
    attr: Option<Ident>,
    ident: Ident,
    scalar_ty: KernelTypeScalar,
    mutable: bool,
    len: Option<IdentOrLiteral>,
    binding: Option<usize>,
}

impl KernelArgMeta {
    fn compute_def_tokens(&self) -> TokenStream2 {
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
            quote! {
                #[spirv(storage_buffer, descriptor_set = #set, binding = #binding)] #ident: &#mut_token [#ty],
            }
        } else {
            TokenStream2::new()
        }
    }
    fn device_fn_def_tokens(&self) -> TokenStream2 {
        let ident = &self.ident;
        let ty = &self.scalar_ty.ident;
        if let Some(attr) = self.attr.as_ref() {
            if attr == "global" {
                let index_trait = if self.mutable {
                    quote! {
                        ::krnl_core::arch::UnsafeIndexMut
                    }
                } else {
                    quote! {
                        ::core::ops::Index
                    }
                };
                quote! {
                    #ident: &(impl #index_trait<usize, Output=#ty> + ::krnl_core::arch::Length),
                }
            } else {
                panic!("unexpected KernelArgMeta attr {:?}", self.attr);
            }
        } else {
            quote! {
                #ident: #ty,
            }
        }
    }
    fn device_fn_call_tokens(&self) -> TokenStream2 {
        let ident = &self.ident;
        if let Some(attr) = self.attr.as_ref() {
            if attr == "global" {
                let offset = format_ident!("__krnl_offset_{ident}");
                let len = format_ident!("__krnl_len_{ident}");
                let slice_ty = if self.mutable {
                    format_ident!("UnsafeSliceMut")
                } else {
                    format_ident!("Slice")
                };
                quote! {
                    unsafe {
                        &::krnl_core::arch::#slice_ty::from_raw_parts(#ident, __krnl_push_consts.#offset as usize, __krnl_push_consts.#len as usize)
                    },
                }
            } else {
                panic!("unexpected KernelArgMeta attr {:?}", self.attr);
            }
        } else {
            quote! {
                __krnl_push_consts.#ident,
            }
        }
    }
}

#[derive(Parse, Debug)]
struct KernelArgAttr {
    pound: Pound,
    ident: InsideBracket<Ident>,
}

trait PartialEqExt<T: ?Sized> {
    fn eq(&self, other: &T) -> bool;
}

impl PartialEqExt<str> for Option<KernelArgAttr> {
    fn eq(&self, other: &str) -> bool {
        if let Some(this) = self.as_ref() {
            this.ident.value == other
        } else {
            false
        }
    }
}

#[derive(Parse, Debug)]
struct KernelTypeSlice {
    #[bracket]
    bracket: Bracket,
    #[inside(bracket)]
    ty: KernelTypeScalar,
}

#[derive(Debug)]
struct KernelMeta {
    attr_meta: KernelAttrMeta,
    spec_metas: Vec<KernelSpecMeta>,
    ident: Ident,
    unsafe_token: Option<Unsafe>,
    arg_metas: Vec<KernelArgMeta>,
    block: Block,
}

impl KernelMeta {
    fn desc(&self) -> Result<KernelDesc> {
        let mut kernel_desc = KernelDesc::default();
        kernel_desc.name = self.ident.to_string();
        kernel_desc.safe = self.unsafe_token.is_none();
        for thread in self.attr_meta.threads.iter() {
            let dim = if let Some(lit) = thread.lit.as_ref() {
                let dim = lit.base10_parse::<u32>()?;
                if lit.base10_parse::<u32>()? == 0 {
                    return Err(Error::new_spanned(&lit, "thread dim cannot be 0"));
                }
                dim
            } else {
                1
            };
            kernel_desc.threads.push(dim);
        }
        for arg_meta in self.arg_metas.iter() {
            let scalar_type = arg_meta.scalar_ty.scalar_type;
            if let Some(ident) = arg_meta.attr.as_ref() {
                if ident == "global" {
                    kernel_desc.slice_descs.push(SliceDesc {
                        name: arg_meta.ident.to_string(),
                        scalar_type,
                        mutable: arg_meta.mutable,
                    })
                } else {
                    todo!("arg_meta.attr != \"global\"");
                }
            } else {
                kernel_desc.push_descs.push(PushDesc {
                    name: arg_meta.ident.to_string(),
                    scalar_type,
                });
                kernel_desc
                    .push_descs
                    .sort_by_key(|x| x.scalar_type.size() as i32);
            }
        }
        Ok(kernel_desc)
    }
    fn device_spec_tokens(&self) -> TokenStream2 {
        self.spec_metas
            .iter()
            .map(KernelSpecMeta::declare)
            .collect()
    }
    fn device_fn_def_tokens(&self) -> TokenStream2 {
        self.spec_metas
            .iter()
            .map(|x| {
                let ident = &x.ident;
                let ty = &x.ty.ident;
                quote! {
                    #ident: #ty,
                }
            })
            .chain(
                self.arg_metas
                    .iter()
                    .map(KernelArgMeta::device_fn_def_tokens),
            )
            .collect()
    }
    fn device_fn_call_tokens(&self) -> TokenStream2 {
        self.spec_metas
            .iter()
            .map(|x| {
                let ident = &x.ident;
                quote! {
                    #ident
                }
            })
            .chain(
                self.arg_metas
                    .iter()
                    .map(KernelArgMeta::device_fn_call_tokens),
            )
            .collect()
    }
    fn dispatch_args(&self) -> TokenStream2 {
        let dim = LitInt::new(
            &self.attr_meta.threads.len().to_string(),
            Span2::call_site(),
        );
        let mut tokens = quote! {
            groups: [u32; #dim],
        };
        for arg in self.arg_metas.iter() {
            let ident = &arg.ident;
            let ty = &arg.scalar_ty.ident;
            if let Some(attr) = arg.attr.as_ref() {
                if attr == "global" {
                    let slice_ty = if arg.mutable {
                        format_ident!("SliceMut")
                    } else {
                        format_ident!("Slice")
                    };
                    tokens.extend(quote! {
                        #ident: #slice_ty<#ty>,
                    });
                }
            } else {
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
            if let Some(attr) = arg.attr.as_ref() {
                if attr == "global" {
                    tokens.extend(quote! {
                        #ident.into(),
                    });
                }
            }
        }
        tokens
    }
    fn dispatch_push_args(&self) -> TokenStream2 {
        let mut tokens = TokenStream2::new();
        for arg in self.arg_metas.iter() {
            let ident = &arg.ident;
            if arg.attr.is_none() {
                tokens.extend(quote! {
                    #ident.into(),
                })
            }
        }
        tokens
    }
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
    thread_dim: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct SliceDesc {
    name: String,
    scalar_type: ScalarType,
    mutable: bool,
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
        let compute_threads: Punctuated<_, Comma> = kernel_meta
            .attr_meta
            .threads
            .iter()
            .map(|x| {
                x.lit
                    .as_ref()
                    .cloned()
                    .unwrap_or(LitInt::new("1", Span2::call_site()))
            })
            .collect();
        let threads = &kernel_meta.attr_meta.threads;
        let threads = if dimensionality == 3 {
            quote! {
                ::krnl_core::glam::UVec3::new(#threads)
            }
        } else if dimensionality == 2 {
            quote! {
                ::krnl_core::glam::UVec2::new(#threads)
            }
        } else {
            threads.to_token_stream()
        };
        let compute_def_args: TokenStream2 = kernel_meta
            .arg_metas
            .iter()
            .map(KernelArgMeta::compute_def_tokens)
            .collect();
        let device_fn_def_args = kernel_meta.device_fn_def_tokens();
        let device_fn_call_args = kernel_meta.device_fn_call_tokens();
        let push_consts_ident = format_ident!("__krnl_{ident}PushConsts");
        let push_const_fields = kernel_desc.push_const_fields();
        let (vector_ty, vector_sfx) = if dimensionality == 1 {
            (
                quote! { u32 },
                quote! {
                    .x
                },
            )
        } else if dimensionality == 2 {
            (
                quote! {
                    ::krnl_core::glam::UVec2
                },
                quote! {
                    .truncate()
                },
            )
        } else {
            (quote! { ::krnl_core::glam::UVec3 }, TokenStream2::new())
        };
        let push_struct_tokens = quote! {
            #[cfg(target_arch = "spirv")]
            #[automatically_derived]
            #[repr(C)]
            pub struct #push_consts_ident {
                #push_const_fields
            }
        };
        if kernel_meta.attr_meta.for_each {
            todo!("for_each")
        } else {
            quote! {
                #push_struct_tokens
                #[cfg(target_arch = "spirv")]

                #[::krnl_core::spirv_std::spirv(compute(threads(#compute_threads)))]
                pub fn #ident(
                    #compute_def_args
                    #[allow(unused)]
                    #[spirv(push_constant)]
                    __krnl_push_consts: &#push_consts_ident,
                    #[allow(unused)]
                    #[spirv(global_invocation_id)]
                    global_id: ::krnl_core::glam::UVec3,
                    #[allow(unused)]
                    #[spirv(num_workgroups)]
                    groups: ::krnl_core::glam::UVec3,
                    #[allow(unused)]
                    #[spirv(workgroup_id)]
                    group_id: ::krnl_core::glam::UVec3,
                    #[allow(unused)]
                    #[spirv(local_invocation_id)]
                    thread_id: ::krnl_core::glam::UVec3,
                    #[allow(unused)]
                    #[spirv(local_invocation_index)]
                    thread_index: u32,
                    #[allow(unused)]
                    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)]
                    #kernel_data: &mut [u32],
                ) {
                    /*unsafe {
                        use ::krnl_core::spirv_std::arch::IndexUnchecked;
                        let kernel_data = #kernel_data;
                        *kernel_data.index_unchecked_mut(0) = 1;
                    }*/
                    #unsafe_token fn #ident(
                        #device_fn_def_args
                        #[allow(unused)]
                        global_id: #vector_ty,
                        #[allow(unused)]
                        groups: #vector_ty,
                        #[allow(unused)]
                        group_id: #vector_ty,
                        #[allow(unused)]
                        threads: #vector_ty,
                        #[allow(unused)]
                        thread_id: #vector_ty,
                        #[allow(unused)]
                        thread_index: u32,
                    ) #block
                    #unsafe_token {
                        #ident (
                            #device_fn_call_args
                            global_id #vector_sfx,
                            groups #vector_sfx,
                            group_id #vector_sfx,
                            #threads,
                            thread_id #vector_sfx,
                            thread_index,
                        );
                    }
                }
            }
        }
    };
    let host_tokens = {
        let dispatch_args = kernel_meta.dispatch_args();
        let dispatch_slice_args = kernel_meta.dispatch_slice_args();
        let dispatch_push_args = kernel_meta.dispatch_push_args();
        quote! {
            #[cfg(not(target_arch = "spirv"))]
            #[automatically_derived]
            pub mod #ident {
                use super::__krnl::{
                    anyhow::{self, Result},
                    buffer::{Slice, SliceMut},
                    device::{Device, Kernel as KernelBase, KernelBuilder as KernelBuilderBase},
                    once_cell::sync::Lazy,
                };

                pub struct KernelBuilder {
                    inner: &'static KernelBuilderBase,
                }

                pub fn builder() -> Result<KernelBuilder> {
                    static BUILDER: Lazy<Result<KernelBuilderBase, String>> = Lazy::new(|| {
                        let bytes = __krnl_kernel!(#ident);
                        KernelBuilderBase::from_bytes(bytes).map_err(|e| e.to_string())
                    });
                    match &*BUILDER {
                        Ok(inner) => {
                            debug_assert!(inner.safe());
                            Ok(KernelBuilder { inner })
                        }
                        Err(e) => Err(anyhow::Error::msg(e)),
                    }
                }

                impl KernelBuilder {
                    pub fn build(&self, device: Device) -> Result<Kernel> {
                        let inner = self.inner.build(device)?;
                        Ok(Kernel { inner })
                    }
                }

                pub struct Kernel {
                    inner: KernelBase,
                }

                impl Kernel {
                    pub fn dispatch(&self, #dispatch_args) -> Result<()> {
                        unsafe { self.inner.dispatch(&groups, &[#dispatch_slice_args], &[#dispatch_push_args]) }
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
        let file = syn::parse2(tokens.clone()).expect("unable to parse module tokens");
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
        let cache: KrnlcCache = bincode::deserialize(&cache_bytes).unwrap();
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
                error.replace("recompile with krnlc".to_string());
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
