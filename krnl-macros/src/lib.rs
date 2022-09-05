#![cfg_attr(not(feature = "build"), allow(dead_code))]
#![allow(warnings)]

use derive_syn_parse::Parse;
use krnl_types::{
    __private::raw_module::{Mutability, PushInfo, RawKernelInfo, RawModule, Safety, SliceInfo},
    scalar::ScalarType,
    version::Version,
};
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
use spirv::Capability;
use std::{
    collections::{hash_map::DefaultHasher, HashSet},
    fs,
    hash::{Hash, Hasher},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};
use syn::{
    parse::{Parse, ParseStream},
    parse_quote,
    punctuated::Punctuated,
    token::{And, Bracket, Colon, Colon2, Comma, Eq, Fn, Gt, Lt, Mut, Paren, Pound, Pub, Unsafe},
    Attribute, Block, Error, Ident, ItemMod, LitInt, LitStr, Stmt, Type,
};

type Result<T, E = Error> = core::result::Result<T, E>;

trait IntoSynResult {
    type Output;
    fn into_syn_result(self, span: Span) -> Result<Self::Output>;
}

impl<T, E: std::fmt::Display> IntoSynResult for Result<T, E> {
    type Output = T;
    fn into_syn_result(self, span: Span) -> Result<T> {
        self.or_else(|e| Err(Error::new(span, e)))
    }
}

fn get_hash<T: Hash>(x: &T) -> u64 {
    let mut h = DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

fn crate_is_krnl() -> bool {
    use std::env::var;
    var("CARGO_PKG_HOMEPAGE").as_deref() == Ok("https://github.com/charles-r-earp/krnl")
        && var("CARGO_CRATE_NAME").as_deref() == Ok("krnl")
}

fn get_krnl_path() -> syn::Path {
    // TODO: detect doc / tests?
    if crate_is_krnl() {
        parse_quote! {
            crate
        }
    } else {
        parse_quote! {
            krnl
        }
    }
}

#[derive(Parse, Debug)]
struct KernelAttributes {
    #[call(Punctuated::parse_terminated)]
    attr: Punctuated<KernelAttribute, Comma>,
}

#[derive(Parse, Debug)]
struct KernelAttribute {
    ident: Ident,
    #[parse_if(ident == "vulkan")]
    vulkan: Option<Vulkan>,
    #[parse_if(ident == "threads")]
    threads: Option<Threads>,
    #[parse_if(ident == "capabilities" || ident == "extensions")]
    features: Option<TargetFeatureList>,
}

#[derive(Parse, Debug)]
struct Vulkan {
    #[paren]
    paren: Paren,
    #[inside(paren)]
    version: LitStr,
}

impl Vulkan {
    fn version(&self) -> Result<Version> {
        Version::from_str(&self.version.value()).into_syn_result(self.version.span())
    }
}

#[derive(Parse, Debug)]
struct Threads {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_terminated)]
    threads: Punctuated<LitInt, Comma>,
}

#[derive(Parse, Debug)]
struct TargetFeatureList {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_terminated)]
    features: Punctuated<LitStr, Comma>,
}

#[derive(Parse, Debug)]
struct Kernel {
    #[allow(unused)]
    pub_token: Pub,
    unsafe_token: Option<Unsafe>,
    #[allow(unused)]
    fn_token: Fn,
    ident: Ident,
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_terminated)]
    args: Punctuated<KernelArg, Comma>,
    block: Block,
}

#[derive(Parse, Debug)]
struct KernelArg {
    #[allow(unused)]
    pound: Option<Pound>,
    #[parse_if(pound.is_some())]
    attr: Option<KernelArgAttribute>,
    ident: Ident,
    #[allow(unused)]
    colon: Colon,
    #[allow(unused)]
    and: Option<And>,
    mut_token: Option<Mut>,
    #[parse_if(and.is_none())]
    ty: Option<Ident>,
    #[parse_if(attr.is_none() && and.is_some())]
    slice_arg: Option<KernelSliceArg>,
    #[parse_if(attr.is_some() && and.is_some())]
    group_arg: Option<KernelGroupArg>,
}

#[derive(Debug)]
struct KernelSliceArg {
    wrapper_ty: Option<Ident>,
    slice_ty: Option<Ident>,
    elem_ty: Ident,
}

impl Parse for KernelSliceArg {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ident1: Ident = input.parse()?;
        if input.peek(Lt) {
            input.parse::<Lt>()?;
            let ident2 = input.parse()?;
            if input.peek(Lt) {
                input.parse::<Lt>()?;
                let elem_ty = input.parse()?;
                input.parse::<Gt>()?;
                input.parse::<Gt>()?;
                Ok(Self {
                    wrapper_ty: Some(ident1),
                    slice_ty: Some(ident2),
                    elem_ty,
                })
            } else {
                input.parse::<Gt>()?;
                Ok(Self {
                    wrapper_ty: None,
                    slice_ty: Some(ident1),
                    elem_ty: ident2,
                })
            }
        } else {
            Ok(Self {
                wrapper_ty: None,
                slice_ty: None,
                elem_ty: ident1,
            })
        }
    }
}

#[derive(Clone, Debug)]
struct KernelGroupArg {
    wrapper_ty: Option<Ident>,
    ty: Type,
}

impl Parse for KernelGroupArg {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        if input.peek2(Lt) {
            let wrapper_ty = input.parse()?;
            input.parse::<Lt>()?;
            let ty = input.parse()?;
            input.parse::<Gt>()?;
            Ok(Self { wrapper_ty, ty })
        } else {
            Ok(Self {
                wrapper_ty: None,
                ty: input.parse()?,
            })
        }
    }
}

#[derive(Parse, Debug)]
struct KernelArgAttribute {
    #[bracket]
    bracket: Bracket,
    #[inside(bracket)]
    ident: Ident,
}

#[derive(Clone, Parse, Debug)]
struct TypedArg {
    #[call(Attribute::parse_outer)]
    attr: Vec<Attribute>,
    ident: Ident,
    #[allow(unused)]
    colon: Colon,
    ty: Type,
}

impl ToTokens for TypedArg {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        for attr in self.attr.iter() {
            attr.to_tokens(tokens);
        }
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
    }
}

static BUILTINS: &[(&str, &str)] = &[
    ("global_threads", "UVec3"),
    ("global_id", "UVec3"),
    ("groups", "UVec3"),
    ("group_id", "UVec3"),
    ("subgroup_id", "UVec3"),
    ("subgroups", "UVec3"),
    ("threads", "UVec3"),
    ("thread_id", "UVec3"),
    ("thread_index", "u32"),
    ("elements", "u32"),
    ("element_index", "u32"),
];

#[derive(Debug)]
struct KernelMeta {
    info: RawKernelInfo,
    has_vulkan_version: bool,
    thread_lits: Punctuated<LitInt, Comma>,
    builtins: HashSet<&'static str>,
    outer_args: Punctuated<TypedArg, Comma>,
    inner_args: Punctuated<TypedArg, Comma>,
    group_args: Vec<(Ident, KernelGroupArg)>,
    push_consts: Punctuated<TypedArg, Comma>,
    push_consts_ident: Option<Ident>,
}

impl KernelMeta {
    fn new(kernel_attr: &KernelAttributes, kernel: &Kernel) -> Result<Self> {
        let safety = if kernel.unsafe_token.is_some() {
            Safety::Unsafe
        } else {
            Safety::Safe
        };
        let mut info = RawKernelInfo {
            name: kernel.ident.to_string(),
            vulkan_version: Version::from_major_minor(0, 0),
            capabilities: Vec::new(),
            extensions: Vec::new(),
            safety,
            slice_infos: Vec::new(),
            push_infos: Vec::new(),
            num_push_words: 0,
            elementwise: false,
            threads: Vec::new(),
            spirv: None,
        };
        let mut has_vulkan_version = false;
        let mut thread_lits = Punctuated::<LitInt, Comma>::new();
        for attr in kernel_attr.attr.iter() {
            let attr_string = attr.ident.to_string();
            if let Some(vulkan) = attr.vulkan.as_ref() {
                info.vulkan_version = vulkan.version()?;
                has_vulkan_version = true;
            } else if attr_string == "elementwise" {
                info.elementwise = true;
            } else if let Some(threads) = attr.threads.as_ref() {
                if !info.threads.is_empty() {
                    return Err(Error::new_spanned(&thread_lits, "threads already declared"));
                }
                for x in threads.threads.iter() {
                    info.threads.push(x.base10_parse()?);
                    thread_lits.push(x.clone());
                }
                while thread_lits.len() < 3 {
                    thread_lits.push(LitInt::new("1", attr.ident.span()));
                }
            } else if let Some(features) = attr.features.as_ref() {
                match attr_string.as_str() {
                    "capabilities" => {
                        for cap in features.features.iter() {
                            let cap_string = cap.value();
                            if let Ok(cap) = Capability::from_str(&cap_string) {
                                info.capabilities.push(cap);
                            } else {
                                return Err(Error::new_spanned(cap, "unknown capability, see https://docs.rs/spirv/latest/spirv/enum.Capability.html"));
                            }
                        }
                    }
                    "extensions" => {
                        for ext in features.features.iter() {
                            info.extensions.push(ext.value());
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                return Err(Error::new_spanned(&attr.ident, "unknown attribute, expected \"target\", \"elementwise\", \"threads\", \"capabilities\", or \"extensions\""));
            }
        }
        if info.threads.is_empty() {
            return Err(Error::new(
                Span::call_site(),
                "threads(..) must be specified, ie #[kernel(threads(256))]",
            ));
        }
        if info.elementwise && info.threads.len() != 1 {
            return Err(Error::new_spanned(
                &thread_lits,
                "can only use 1 dimensional threads in elementwise kernel",
            ));
        }
        let mut builtins = HashSet::<&'static str>::new();
        let mut outer_args = Punctuated::<TypedArg, Comma>::new();
        let mut inner_args = Punctuated::<TypedArg, Comma>::new();
        let mut group_args = Vec::<(Ident, KernelGroupArg)>::new();
        let mut push_consts = Vec::<(ScalarType, TypedArg)>::new();
        let mut set_lit = LitInt::new("0", Span::call_site());
        let mut binding = 0;
        for arg in kernel.args.iter() {
            let arg_ident = &arg.ident;
            let ident_string = arg_ident.to_string();
            let mut_token = &arg.mut_token;
            let mutable = mut_token.is_some();
            if let Some(ty) = arg.ty.as_ref() {
                if let Some(attr) = arg.attr.as_ref() {
                    if attr.ident == "builtin" {
                        if info.elementwise
                            && !matches!(ident_string.as_str(), "elements" | "element_index")
                        {
                            return Err(Error::new_spanned(&arg_ident, format!("builtin `{ident_string}` can not be used in elementwise kernel, use `elements` or `element_index` instead")));
                        }
                        let (builtin, builtin_ty) = BUILTINS
                            .iter()
                            .find(|x| x.0 == ident_string.as_str())
                            .ok_or_else(|| {
                                let mut msg = "unknown builtin, expected ".to_string();
                                for (b, _) in BUILTINS.iter() {
                                    msg.push('`');
                                    msg.push_str(b);
                                    msg.push('`');
                                    if b != &BUILTINS.last().unwrap().0 {
                                        msg.push_str(", ");
                                    }
                                }
                                msg.push('.');
                                Error::new_spanned(&attr.ident, msg)
                            })?;
                        let arg_ty = arg
                            .ty
                            .as_ref()
                            .or_else(|| {
                                dbg!(&arg);
                                None
                            })
                            .unwrap();
                        let ty_string = arg_ty.to_string();
                        if &ty_string != builtin_ty {
                            return Err(Error::new_spanned(&arg_ty, "expected `{builtin_ty}`"));
                        }
                        match ident_string.as_str() {
                            "global_threads" => {
                                if !builtins.contains("groups") {
                                    outer_args.push(parse_quote! {
                                        #[spirv(num_workgroups)]
                                        groups: UVec3
                                    });
                                }
                                builtins.extend(["global_threads", "groups", "threads"]);
                            }
                            "threads" => {
                                builtins.insert("threads");
                            }
                            "elements" => {
                                inner_args.push(parse_quote! {
                                    #arg_ident: u32
                                });
                            }
                            "element_index" => {
                                inner_args.push(parse_quote! {
                                    #arg_ident: u32
                                });
                            }
                            _ => {
                                let spirv_attr = match ident_string.as_str() {
                                    "global_id" => "global_invocation_id",
                                    "groups" => "num_workgroups",
                                    "group_id" => "workgroup_id",
                                    "subgroup_id" => "subgroup_id",
                                    "thread_id" => "local_invocation_id",
                                    "thread_index" => "local_invocation_index",
                                    _ => unreachable!("unexpected spirv_attr {ident_string:?}"),
                                };
                                let spirv_attr = Ident::new(spirv_attr, arg.ident.span());
                                outer_args.push(parse_quote! {
                                    #[spirv(#spirv_attr)]
                                    #arg_ident: #arg_ty
                                });
                                builtins.insert(builtin);
                            }
                        }
                        inner_args.push(parse_quote! {
                            #arg_ident: #arg_ty
                        });
                    } else {
                        return Err(Error::new_spanned(
                            &attr.ident,
                            "unknown attribute, expected \"builtin\" or \"subgroup\"",
                        ));
                    }
                } else {
                    if ident_string.starts_with("__krnl") {
                        return Err(Error::new_spanned(&arg_ident, "\"__krnl\" is reserved"));
                    }
                    let push_ty = arg.ty.as_ref().unwrap();
                    let push_ty_string = push_ty.to_string();
                    let scalar_type = ScalarType::from_str(&push_ty_string)
                        .map_err(|_| Error::new_spanned(&push_ty, "expected a scalar"))?;

                    let typed_arg: TypedArg = parse_quote! {
                        #arg_ident: #push_ty
                    };
                    push_consts.push((scalar_type, typed_arg.clone()));
                    inner_args.push(typed_arg);
                }
            } else if let Some(slice_arg) = arg.slice_arg.as_ref() {
                let elementwise = slice_arg.slice_ty.is_none();
                let elem_ty = &slice_arg.elem_ty;
                let elem_ty_string = elem_ty.to_string();
                if elementwise && !info.elementwise {
                    return Err(Error::new_spanned(elem_ty, format!("can not use `&{}{elem_ty_string}` outside of elementwise kernel, add `elementwise` to `#[kernel(..)]` attributes to enable", if mutable { "mut " } else { "" })));
                }
                if let Some(wrapper_ty) = slice_arg.wrapper_ty.as_ref() {
                    if wrapper_ty == "GlobalMut" {
                        if info.elementwise {
                            return Err(Error::new_spanned(
                                wrapper_ty,
                                "can not use `GlobalMut<_>` in elementwise kernel",
                            ));
                        }
                        if !mutable {
                            return Err(Error::new_spanned(
                                arg.and.as_ref().unwrap(),
                                "expected `&mut _`",
                            ));
                        }
                    } else {
                        return Err(Error::new_spanned(wrapper_ty, "expected `GlobalMut<_>`"));
                    }
                }
                if let Some(slice_ty) = slice_arg.slice_ty.as_ref() {
                    if slice_ty == "SliceMut" {
                        if slice_arg.wrapper_ty.is_none() {
                            return Err(Error::new_spanned(mut_token, "`SliceMut<_>` can not be used as a kernel argument directly, use `krnl_core::mem::GlobalMut<SliceMut<_>>` instead"));
                        }
                    } else if slice_ty == "Slice" {
                        if let Some(mut_token) = mut_token {
                            return Err(Error::new_spanned(mut_token, "unexpected `mut`"));
                        }
                    }
                    if let Some(wrapper_ty) = slice_arg.wrapper_ty.as_ref() {
                        inner_args.push(parse_quote! {
                            #arg_ident: & #mut_token #wrapper_ty<#slice_ty<#elem_ty>>
                        });
                    } else {
                        inner_args.push(parse_quote! {
                            #arg_ident: & #mut_token #slice_ty<#elem_ty>
                        });
                    }
                } else {
                    inner_args.push(parse_quote! {
                        #arg_ident: & #mut_token #elem_ty
                    })
                }
                {
                    set_lit.set_span(arg_ident.span());
                    let binding_lit = LitInt::new(&binding.to_string(), arg_ident.span());
                    outer_args.push(parse_quote! {
                        #[spirv(descriptor_set = #set_lit, binding = #binding_lit, storage_buffer)]
                        #arg_ident: & #mut_token [#elem_ty]
                    });
                    binding += 1;
                }
                let mutability = if mutable {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                let scalar_type = ScalarType::from_str(&elem_ty_string)
                    .map_err(|_| Error::new_spanned(&elem_ty, "expected a scalar"))?;
                info.slice_infos.push(SliceInfo {
                    name: arg_ident.to_string(),
                    scalar_type,
                    mutability,
                    elementwise,
                });
            } else if let Some(group_arg) = arg.group_arg.as_ref() {
                let attr = arg.attr.as_ref().expect("group_arg.attr.is_some()");
                let group_ty = &group_arg.ty;
                if attr.ident == "group" {
                    if let Some(wrapper_ty) = group_arg.wrapper_ty.as_ref() {
                        if wrapper_ty != "GroupUninitMut" {
                            return Err(Error::new_spanned(
                                group_ty,
                                "expected `krnl_core::mem::GroupUninitMut<_>`",
                            ));
                        }
                        outer_args.push(parse_quote! {
                            #[spirv(workgroup)]
                            #arg_ident: & #mut_token #group_ty
                        });
                        inner_args.push(parse_quote! {
                            #arg_ident: & #mut_token #wrapper_ty<#group_ty>
                        });
                    } else {
                        return Err(Error::new_spanned(
                            group_ty,
                            "expected `krnl_core::mem::GroupUninitMut<_>`",
                        ));
                    }
                } else if attr.ident == "subgroup" {
                    outer_args.push(parse_quote! {
                        #[spirv(subgroup)]
                        #arg_ident: #group_ty
                    });
                    inner_args.push(parse_quote! {
                        #arg_ident: #group_ty
                    });
                } else {
                    return Err(Error::new_spanned(
                        &arg_ident,
                        "expected `krnl_core::mem::GroupUninitMut<_>`",
                    ));
                }
                group_args.push((arg_ident.clone(), group_arg.clone()));
            }
            if let Some(arg_attr) = arg.attr.as_ref() {
                let attr_string = arg_attr.ident.to_string();
                match attr_string.as_str() {
                    "subgroup" => {
                        if info.elementwise {
                            return Err(Error::new_spanned(
                                &arg_ident,
                                "`#[subgroup]` can not be used in elementwise kernel",
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
        for slice_info in info.slice_infos.iter() {
            let offset_pad = format_ident!("__krnl_offset_pad_{}", slice_info.name);
            push_consts.push((
                ScalarType::U32,
                parse_quote! {
                    #offset_pad: u32
                },
            ));
        }
        if info.elementwise {
            builtins.extend([
                "global_threads",
                "global_id",
                "groups",
                "threads",
                "thread_id",
            ]);
            outer_args.push(parse_quote! {
                #[spirv(global_invocation_id)]
                global_id: ::krnl_core::glam::UVec3
            });
            outer_args.push(parse_quote! {
                #[spirv(num_workgroups)]
                groups: ::krnl_core::glam::UVec3
            });
            outer_args.push(parse_quote! {
                #[spirv(local_invocation_id)]
                thread_id: ::krnl_core::glam::UVec3
            });
        }
        push_consts.sort_by_key(|x| x.0.size());
        let mut offset = 0;
        for (scalar_type, typed_arg) in push_consts.iter().rev() {
            info.push_infos.push(PushInfo {
                name: typed_arg.ident.to_string(),
                scalar_type: *scalar_type,
                offset: offset,
            });
            offset += scalar_type.size() as u32;
        }
        info.num_push_words = offset / 4 + if offset % 4 != 0 { 1 } else { 0 };
        let mut push_consts = push_consts
            .into_iter()
            .rev()
            .map(|x| x.1)
            .collect::<Punctuated<_, Comma>>();
        for i in 0..offset % 4 {
            let field = format_ident!("__krnl_pad{}", i);
            push_consts.push(parse_quote! {
                #field: u8
            });
        }
        let push_consts_ident = if !push_consts.is_empty() {
            let ident = format_ident!("__{}PushConsts", kernel.ident);
            outer_args.push(parse_quote! {
                #[spirv(push_constant)] push_consts: &#ident
            });
            Some(ident)
        } else {
            None
        };
        Ok(Self {
            info,
            has_vulkan_version,
            thread_lits,
            builtins,
            outer_args,
            inner_args,
            group_args,
            push_consts,
            push_consts_ident,
        })
    }
}

#[allow(unused_variables)]
fn kernel_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let span = Span::call_site();
    let kernel_attr: KernelAttributes = syn::parse(attr)?;
    let kernel: Kernel = syn::parse(item)?;
    let meta = KernelMeta::new(&kernel_attr, &kernel)?;
    let info = &meta.info;
    let builtins = &meta.builtins;
    let slice_infos = &info.slice_infos;
    let kernel_ident = &kernel.ident;
    let block = &kernel.block;
    let thread_lits = &meta.thread_lits;
    let full_thread_lits = thread_lits
        .iter()
        .cloned()
        .chain([LitInt::new("1", span), LitInt::new("1", span)])
        .take(3)
        .collect::<Punctuated<_, Comma>>();
    let mut builtin_stmts = Vec::<Stmt>::new();
    let outer_args = meta.outer_args;
    let inner_args = meta.inner_args;
    let group_args = meta.group_args;
    let call_args = inner_args
        .iter()
        .map(|x| &x.ident)
        .collect::<Punctuated<_, Comma>>();
    let push_consts = meta.push_consts;
    let push_const_idents = push_consts
        .iter()
        .map(|x| &x.ident)
        .collect::<Punctuated<_, Comma>>();
    if builtins.contains("threads") {
        builtin_stmts.push(parse_quote! {
            let threads = ::krnl_core::glam::UVec3::new(#full_thread_lits);
        });
    }
    if builtins.contains("global_threads") {
        builtin_stmts.push(parse_quote! {
            let global_threads = groups * threads;
        });
    }
    if let Some(push_consts_ident) = meta.push_consts_ident.as_ref() {
        builtin_stmts.push(parse_quote! {
            let &#push_consts_ident {
                #push_const_idents
            } = push_consts;
        });
    }
    for slice_info in slice_infos {
        let ident = format_ident!("{}", slice_info.name);
        let offset_pad = format_ident!("__krnl_offset_pad_{}", slice_info.name);
        let (slice_ty, slice_raw_fn) = if slice_info.mutability.is_mutable() {
            ("SliceMut", "__from_raw_parts_mut")
        } else {
            ("Slice", "__from_raw_parts")
        };
        let slice_ty = format_ident!("{}", slice_ty);
        let slice_raw_fn = format_ident!("{}", slice_raw_fn);
        let mut_token = if slice_info.mutability.is_mutable() {
            Some(Mut::default())
        } else {
            None
        };
        builtin_stmts.push(parse_quote! {
            let ref #mut_token #ident = unsafe {
                use ::krnl_core::slice::#slice_ty;
                let pad = (#offset_pad & 255) as usize;
                let offset = (#offset_pad >> 8) as usize;
                let len = #ident .len() - (offset + pad);
                #slice_ty :: #slice_raw_fn (#ident, offset, len)
            };
        });
        if slice_info.mutability.is_mutable() && !slice_info.elementwise {
            builtin_stmts.push(parse_quote! {
                let ref mut #ident = ::krnl_core::mem::GlobalMut::__new(#ident);
            });
        }
    }
    for (ident, group_arg) in group_args.iter() {
        if let Some(wrapper_ty) = group_arg.wrapper_ty.as_ref() {
            builtin_stmts.push(parse_quote! {
                let ref mut #ident = #wrapper_ty::__new(#ident);
            })
        }
    }
    let mut elementwise_stmts = Vec::<Stmt>::new();
    if meta.info.elementwise {
        if let Some(slice_info) = meta.info.slice_infos.iter().find(|x| x.elementwise) {
            let ident = format_ident!("{}", slice_info.name);
            builtin_stmts.push(parse_quote! {
                let elements = #ident.len() as u32;
            });
        } else {
            builtin_stmts.push(parse_quote! {
                let elements = 0;
            })
        }
        for slice_info in meta.info.slice_infos.iter().filter(|x| x.elementwise) {
            let ident = format_ident!("{}", slice_info.name);
            let mut_token = match slice_info.mutability {
                Mutability::Mutable => Some(Mut::default()),
                Mutability::Immutable => None,
            };
            elementwise_stmts.push(parse_quote! {
                let #ident = & #mut_token #ident[element_index as usize];
            });
        }
    }
    let features = meta
        .info
        .capabilities
        .iter()
        .map(|x| format_ident!("{}", format!("{x:?}")))
        .chain(
            meta.info
                .extensions
                .iter()
                .map(|x| format_ident!("ext:{}", x)),
        )
        .collect::<Punctuated<_, Comma>>();
    let mut output = TokenStream2::new();
    if let Some(push_consts_ident) = meta.push_consts_ident.as_ref() {
        quote! {
            #[cfg(all(target_arch = "spirv", #features))]
            #[allow(non_camel_case_types)]
            #[repr(C)]
            pub struct #push_consts_ident {
                #push_consts
            }
        }
        .to_tokens(&mut output);
    }
    let unsafe_token = &kernel.unsafe_token;
    let call = if let Some(unsafe_token) = kernel.unsafe_token.as_ref() {
        quote! {
            #unsafe_token {
                #kernel_ident(#call_args);
            }
        }
    } else {
        quote! {
            #kernel_ident(#call_args);
        }
    };
    let generated_body = if meta.info.elementwise {
        quote! {
            let mut element_index = global_id.x;
            while element_index < elements {
                #(#elementwise_stmts)*
                #call
                element_index += global_threads.x;
            }
        }
    } else {
        call
    };
    quote! {
        #[cfg(all(target_arch = "spirv", #features))]
        #[spirv(compute(threads(#thread_lits)))]
        pub fn #kernel_ident (
            #outer_args
        ) {
            #(#builtin_stmts)*
            fn #kernel_ident (
                #inner_args
            ) #block
            #generated_body
        }
    }
    .to_tokens(&mut output);
    let module_path = std::env::var("KRNL_MODULE_PATH").ok();
    if let Some(module_path) = module_path.as_ref() {
        let bytes = fs::read(&module_path).into_syn_result(span)?;
        let mut raw_module: RawModule = bincode::deserialize(&bytes).into_syn_result(span)?;
        let mut info = meta.info;
        if !meta.has_vulkan_version {
            info.vulkan_version = raw_module.vulkan_version;
        }
        raw_module.kernels.insert(info.name.clone(), Arc::new(info));
        let bytes = bincode::serialize(&raw_module).into_syn_result(span)?;
        fs::write(&module_path, &bytes).into_syn_result(span)?;
    }
    // dbg!(output.to_string().split('\n').collect::<Vec<_>>());
    Ok(output.into())
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match kernel_impl(attr, item) {
        Ok(x) => x,
        Err(e) => e.into_compile_error().into(),
    }
}

#[derive(Parse, Debug)]
struct ModuleAttributes {
    #[call(Punctuated::parse_terminated)]
    attr: Punctuated<ModuleAttribute, Comma>,
}

impl ModuleAttributes {
    fn module_path(&self) -> Option<&ModulePath> {
        self.attr.iter().find_map(|x| x.path.as_ref())
    }
}

#[derive(Parse, Debug)]
struct ModuleAttribute {
    ident: Ident,
    #[parse_if(ident == "path")]
    path: Option<ModulePath>,
    #[parse_if(ident == "vulkan")]
    vulkan: Option<Vulkan>,
    #[parse_if(ident == "dependency")]
    dependency: Option<Dependency>,
    #[parse_if(ident == "attr")]
    attr: Option<ModuleAttr>,
}

#[derive(Parse, Debug)]
struct ModulePath {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_terminated)]
    segments: Punctuated<Ident, Colon2>,
}

impl ModulePath {
    fn module_name_prefix(&self) -> String {
        let mut string = String::new();
        for segment in self.segments.iter() {
            string.push_str(&format!("{segment}_"));
        }
        string
    }
    fn file_path(&self) -> PathBuf {
        let mut path = PathBuf::new();
        for segment in self.segments.iter() {
            path.push(segment.to_string());
        }
        path
    }
}

#[derive(Parse, Debug)]
struct Dependency {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    name: LitStr,
    #[inside(paren)]
    #[allow(unused)]
    comma: Comma,
    #[inside(paren)]
    #[call(Punctuated::parse_terminated)]
    key_values: Punctuated<DependencyKeyValue, Comma>,
}

impl Dependency {
    fn to_toml_string(&self) -> Result<String> {
        let mut s = format!("{} = {{ ", self.name.value());
        for (i, kv) in self.key_values.iter().enumerate() {
            s.push_str(&kv.to_toml_string()?);
            if i != self.key_values.len() - 1 {
                s.push(',');
            }
        }
        s.push_str(" }");
        Ok(s)
    }
}

#[derive(Parse, Debug)]
struct DependencyKeyValue {
    key: Ident,
    #[allow(unused)]
    eq: Eq,
    value: Option<LitStr>,
    #[parse_if(value.is_none())]
    value_list: Option<DependencyValueList>,
}

impl DependencyKeyValue {
    fn to_toml_string(&self) -> Result<String> {
        let key_string = self.key.to_string();
        let mut s = format!("{key_string} = ");
        match key_string.as_str() {
            "version" | "git" | "branch" | "tag" | "path" => {
                let value = self
                    .value
                    .as_ref()
                    .ok_or_else(|| Error::new_spanned(&self.key, "expected a str, found a list"))?;
                let mut value_string = value.value();
                if key_string.as_str() == "path" {
                    value_string = PathBuf::from(value_string)
                        .canonicalize()
                        .into_syn_result(value.span())?
                        .to_string_lossy()
                        .into_owned();
                }
                s.push('"');
                s.push_str(&value_string);
                s.push('"');
            }
            "features" => {
                let value_list = self
                    .value_list
                    .as_ref()
                    .ok_or_else(|| Error::new_spanned(&self.key, "expected a list, found a str"))?;
                s.push_str(&value_list.to_toml_string());
            }
            _ => {
                return Err(Error::new_spanned(
                    &self.key,
                    format!("unexpected dependency key {key_string}"),
                ));
            }
        }
        Ok(s)
    }
}

#[derive(Parse, Debug)]
struct DependencyValueList {
    #[allow(unused)]
    #[bracket]
    bracket: Bracket,
    #[inside(bracket)]
    #[call(Punctuated::parse_terminated)]
    values: Punctuated<LitStr, Comma>,
}

impl DependencyValueList {
    fn to_toml_string(&self) -> String {
        let mut s = "[".to_string();
        for (i, value) in self.values.iter().enumerate() {
            s.push('"');
            s.push_str(&value.value());
            s.push('"');
            if i != self.values.len() - 1 {
                s.push(',');
            }
        }
        s.push(']');
        s
    }
}

#[derive(Parse, Debug)]
struct ModuleAttr {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    tokens: TokenStream2,
}

#[cfg(feature = "build")]
fn build_module(module_attr: &ModuleAttributes, module: &ItemMod, module_prefix_path: &PathBuf, invocation_hash: u64) -> Result<()> {
    use std::process::Command;
    let span = Span::call_site();
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").into_syn_result(span)?);
    let krnl_dir = manifest_dir.join("target").join(".krnl");
    fs::create_dir_all(&krnl_dir).into_syn_result(span)?;
    let builder_dir = krnl_dir.join("builder");
    let toolchain_toml = r#"[toolchain]
channel = "nightly-2022-07-04"
components = ["rust-src", "rustc-dev", "llvm-tools-preview"]"#;
    if true /* !builder_dir.exists() */ {
        if !builder_dir.exists() {
            fs::create_dir(&builder_dir).into_syn_result(span)?;
        }
        let manifest = r#"[package]
name = "builder"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
krnl-builder = { path = "/home/charles/Documents/rust/krnl/krnl-builder" }
bincode = "1.3.3"
"#;
        fs::write(builder_dir.join("Cargo.toml"), manifest).into_syn_result(span)?;
        fs::write(builder_dir.join("rust-toolchain.toml"), toolchain_toml).into_syn_result(span)?;

        let src_dir = builder_dir.join("src");
        if !src_dir.exists() {
            fs::create_dir(&src_dir).into_syn_result(span)?;
        }
        let main = r#"use krnl_builder::{ModuleBuilder, version::Version};
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let crate_path = &args[1];
    let vulkan_version = Version::from_str(&args[2])?;
    let hash = u64::from_str(&args[3])?;
    let module_path = &args[4];
    let module = ModuleBuilder::new(crate_path)
        .vulkan(vulkan_version)
        .build()?;
    let mut bytes = hash.to_be_bytes().to_vec();
    bincode::serialize_into(&mut bytes, &module.__raw())?;
    std::fs::write(module_path, &bytes)?;
    Ok(())
}"#;
        fs::write(src_dir.join("main.rs"), main).into_syn_result(span)?;
    }
    let module_name_prefix = module_attr.module_path().map_or(String::new(), ModulePath::module_name_prefix);
    let module_name = module.ident.to_string();
    let crate_name = format!("{module_name_prefix}{module_name}");
    let mut tokens = quote! {
        #![cfg_attr(
            target_arch = "spirv",
            no_std,
            feature(register_attr),
            register_attr(spirv),
            deny(warnings),
        )]
    };
    let mut vulkan_version = None;
    let mut has_krnl_core_dep = false;
    let mut dependencies = String::new();
    for attr in module_attr.attr.iter() {
        if let Some(vulkan) = attr.vulkan.as_ref() {
            vulkan_version.replace(vulkan.version()?);
        } else if let Some(dep) = attr.dependency.as_ref() {
            if dep.name.value() == "krn-core" {
                has_krnl_core_dep = true;
            }
            dependencies.push_str(&dep.to_toml_string()?);
            dependencies.push('\n');
        } else if let Some(attr) = attr.attr.as_ref() {
            let attr = &attr.tokens;
            quote! {
                #![#attr]
            }
            .to_tokens(&mut tokens);
        }
    }
    let crate_dir = krnl_dir
        .join("modules")
        .join(&module_prefix_path)
        .join(&module_name);
    if !crate_dir.exists() {
        fs::create_dir_all(&crate_dir).into_syn_result(span)?;
    }
    let mut manifest = format!("[package]\nname = {crate_name:?}");
    manifest.push_str(
        r#"
version = "0.1.0"
edition = "2021"
publish = false

[lib]
crate-type = ["dylib"]

[dependencies]
"#);
    let vulkan_version = vulkan_version
        .ok_or_else(|| Error::new(span, "expected a default vulkan version, ie `vulkan(1, 1)`"))?;
    if !has_krnl_core_dep {
        let krnl_core_path = PathBuf::from("krnl-core")
            .canonicalize()
            .into_syn_result(span)?
            .to_string_lossy()
            .into_owned();
        manifest.push_str(&format!(
            "krnl-core = {{ path = {krnl_core_path:?}, features = [\"spirv-panic\"] }}\n"
        ));
    }
    manifest.push_str(&dependencies);
    if let Some(content) = module.content.as_ref() {
        for item in content.1.iter() {
            item.to_tokens(&mut tokens);
        }
    }
    let source = tokens.to_string();
    fs::write(crate_dir.join("Cargo.toml"), manifest).into_syn_result(span)?;
    let src_dir = crate_dir.join("src");
    if !src_dir.exists() {
        fs::create_dir(&src_dir).into_syn_result(span)?;
    }
    fs::write(src_dir.join("lib.rs"), &source).into_syn_result(span)?;
    let _ = Command::new("cargo")
        .arg("fmt")
        .current_dir(&crate_dir)
        .status();
    let module_dir = manifest_dir
        .join(".krnl")
        .join("modules")
        .join(&module_prefix_path);
    fs::create_dir_all(&module_dir).into_syn_result(span)?;
    let module_path = module_dir.join(&module_name).with_extension("bincode");
    let status = Command::new("cargo")
        .args(&[
            "run",
            "--release",
            "--",
            &*crate_dir.to_string_lossy(),
            &vulkan_version.to_string(),
            &invocation_hash.to_string(),
            &*module_path.to_string_lossy(),
        ])
        .current_dir(&builder_dir)
        .env("RUSTUP_TOOLCHAIN", "")
        .status()
        .into_syn_result(span)?;
    if !status.success() {
        return Err(Error::new(
            span,
            format!("Failed to compile module {module_name:?}!"),
        ));
    }
    Ok(())
}

fn module_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    #[allow(unused)]
    let module_attr: ModuleAttributes = syn::parse(attr.clone())?;
    let mut module: ItemMod = syn::parse(item.clone())?;
    let attr = TokenStream2::from(attr);
    let item = TokenStream2::from(item);
    let module_prefix_path = module_attr.module_path().map_or(PathBuf::new(), ModulePath::file_path);
    let invocation = quote! {
        #[module(#attr)]
        #item
    }
    .to_string();
    let invocation_hash = get_hash(&invocation);
    #[cfg(feature = "build")]
    {
        if std::env::var("CARGO_PRIMARY_PACKAGE").is_ok() {
            build_module(&module_attr, &module, &module_prefix_path, invocation_hash)?;
        }
    }
    let module_name = module.ident.to_string();
    let module_prefix_path_string = module_prefix_path.to_string_lossy();
    let panic_msg = format!("module `{module_name}` has been modified, rebuild with `cargo +nightly build --features krnl/build`");
    let krnl = if crate_is_krnl() {
        quote! {
            crate
        }
    } else {
        quote! {
            ::krnl
        }
    };
    let module_fn = parse_quote! {
        pub fn module() -> #krnl::anyhow::Result<#krnl::kernel::Module> {
            use #krnl::{bincode, kernel::Module, krnl_core::__private::raw_module::RawModule};
            use ::std::sync::Arc;
            static BYTES: &[u8] = {
                let bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/.krnl/modules/", #module_prefix_path_string, "/", #module_name, ".bincode"));
                let hash_bytes = [
                    bytes[0],
                    bytes[1],
                    bytes[2],
                    bytes[3],
                    bytes[4],
                    bytes[5],
                    bytes[6],
                    bytes[7],
                ];
                let hash = u64::from_be_bytes(hash_bytes);
                if hash != #invocation_hash {
                    panic!(#panic_msg);
                }
                bytes.as_slice()
            };
            let raw_module: RawModule = bincode::deserialize(&BYTES[8..])?;
            Ok(Module::__from_raw(Arc::new(raw_module)))
        }
    };
    module.content.as_mut().expect("module.context.is_some()").1.push(module_fn);
    Ok(module.to_token_stream().into())
}

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    match module_impl(attr, item) {
        Ok(x) => x,
        Err(e) => e.into_compile_error().into(),
    }
}
