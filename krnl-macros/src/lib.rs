#![allow(warnings)]
use derive_syn_parse::Parse;
use proc_macro::TokenStream;
use proc_macro2::{Span as Span2, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
use siphasher::sip::SipHasher13;
use std::{
    collections::{HashMap, HashSet},
    hash::Hasher,
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
    Result, Stmt, Type, Visibility,
};

macro_rules! todo {
    () => {
        std::todo!("[{}:{}]", file!(), line!())
    };
}

macro_rules! unreachable {
    () => {
        std::unreachable!("[{}:{}]", file!(), line!())
    };
}

macro_rules! unwrap_or_compile_error {
    ($e:expr) => {{
        match $e {
            Ok(x) => x,
            Err(e) => {
                return e.into_compile_error().into();
            }
        }
    }};
}

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as ModuleAttr);
    let mut dependencies = String::new();
    for arg in attr.args {
        if let Some(deps) = arg.dependencies {
            dependencies = deps.value.value().trim().to_string();
        }
    }
    let mut item = syn::parse_macro_input!(item as ModuleItem);
    let mut build = true;
    let mut krnl = parse_quote! { krnl };
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
                } else if let Some(krnl_build) = arg.krnl_build.as_ref() {
                    build = krnl_build.value();
                } else {
                    let ident = arg.ident.as_ref().unwrap();
                    return Error::new_spanned(
                        ident,
                        format!("unknown krnl arg `{ident}`, expected `build` or `crate`"),
                    )
                    .into_compile_error()
                    .into();
                }
            }
        } else {
            item.attr.push(attr);
        }
    }
    let cache = if build {
        let module_name = item.ident.to_string();
        let module_tokens = item.tokens.to_string();
        let module_src = format!("(dependencies({dependencies:?})) => ({module_tokens})");
        let module_src_indices = (0..module_src.len()).into_iter().collect::<Vec<_>>();
        let mut data = HashMap::new();
        data.insert("dependencies", dependencies);
        data.insert("krnl_module_tokens", module_tokens);
        let data = bincode::serialize(&data).unwrap();
        let data_len = data.len();
        let module_msg = format!("module `{module_name}` has been modified, rebuild with `krnlc build`, install krnlc with `cargo install krnlc`");
        quote! {
            const krnlc__krnl_module_data: [u8; #data_len] = [#(#data),*];
            static __module_check: () = {
                let mod_path = module_path!();
                let src = #module_src.as_bytes();
                let mut success = false;
                if let Some(cached_src) = __module(mod_path) {
                    let cached_src = cached_src.as_bytes();
                    if src.len() == cached_src.len() {
                        success = #(src[#module_src_indices] == cached_src[#module_src_indices])&&*;
                    }
                }
                if !success {
                    panic!(#module_msg);
                }
            };
            include!(concat!(env!("CARGO_MANIFEST_DIR"), "/.krnl/packages/", env!("CARGO_PKG_NAME"), "/cache"));
        }
    } else {
        quote! {
            pub(super) const fn __kernel(_: &'static str) -> Option<(u64, &'static [u32], Features)> {
                None
            }
        }
    };
    item.tokens.extend(quote! {
        #[doc(hidden)]
        #[automatically_derived]
        mod module {
            pub(super) use #krnl as __krnl;
            use __krnl::device::Features;
            pub(super) const __BUILD: bool = #build;
            #cache
        }
    });
    item.to_token_stream().into()
}

#[derive(Parse, Debug)]
struct ModuleAttr {
    #[call(Punctuated::parse_terminated)]
    args: Punctuated<ModuleAttrArg, Comma>,
}

#[derive(Parse, Debug)]
struct ModuleAttrArg {
    ident: Ident,
    #[parse_if(ident == "dependencies")]
    dependencies: Option<InsideParen<LitStr>>,
}

#[derive(Parse, Debug)]
struct InsideParen<T> {
    #[paren]
    paren: Paren,
    #[inside(paren)]
    value: T,
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
    #[parse_if(crate_token.is_none())]
    ident: Option<Ident>,
    eq: SynEq,
    #[parse_if(crate_token.is_some())]
    krnl_crate: Option<syn::Path>,
    #[parse_if(ident.as_ref().map_or(false, |x| x == "build"))]
    krnl_build: Option<LitBool>,
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut hasher = SipHasher13::new();
    hasher.write(attr.to_string().as_bytes());
    let attr = parse_macro_input!(attr as KernelAttr);
    let attr_meta = unwrap_or_compile_error!(attr.meta());
    let kernel = parse_macro_input!(item as Kernel);
    hasher.write(kernel.ident.to_string().as_bytes());
    hasher.write(kernel.block.to_token_stream().to_string().as_bytes());
    let hash = hasher.finish();
    unwrap_or_compile_error!(kernel.validate(&attr_meta));
    let unsafe_token = &kernel.unsafe_token;
    let kernel_ident = &kernel.ident;
    let push_const_fields = kernel.push_const_fields();

    let push_consts_ident = kernel.push_consts_ident();
    let push_consts_struct = if let Some(ident) = push_consts_ident.as_ref() {
        quote! {
            #[automatically_derived]
            #[cfg(target_arch = "spirv")]
            #[repr(C)]
            pub struct #ident {
                #(#push_const_fields),*
            }
        }
    } else {
        TokenStream2::new()
    };
    let push_idents: Punctuated<_, Comma> = push_const_fields.iter().map(|x| &x.ident).collect();
    let compute_threads = attr_meta.compute_threads();
    let host_threads = attr_meta.host_threads();
    let mut compute_body = TokenStream2::new();
    let mut for_each_body = TokenStream2::new();
    compute_body.extend(kernel.declare_specs());
    if let Some(ident) = push_consts_ident.as_ref() {
        compute_body.extend(quote! {
            let &#ident {
                #push_idents
            } = push_consts;
        });
    }
    compute_body.extend(unwrap_or_compile_error!(kernel.builtin_tokens(&attr_meta)));
    compute_body.extend(kernel.compute_tokens(attr_meta.for_each));
    if attr_meta.for_each {
        for_each_body.extend(kernel.for_each_tokens());
    }
    let compute_args = kernel.compute_args(attr_meta.for_each);
    let spec_args = kernel.spec_args();
    let device_args = kernel.device_args();
    let device_call_args: Punctuated<_, Comma> = spec_args
        .iter()
        .map(|x| &x.ident)
        .chain(device_args.iter().map(|x| &x.ident))
        .collect();
    let device_args: Punctuated<_, Comma> = spec_args
        .iter()
        .map(|x| {
            quote! {
                #[allow(non_snake_case)] #x
            }
        })
        .chain(device_args.iter().map(ToTokens::to_token_stream))
        .collect();
    let spec_args: Punctuated<_, Comma> = spec_args
        .iter()
        .map(|x| {
            TypedArg::new(
                Ident::new(&x.ident.to_string().to_lowercase(), x.ident.span()),
                x.ty,
            )
        })
        .collect();
    let spec_idents: Vec<_> = spec_args.iter().map(|x| &x.ident).collect();
    let kernel_name_tokens = kernel.kernel_name_tokens();
    let buffer_descs = kernel.buffer_descs();
    let push_consts_size: u32 = {
        let mut size = push_const_fields.iter().map(|x| x.ty.size() as u32).sum();
        if size % 4 != 0 {
            size += 4 - (size % 4);
        }
        size += 2 * buffer_descs.len() as u32 * 4;
        size
    };
    let compute_ident = format_ident!("__krnl_{hash}_{push_consts_size}_{kernel_ident}");
    let device_block = &kernel.block;
    let dispatch_args = kernel.dispatch_args();
    let dispatch_slices = kernel.dispatch_slices();
    let dimensionality = attr_meta.dimensionality;
    let dispatch_builder_dim_impls = if dimensionality == 3 {
        quote! {
            pub fn global_threads(mut self, global_threads: impl Into<UVec3>) -> Result<Self> {
                let raw = self.raw.global_threads(global_threads.into())?;
                Ok(Self {
                    raw,
                })
            }
            pub fn groups(mut self, groups: impl Into<UVec3>) -> Result<Self> {
                let raw = self.raw.groups(groups.into())?;
                Ok(Self {
                    raw,
                })
            }
        }
    } else if dimensionality == 2 {
        quote! {
            pub fn global_threads(mut self, global_threads: impl Into<UVec2>) -> Result<Self> {
                let raw = self.raw.global_threads(global_threads.into())?;
                Ok(Self {
                    raw,
                })
            }
            pub fn groups(mut self, groups: impl Into<UVec2>) -> Result<Self> {
                let raw = self.raw.groups(groups.into())?;
                Ok(Self {
                    raw,
                })
            }
        }
    } else {
        quote! {
            pub fn global_threads(mut self, global_threads: u32) -> Result<Self> {
                let raw = self.raw.global_threads(global_threads.into())?;
                Ok(Self {
                    raw,
                })
            }
            pub fn groups(mut self, groups: u32) -> Result<Self> {
                let raw = self.raw.groups(groups.into())?;
                Ok(Self {
                    raw,
                })
            }
        }
    };
    let device_call = if attr_meta.for_each {
        quote! {
            let mut item_index = global_id;
            while item_index < items {
                #for_each_body
                #kernel_ident(#device_call_args);
                item_index += global_threads;
            }
        }
    } else {
        quote! {
            #kernel_ident(#device_call_args);
        }
    };
    let device_tokens = quote! {
        #push_consts_struct
        #[automatically_derived]
        #[cfg(target_arch = "spirv")]
        #[::krnl_core::spirv_std::spirv(compute(threads(#compute_threads)))]
        pub fn #compute_ident(#compute_args) {
            #compute_body
            fn #kernel_ident(#device_args) #device_block
            #device_call
        }
    };
    //let device_src = prettyplease::unparse(&parse_quote!(#device_tokens));
    //eprintln!("{device_src}");
    let tokens = quote! {
        #device_tokens
        #[automatically_derived]
        #[cfg(not(target_arch = "spirv"))]
        pub mod #kernel_ident {
            use super::module::{
                __BUILD,
                __krnl::{
                    anyhow::Result,
                    scalar::ScalarType,
                    device::{Device, Features},
                    buffer::{Slice, SliceMut},
                    kernel::__private::{
                        Kernel as RawKernel,
                        BufferDesc,
                        builder::{
                            KernelBuilder as RawKernelBuilder,
                            DispatchBuilder as RawDispatchBuilder,
                        },
                    },
                    krnl_core::{
                        half::{f16, bf16},
                        glam::{UVec2, UVec3},
                    },
                    __private::bytemuck::{self, NoUninit},
                },
                __kernel,
            };
            use std::sync::Arc;

            const KERNEL: Result<(&'static [u32], Features), &'static str> = {
                if __BUILD {
                    let kernel = if let Some((hash, spirv, features)) = __kernel(module_path!()) {
                        if hash == #hash {
                            Some((spirv, features))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(kernel) = kernel {
                        Ok(kernel)
                    } else {
                        Err(concat!("Kernel `", module_path!(), "`, module has been modified!"))
                    }
                } else {
                    Err(concat!("Kernel `", module_path!(), "` not built, module has `#[krnl(build=false)]`!"))
                }
            };

            pub mod builder {
                use super::*;

                pub struct KernelBuilder {
                    pub(super) raw: RawKernelBuilder,
                }

                impl KernelBuilder {
                    pub fn build(self, device: Device) -> Result<Kernel> {
                        Ok(Kernel {
                            raw: self.raw.build(device)?
                        })
                    }
                }

                pub struct DispatchBuilder<'a> {
                    pub(super) raw: RawDispatchBuilder<'a>,
                }

                impl DispatchBuilder<'_> {
                    #dispatch_builder_dim_impls
                    pub #unsafe_token fn dispatch(self) -> Result<()> {
                        unsafe {
                            self.raw.dispatch()
                        }
                    }
                }
            }
            use builder::{KernelBuilder, DispatchBuilder};

            #[repr(C)]
            #[derive(Clone, Copy)]
            pub struct PushConsts {
                #(#push_const_fields),*
            }

            unsafe impl NoUninit for PushConsts {}

            pub struct Kernel {
                raw: RawKernel,
            }

            impl Kernel {
                pub fn builder(#spec_args) -> KernelBuilder {
                    let (kernel_name, specs) = (
                        #kernel_name_tokens,
                        &[#(#spec_idents.into()),*],
                    );
                    const BUFFER_DESCS: &[BufferDesc] = &[#buffer_descs];
                    let (spirv, features) = KERNEL.unwrap();
                    let raw = RawKernel::builder(kernel_name, spirv)
                        .features(features)
                        .threads(UVec3::new(#host_threads))
                        .specs(specs)
                        .buffer_descs(BUFFER_DESCS)
                        .push_consts_size(#push_consts_size);
                    KernelBuilder {
                        raw,
                    }
                }
                pub fn dispatch_builder<'a>(&self, #dispatch_args)  -> Result<DispatchBuilder<'a>> {
                    let slices = &mut [#(#dispatch_slices.into(),)*];
                    let push_consts = PushConsts {
                        #push_idents
                    };
                    let push_consts = bytemuck::bytes_of(&push_consts);
                    let raw = self.raw.dispatch_builder(slices, push_consts)?;
                    Ok(DispatchBuilder {
                        raw,
                    })
                }
            }
        }
    };
    tokens.into()
}

#[derive(Parse, Debug)]
struct KernelAttr {
    #[call(Punctuated::parse_terminated)]
    args: Punctuated<KernelAttrArg, Comma>,
}

impl KernelAttr {
    fn meta(&self) -> Result<KernelAttrMeta> {
        let mut meta = KernelAttrMeta {
            threads: [1; 3],
            ..KernelAttrMeta::default()
        };
        let mut has_threads = false;
        for arg in self.args.iter() {
            if let Some(threads) = arg.threads.as_ref() {
                let threads = &threads.value.inner;
                if has_threads {
                    return Err(Error::new_spanned(
                        &arg.ident,
                        "`threads` already specified",
                    ));
                }
                has_threads = true;
                if threads.len() > 3 {
                    return Err(Error::new_spanned(
                        &arg.ident,
                        "expected 1, 2, or 3 dimensional `threads`",
                    ));
                } else {
                    meta.dimensionality = threads.len() as u32;
                };
                for (i, value) in threads.iter().enumerate() {
                    if let Some(spec) = value.spec.as_ref() {
                        meta.thread_specs[i].replace(spec.clone());
                    } else {
                        let constant = value.constant.as_ref().unwrap();
                        let dim: u32 = constant.base10_parse()?;
                        if dim == 0 {
                            return Err(Error::new_spanned(&constant, "thread dim cannot be 0"));
                        }
                        meta.threads[i] = dim;
                    }
                }
            } else if arg.ident == "for_each" {
                meta.for_each = true;
            } else {
                return Err(Error::new_spanned(&arg.ident, "unknown arg"));
            }
        }
        if !has_threads {
            return Err(Error::new(Span2::call_site(), "expected `threads`"));
        }
        if meta.for_each && meta.dimensionality != 1 {
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
    dimensionality: u32,
    threads: [u32; 3],
    thread_specs: [Option<Ident>; 3],
}

impl KernelAttrMeta {
    fn compute_threads(&self) -> Punctuated<LitInt, Comma> {
        if self.thread_specs.iter().any(|x| x.is_some()) {
            std::iter::from_fn(|| Some(LitInt::new("1", Span2::call_site())))
                .take(self.dimensionality as usize)
                .collect()
        } else {
            self.threads
                .iter()
                .map(|x| LitInt::new(&x.to_string(), Span2::call_site()))
                .take(self.dimensionality as usize)
                .collect()
        }
    }
    fn threads_tokens(&self, spec_ids: &HashMap<String, u32>) -> Result<TokenStream2> {
        if !self.thread_specs.iter().any(|x| x.is_some()) {
            let threads = self.compute_threads();
            let tokens = if self.dimensionality == 1 {
                quote! {
                    let threads = #threads;
                }
            } else if self.dimensionality == 2 {
                quote! {
                    let threads = ::krnl_core::glam::UVec2::new(#threads);
                }
            } else if self.dimensionality == 3 {
                quote! {
                    let threads = ::krnl_core::glam::UVec3::new(#threads);
                }
            } else {
                unreachable!();
            };
            return Ok(tokens);
        }
        let mut asm_strings = Vec::new();
        asm_strings.push("%ty_u32 = OpTypeInt 32 0".to_string());
        let mut thread_values = Punctuated::<Expr, Comma>::new();
        for (i, (thread, spec)) in self
            .threads
            .iter()
            .zip(self.thread_specs.iter())
            .enumerate()
        {
            if let Some(spec) = spec.as_ref() {
                if let Some(id) = spec_ids.get(&spec.to_string()).copied() {
                    asm_strings.push(format!("%t{i} = OpSpecConstant %ty_u32 {thread}"));
                    asm_strings.push(format!("OpDecorate SpecId %t{i} {id}"));
                    thread_values.push(parse_quote!(#spec));
                } else {
                    return Err(Error::new_spanned(
                        &spec,
                        "expected spec const identifier or literal",
                    ));
                }
            } else {
                asm_strings.push(format!("%t{i} = OpConstant %ty_u32 {thread}"));
                thread_values.push(parse_quote!(#thread));
            }
        }
        let threads_tokens = if self.dimensionality == 1 {
            quote! {
                let threads = #thread_values;
            }
        } else if self.dimensionality == 2 {
            quote! {
                let threads = ::krnl_core::glam::UVec2::new(#thread_values);
            }
        } else if self.dimensionality == 3 {
            quote! {
                let threads = ::krnl_core::glam::UVec3::new(#thread_values);
            }
        } else {
            unreachable!();
        };
        Ok(quote! {
            {
                ::core::arch::asm! {
                    #(#asm_strings),*
                    "%uvec = OpVector %ty_u32 3",
                    "%workgroup_size = OpSpecConstantComposite %t0 %t1 %t2",
                    "OpDecorate WorkgroupSize %workgroup_size",
                    options(noreturn),
                };
            }
            #threads_tokens
        })
    }
    fn host_threads(&self) -> Punctuated<Expr, Comma> {
        let mut host_threads = Punctuated::new();
        for (thread, spec) in self.threads.iter().zip(self.thread_specs.iter()) {
            if let Some(spec) = spec.as_ref() {
                let ident = format_ident!("{}", spec.to_string().to_lowercase());
                host_threads.push(parse_quote! {
                    #ident
                });
            } else {
                host_threads.push(parse_quote! {
                    #thread
                });
            }
        }
        host_threads
    }
}

#[derive(Parse, Debug)]
struct KernelAttrArg {
    ident: Ident,
    #[parse_if(ident == "threads")]
    threads: Option<InsideParen<Threads>>,
}

#[derive(Parse, Debug)]
struct Threads {
    #[call(Punctuated::parse_separated_nonempty)]
    inner: Punctuated<ThreadValue, Comma>,
}

#[derive(Parse, Debug)]
struct ThreadValue {
    spec: Option<Ident>,
    #[parse_if(spec.is_none())]
    constant: Option<LitInt>,
}

#[derive(Parse, Debug)]
struct Kernel {
    #[call(Attribute::parse_outer)]
    attrs: Vec<Attribute>,
    #[allow(unused)]
    pub_token: Pub,
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

impl Kernel {
    fn compute_buffer_suffix(mut_token: Option<Mut>) -> &'static str {
        if mut_token.is_some() {
            "rw"
        } else {
            "r"
        }
    }
    fn validate(&self, meta: &KernelAttrMeta) -> Result<()> {
        if let Some(generics) = self.generics.as_ref() {
            generics.validate()?;
        }
        for arg in self.args.iter() {
            arg.validate(meta)?;
        }
        Ok(())
    }
    fn kernel_name_tokens(&self) -> TokenStream2 {
        if let Some(generics) = self.generics.as_ref() {
            let mut fmt_string = format!("{}<", self.ident);
            for (i, spec) in generics.specs.iter().enumerate() {
                use std::fmt::Write;
                let ident = &spec.ident;

                if i > 0 {
                    fmt_string.push_str(", ");
                }
                write!(&mut fmt_string, "{ident}={{}}");
            }
            let args: Punctuated<Ident, Comma> = generics
                .specs
                .iter()
                .map(|spec| Ident::new(&spec.ident.to_string().to_lowercase(), spec.ident.span()))
                .collect();
            fmt_string.push('>');
            quote! {
                format!(#fmt_string, #args)
            }
        } else {
            quote! {
                module_path!()
            }
        }
    }
    fn spec_ids(&self) -> HashMap<String, u32> {
        self.generics
            .as_ref()
            .map(KernelGenerics::spec_ids)
            .unwrap_or_default()
    }
    fn spec_args(&self) -> Vec<TypedArg<ScalarType>> {
        self.generics
            .as_ref()
            .map(KernelGenerics::spec_args)
            .unwrap_or_default()
    }
    fn declare_specs(&self) -> TokenStream2 {
        self.generics
            .as_ref()
            .map(KernelGenerics::declare_specs)
            .unwrap_or_default()
    }
    fn push_consts_ident(&self) -> Option<Ident> {
        if self.args.iter().any(|x| x.push.is_some()) {
            Some(format_ident!("__krnl_{}PushConsts", self.ident.to_string()))
        } else {
            None
        }
    }
    fn push_const_fields(&self) -> Vec<TypedArg<ScalarType>> {
        let mut push_consts: Vec<_> = self
            .args
            .iter()
            .filter_map(|arg| {
                arg.push
                    .as_ref()
                    .map(|push| TypedArg::new(arg.ident.clone(), *push))
            })
            .collect();
        push_consts.sort_by_key(|x| -(x.ty.size() as i32));
        push_consts
    }
    fn compute_args(&self, for_each: bool) -> Punctuated<FnArg, Comma> {
        fn buffer_arg(idx: u32, ident: &Ident, mut_token: Option<Mut>, ty: &ScalarType) -> FnArg {
            let binding = LitInt::new(&format!("{idx}"), ident.span());
            let ident = format_ident!("{ident}_{}", Kernel::compute_buffer_suffix(mut_token));
            parse_quote! {
                #[spirv(storage_buffer, descriptor_set = 0, binding = #binding)]
                #ident: &#mut_token [#ty]
            }
        }
        let mut compute_args = Punctuated::<FnArg, Comma>::new();
        let builtins = self.builtins(for_each);
        for builtin in builtins.iter() {
            match builtin.as_str() {
                "global_id" => {
                    compute_args.push(parse_quote! {
                        #[spirv(global_invocation_id)]
                        global_id: ::krnl_core::glam::UVec3
                    });
                }
                "group_id" => {
                    compute_args.push(parse_quote! {
                        #[spirv(workgroup_id)]
                        group_id: ::krnl_core::glam::UVec3
                    });
                }
                "groups" => {
                    compute_args.push(parse_quote! {
                        #[spirv(num_workgroups)]
                        groups: ::krnl_core::glam::UVec3
                    });
                }
                "subgroup_index" => {
                    compute_args.push(parse_quote! {
                        #[spirv(subgroup_id)]
                        subgroup_index: u32
                    });
                }
                "subgroups" => {
                    compute_args.push(parse_quote! {
                        #[spirv(num_subgroups)]
                        subgroups: u32
                    });
                }
                "subgroup_threads" => {
                    compute_args.push(parse_quote! {
                        #[spirv(subgroup_size)]
                        subgroup_threads: u32
                    });
                }
                "subgroup_thread_index" => {
                    compute_args.push(parse_quote! {
                        #[spirv(subgroup_invocation_id)]
                        subgroup_threads: u32
                    });
                }
                "thread_id" => {
                    compute_args.push(parse_quote! {
                        #[spirv(local_invocation_id)]
                        thread_id: ::krnl_core::glam::UVec3
                    });
                }
                "thread_index" => {
                    compute_args.push(parse_quote! {
                        #[spirv(local_invocation_index)]
                        thread_index: u32
                    });
                }
                _ => (),
            }
        }
        let mut buffer_idx = 0;
        for arg in self.args.iter() {
            if let Some(global) = arg.global.as_ref() {
                compute_args.push(buffer_arg(
                    buffer_idx,
                    &arg.ident,
                    global.mut_token,
                    &global.slice.elem,
                ));
                buffer_idx += 1;
            } else if let Some(item) = arg.item.as_ref() {
                compute_args.push(buffer_arg(buffer_idx, &arg.ident, item.mut_token, &item.ty));
                buffer_idx += 1;
            } else if let Some(group) = arg.group.as_ref() {
                let ident = &arg.ident;
                let array = &group.array;
                compute_args.push(parse_quote! {
                    #[spirv(workgroup)]
                    #ident: &mut #array
                });
            } else if let Some(subgroup) = arg.subgroup.as_ref() {
                todo!();
            }
        }
        if let Some(ident) = self.push_consts_ident() {
            compute_args.push(parse_quote! {
                #[spirv(push_constant)]
                push_consts: &#ident
            })
        }
        compute_args
    }
    fn builtins(&self, for_each: bool) -> HashSet<String> {
        let mut builtins = HashSet::new();
        if for_each {
            builtins.extend([
                "global_threads".to_string(),
                "global_id".to_string(),
                "groups".to_string(),
                "threads".to_string(),
            ]);
        }
        for arg in self.args.iter().filter(|x| x.builtin.is_some()) {
            let builtin = arg.ident.to_string();
            match builtin.as_str() {
                "global_threads" => {
                    builtins.insert("groups".to_string());
                }
                "global_index" => {
                    builtins.extend([
                        "group_id".to_string(),
                        "group_index".to_string(),
                        "groups".to_string(),
                        "threads".to_string(),
                        "thread_index".to_string(),
                    ]);
                }
                "group_index" => {
                    builtins.insert("group_id".to_string());
                }
                _ => (),
            }
            builtins.insert(builtin);
        }
        builtins
    }
    fn builtin_tokens(&self, meta: &KernelAttrMeta) -> Result<TokenStream2> {
        let dimensionality = meta.dimensionality;
        let builtins = self.builtins(meta.for_each);
        let mut output = TokenStream2::new();
        if builtins.contains("threads") {
            output.extend(meta.threads_tokens(&self.spec_ids())?);
        }
        let vector_builtins = [
            "global_id",
            "groups",
            "global_threads",
            "group_id",
            "thread_id",
        ];
        for builtin in vector_builtins.iter().filter(|x| builtins.contains(**x)) {
            if *builtin == "global_threads" {
                output.extend(quote! {
                    let global_threads = groups * threads;
                });
            } else if dimensionality == 1 {
                let ident = format_ident!("{builtin}");
                output.extend(quote! {
                    let #ident = #ident.x;
                })
            } else if dimensionality == 2 {
                let ident = format_ident!("{builtin}");
                output.extend(quote! {
                    let #ident = #ident.truncate();
                })
            }
        }
        let scalar_builtins = ["group_index", "global_index"];
        for builtin in scalar_builtins.iter().filter(|x| builtins.contains(**x)) {
            if *builtin == "group_index" {
                if dimensionality == 1 {
                    output.extend(quote! {
                        let group_index = group_id;
                    });
                } else if dimensionality == 2 {
                    output.extend(quote! {
                        let group_index = group_id.x + group_id.y * threads.x;
                    });
                } else if dimensionality == 3 {
                    output.extend(quote! {
                        let group_index = group_id.x + group_id.y * threads.x + group_id.z * threads.x * threads.y;
                    });
                }
            } else if *builtin == "global_index" {
                if dimensionality == 1 {
                    output.extend(quote! {
                        let global_index = group_index * threads + thread_index;
                    });
                } else if dimensionality == 2 {
                    output.extend(quote! {
                        let global_index = group_index * threads.x * threads.y + thread_index;
                    });
                } else if dimensionality == 3 {
                    output.extend(quote! {
                        let global_index = group_index * threads.x * threads.y * threads.z + thread_index;
                    });
                }
            }
        }
        Ok(output)
    }
    fn compute_tokens(&self, for_each: bool) -> TokenStream2 {
        let spec_ids = self.spec_ids();
        let mut tokens = TokenStream2::new();
        let mut has_items = false;
        for arg in self.args.iter() {
            let ident = &arg.ident;
            if let Some(global) = arg.global.as_ref() {
                let compute_ident = format_ident!(
                    "{ident}_{}",
                    Kernel::compute_buffer_suffix(global.mut_token)
                );
                if global.mut_token.is_some() {
                    tokens.extend(quote! {
                        let ref mut #ident = ::krnl_core::mem::UnsafeMut::from_mut(#compute_ident);
                    });
                } else {
                    tokens.extend(quote! {
                        let ref #ident = #compute_ident;
                    });
                }
            } else if let Some(group) = arg.group.as_ref() {
                tokens.extend(quote! {
                    let ref mut #ident = ::krnl_core::mem::UninitUnsafeMut::from_mut(#ident);
                });
            } else if let Some(subgroup) = arg.subgroup.as_ref() {
                tokens.extend(quote! {
                    *#ident = Default::default();
                });
            } else if let Some(item) = arg.item.as_ref() {
                let compute_ident =
                    format_ident!("{ident}_{}", Kernel::compute_buffer_suffix(item.mut_token));
                tokens.extend(quote! {
                    let #ident = #compute_ident;
                });
                if !has_items {
                    tokens.extend(quote! {
                        let items = #ident.len() as u32;
                    });
                    has_items = true;
                }
            }
        }
        if for_each && !has_items {
            tokens.extend(quote! {
                let items = 0u32;
            });
        }
        tokens
    }
    fn for_each_tokens(&self) -> TokenStream2 {
        let mut tokens = TokenStream2::new();
        for arg in self.args.iter() {
            if let Some(item) = arg.item.as_ref() {
                let ident = &arg.ident;
                if item.mut_token.is_some() {
                    tokens.extend(quote! {
                        let #ident = unsafe {
                            ::krnl_core::spirv_std::arch::IndexUnchecked::index_unchecked_mut(#ident, item_index as usize)
                        };
                    });
                } else {
                    tokens.extend(quote! {
                        let #ident = unsafe {
                            ::krnl_core::spirv_std::arch::IndexUnchecked::index_unchecked(#ident, item_index as usize)
                        };
                    });
                }
            }
        }
        tokens
    }
    fn device_args(&self) -> Punctuated<TypedArg<Type>, Comma> {
        self.args.iter().map(KernelArg::device_arg).collect()
    }
    fn dispatch_args(&self) -> Punctuated<TypedArg<Type>, Comma> {
        self.args
            .iter()
            .filter_map(KernelArg::dispatch_arg)
            .collect()
    }
    fn buffer_descs(&self) -> Punctuated<TokenStream2, Comma> {
        self.args
            .iter()
            .filter_map(KernelArg::buffer_desc)
            .collect()
    }
    fn dispatch_slices(&self) -> Vec<Ident> {
        self.args
            .iter()
            .filter_map(KernelArg::dispatch_slice)
            .collect()
    }
}

#[derive(Parse, Debug)]
struct KernelGenerics {
    lt: Lt,
    #[call(Punctuated::parse_separated_nonempty)]
    specs: Punctuated<KernelSpec, Comma>,
    gt: Gt,
}

impl KernelGenerics {
    fn validate(&self) -> Result<()> {
        for spec in self.specs.iter() {
            if spec.spec.value != "spec" {
                return Err(Error::new_spanned(&spec.spec.value, "expected spec"));
            }
            if spec.ident.to_string().starts_with("__krnl_") {
                return Err(Error::new_spanned(&spec.ident, "`__krnl_` is reserved"));
            }
        }
        Ok(())
    }
    fn spec_ids(&self) -> HashMap<String, u32> {
        self.specs
            .iter()
            .enumerate()
            .map(|(i, x)| (x.ident.to_string(), i as u32))
            .collect()
    }
    fn spec_args(&self) -> Vec<TypedArg<ScalarType>> {
        self.specs
            .iter()
            .map(|x| {
                let ident = Ident::new(&x.ident.to_string(), x.ident.span());
                TypedArg::new(ident, x.ty)
            })
            .collect()
    }
    fn declare_specs(&self) -> TokenStream2 {
        let mut tokens = TokenStream2::new();
        for (i, spec) in self.specs.iter().enumerate() {
            spec.declare(&mut tokens, i as u32);
        }
        tokens
    }
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
    ty: ScalarType,
}

impl KernelSpec {
    fn declare(&self, tokens: &mut TokenStream2, spec_id: u32) {
        let bits = self.ty.size() * 8;
        let signed = if self.ty.name.starts_with('i') { 1 } else { 0 };
        let float = matches!(self.ty.name, "f32" | "f64");
        let ty_string = if float {
            format!("%ty = OpTypeFloat {bits}")
        } else {
            format!("%ty = OpTypeInt {bits} {signed}")
        };
        let values = if bits == 64 { "0 0" } else { "0" };
        let spec_string = format!("%spec = OpSpecConstant %ty {values}");
        let spec_id_string = format!("OpDecorate %spec SpecId {spec_id}");
        let ident = &self.ident;
        tokens.extend(quote! {
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
        });
    }
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
struct KernelArg {
    #[allow(unused)]
    pound: Pound,
    attr: InsideBracket<Ident>,
    ident: Ident,
    #[allow(unused)]
    colon: Colon,
    #[parse_if(attr.value == "builtin")]
    builtin: Option<Ident>,
    #[parse_if(attr.value == "global")]
    global: Option<KernelGlobal>,
    #[parse_if(attr.value == "group")]
    group: Option<KernelGroup>,
    #[parse_if(attr.value == "subgroup")]
    subgroup: Option<KernelSubgroup>,
    #[parse_if(attr.value == "item")]
    item: Option<KernelItem>,
    #[parse_if(attr.value == "push")]
    push: Option<ScalarType>,
}

impl KernelArg {
    fn validate(&self, meta: &KernelAttrMeta) -> Result<()> {
        if self.ident.to_string().starts_with("__krnl_") {
            return Err(Error::new_spanned(&self.ident, "`__krnl_` is reserved"));
        }
        let attrs = ["builtin", "global", "group", "subgroup", "item", "push"];
        if !attrs.iter().copied().any(|x| self.attr.value == *x) {
            return Err(Error::new_spanned(&self.attr.value, "unknown attribute"));
        }
        if let Some(builtin_ty) = self.builtin.as_ref() {
            let for_each_builtins = &["item_index", "items"];
            let builtins_u32 = &[
                "global_index",
                "group_index",
                "subgroup_index",
                "thread_index",
            ];
            let builtins_vector = &[
                "global_id",
                "global_threads",
                "group_id",
                "groups",
                "subgroup_id",
                "subgroups",
                "thread_id",
                "threads",
            ];
            let mut builtins = for_each_builtins
                .into_iter()
                .chain(builtins_u32)
                .chain(builtins_vector)
                .copied();
            if let Some(builtin) = builtins.find(|x| self.ident == *x) {
                let is_for_each = for_each_builtins.into_iter().any(|x| builtin == *x);
                if is_for_each {
                    return Err(Error::new_spanned(
                        &self.ident,
                        "requires `for_each` kernel",
                    ));
                } else if meta.for_each {
                    if !is_for_each {
                        return Err(Error::new_spanned(
                            &self.ident,
                            "not allowed in `for_each` kernel",
                        ));
                    } else {
                        if builtin_ty != "u32" {
                            return Err(Error::new_spanned(builtin_ty, "expected u32"));
                        }
                    }
                } else if builtins_u32.into_iter().any(|x| builtin == *x) {
                    if builtin_ty != "u32" {
                        return Err(Error::new_spanned(builtin_ty, "expected u32"));
                    }
                } else if builtins_vector.into_iter().any(|x| builtin == *x) {
                    let dimensionality = meta.dimensionality;
                    if dimensionality == 1 && builtin_ty != "u32" {
                        return Err(Error::new_spanned(builtin_ty, "expected u32"));
                    } else if dimensionality == 2 && builtin_ty != "UVec2" {
                        return Err(Error::new_spanned(builtin_ty, "expected UVec2"));
                    } else if dimensionality == 3 && builtin_ty != "UVec3" {
                        return Err(Error::new_spanned(builtin_ty, "expected UVec3"));
                    }
                } else {
                    unreachable!();
                }
            } else {
                return Err(Error::new_spanned(&self.ident, "unknown builtin"));
            }
        } else if let Some(global) = self.global.as_ref() {
            if let Some(unsafe_mut) = global.unsafe_mut.as_ref() {
                if meta.for_each {
                    return Err(Error::new(
                        unsafe_mut.span,
                        "not allowed in `for_each` kernel",
                    ));
                }
            }
        } else if let Some(group) = self.group.as_ref() {
            if meta.for_each {
                return Err(Error::new(
                    group.uninit_unsafe_mut.span,
                    "not allowed in `for_each` kernel",
                ));
            }
        } else if let Some(item) = self.item.as_ref() {
            if !meta.for_each {
                return Err(Error::new_spanned(
                    &self.attr.value,
                    "requires `for_each` kernel",
                ));
            }
        }
        Ok(())
    }
    fn device_arg(&self) -> TypedArg<Type> {
        let ty = if let Some(builtin) = self.builtin.as_ref() {
            parse_quote!(#builtin)
        } else if let Some(global) = self.global.as_ref() {
            parse_quote!(#global)
        } else if let Some(group) = self.group.as_ref() {
            parse_quote!(#group)
        } else if let Some(subgroup) = self.subgroup.as_ref() {
            parse_quote!(#subgroup)
        } else if let Some(item) = self.item.as_ref() {
            parse_quote!(#item)
        } else if let Some(push) = self.push.as_ref() {
            parse_quote!(#push)
        } else {
            unreachable!()
        };
        TypedArg::new(self.ident.clone(), ty)
    }
    fn dispatch_arg(&self) -> Option<TypedArg<Type>> {
        let ty = if let Some(global) = self.global.as_ref() {
            global.dispatch_type()
        } else if let Some(item) = self.item.as_ref() {
            item.dispatch_type()
        } else if let Some(push) = self.push.as_ref() {
            parse_quote!(#push)
        } else {
            return None;
        };
        Some(TypedArg::new(self.ident.clone(), ty))
    }
    fn buffer_desc(&self) -> Option<TokenStream2> {
        if let Some(global) = self.global.as_ref() {
            let name = self.ident.to_string();
            let scalar_type = &global.slice.elem;
            let scalar_type = Ident::new(&scalar_type.name.to_uppercase(), scalar_type.span);
            let mutable = global.mut_token.is_some();
            Some(quote! {
                BufferDesc::new(#name, ScalarType::#scalar_type)
                    .with_mutable(#mutable)
            })
        } else if let Some(item) = self.item.as_ref() {
            let name = self.ident.to_string();
            let scalar_type = &item.ty;
            let scalar_type = Ident::new(&scalar_type.name.to_uppercase(), scalar_type.span);
            let mutable = item.mut_token.is_some();
            Some(quote! {
                BufferDesc::new(#name, ScalarType::#scalar_type)
                    .with_mutable(#mutable)
                    .with_item(true)
            })
        } else {
            None
        }
    }
    fn dispatch_slice(&self) -> Option<Ident> {
        if self.global.is_some() || self.item.is_some() {
            Some(self.ident.clone())
        } else {
            None
        }
    }
}

#[derive(Parse, Debug)]
struct KernelSlice<E> {
    #[bracket]
    bracket: Bracket,
    #[inside(bracket)]
    elem: E,
}

impl<E: ToTokens> ToTokens for KernelSlice<E> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.bracket
            .surround(tokens, |tokens| self.elem.to_tokens(tokens));
    }
}

#[derive(Parse, Debug)]
struct KernelGlobal {
    #[allow(unused)]
    and_token: And,
    mut_token: Option<Mut>,
    #[parse_if(mut_token.is_some())]
    unsafe_mut: Option<KernelUnsafeMut>,
    #[parse_if(unsafe_mut.is_some())]
    lt: Option<Lt>,
    slice: KernelSlice<ScalarType>,
    #[parse_if(lt.is_some())]
    gt: Option<Gt>,
}

impl KernelGlobal {
    fn dispatch_type(&self) -> Type {
        let elem = &self.slice.elem;
        if self.mut_token.is_some() {
            parse_quote! {
                SliceMut<'a, #elem>
            }
        } else {
            parse_quote! {
                Slice<'a, #elem>
            }
        }
    }
}

impl ToTokens for KernelGlobal {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.and_token.to_tokens(tokens);
        if let Some((mut_token, unsafe_mut)) = self.mut_token.as_ref().zip(self.unsafe_mut.as_ref())
        {
            mut_token.to_tokens(tokens);
            unsafe_mut.to_tokens(tokens);
        }
        if let Some(lt) = self.lt.as_ref() {
            lt.to_tokens(tokens);
        }
        self.slice.to_tokens(tokens);
        if let Some(gt) = self.gt.as_ref() {
            gt.to_tokens(tokens);
        }
    }
}

#[derive(Parse, Debug)]
struct KernelArray<E> {
    #[bracket]
    bracket: Bracket,
    #[inside(bracket)]
    elem: E,
    #[inside(bracket)]
    semi: Semi,
    len: Expr,
}

impl<E: ToTokens> ToTokens for KernelArray<E> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.bracket.surround(tokens, |tokens| {
            self.elem.to_tokens(tokens);
            self.semi.to_tokens(tokens);
            self.len.to_tokens(tokens);
        });
    }
}

#[derive(Parse, Debug)]
struct KernelGroup {
    #[allow(unused)]
    and: And,
    #[allow(unused)]
    mut_token: Mut,
    uninit_unsafe_mut: KernelUninitUnsafeMut,
    lt: Lt,
    array: KernelArray<ScalarType>,
    gt: Gt,
}

impl ToTokens for KernelGroup {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.and.to_tokens(tokens);
        self.mut_token.to_tokens(tokens);
        self.uninit_unsafe_mut.to_tokens(tokens);
        self.lt.to_tokens(tokens);
        self.array.to_tokens(tokens);
        self.gt.to_tokens(tokens);
    }
}

#[derive(Parse, Debug)]
struct KernelSubgroup {
    #[allow(unused)]
    and: And,
    #[allow(unused)]
    mut_token: Mut,
    array: KernelArray<ScalarType>,
}

impl ToTokens for KernelSubgroup {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.and.to_tokens(tokens);
        self.mut_token.to_tokens(tokens);
        self.array.to_tokens(tokens);
    }
}

#[derive(Parse, Debug)]
struct KernelItem {
    #[allow(unused)]
    and: And,
    #[allow(unused)]
    mut_token: Option<Mut>,
    ty: ScalarType,
}

impl KernelItem {
    fn dispatch_type(&self) -> Type {
        let elem = &self.ty;
        if self.mut_token.is_some() {
            parse_quote! {
                SliceMut<'a, #elem>
            }
        } else {
            parse_quote! {
                Slice<'a, #elem>
            }
        }
    }
}

impl ToTokens for KernelItem {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.and.to_tokens(tokens);
        if let Some(mut_token) = self.mut_token.as_ref() {
            mut_token.to_tokens(tokens);
        }
        self.ty.to_tokens(tokens);
    }
}

#[derive(Parse, Debug)]
struct TypedArg<T> {
    ident: Ident,
    colon: Colon,
    ty: T,
}

impl<T> TypedArg<T> {
    fn new(ident: Ident, ty: T) -> Self {
        Self {
            ident,
            colon: Colon::default(),
            ty,
        }
    }
}

impl<T: ToTokens> ToTokens for TypedArg<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
    }
}

#[derive(Debug)]
struct KernelUnsafeMut {
    span: Span2,
}

impl Parse for KernelUnsafeMut {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ident: Ident = input.parse()?;
        if ident == "UnsafeMut" {
            Ok(Self { span: ident.span() })
        } else {
            Err(Error::new_spanned(&ident, "expected `UnsafeMut`"))
        }
    }
}

impl ToTokens for KernelUnsafeMut {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        Ident::new("UnsafeMut", self.span).to_tokens(tokens);
    }
}

#[derive(Debug)]
struct KernelUninitUnsafeMut {
    span: Span2,
}

impl Parse for KernelUninitUnsafeMut {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ident: Ident = input.parse()?;
        if ident == "UninitUnsafeMut" {
            Ok(Self { span: ident.span() })
        } else {
            Err(Error::new_spanned(&ident, "expected `UninitUnsafeMut`"))
        }
    }
}

impl ToTokens for KernelUninitUnsafeMut {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        Ident::new("UninitUnsafeMut", self.span).to_tokens(tokens);
    }
}

#[derive(Debug, Clone, Copy)]
struct ScalarType {
    name: &'static str,
    span: Span2,
}

impl ScalarType {
    fn from_ident(ident: &Ident) -> Result<Self> {
        let span = ident.span();
        let scalars = [
            "u8", "i8", "u16", "i16", "f16", "bf16", "u32", "i32", "f32", "u64", "i64", "f64",
        ];
        if let Some(name) = scalars.iter().find(|x| ident == **x) {
            Ok(Self { name, span })
        } else {
            Err(Error::new(span, "expected scalar"))
        }
    }
    fn size(&self) -> usize {
        match self.name {
            "u8" | "i8" => 1,
            "u16" | "i16" | "f16" | "bf16" => 2,
            "u32" | "i32" | "f32" => 4,
            "u64" | "i64" | "f64" => 8,
            _ => unreachable!(),
        }
    }
}

impl Parse for ScalarType {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        Self::from_ident(&input.parse()?)
    }
}

impl ToTokens for ScalarType {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        Ident::new(&self.name, self.span).to_tokens(tokens);
    }
}
