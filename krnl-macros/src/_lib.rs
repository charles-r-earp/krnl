#![cfg_attr(krnl_build, feature(proc_macro_span))]
#![allow(warnings)]
use derive_syn_parse::Parse;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use proc_macro::{Span, TokenStream};
use proc_macro2::{Span as Span2, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
use serde::{Deserialize, Serialize};
use spirv::Capability;
use std::{
    collections::{HashMap, HashSet},
    env::{var, vars},
    fs::{self, File},
    hash::Hash,
    path::PathBuf,
    process::Command,
    str::FromStr,
    sync::Arc,
};
use syn::{
    parse::{Parse, ParseStream},
    parse_quote,
    punctuated::Punctuated,
    token::{
        And, Brace, Bracket, Colon, Colon2, Comma, Eq as SynEq, Fn, Gt, Lt, Mod, Mut, Paren, Pound,
        Pub, Unsafe,
    },
    Attribute, Block, Error, FnArg, Ident, ItemMod, LitBool, LitInt, LitStr, Stmt, Type,
    Visibility,
};

mod kernel_info;
use kernel_info::*;

type Result<T, E = Error> = std::result::Result<T, E>;

trait IntoSynResult {
    type Output;
    fn into_syn_result(self, span: Span2) -> Result<Self::Output>;
}

impl<T, E: std::fmt::Display> IntoSynResult for Result<T, E> {
    type Output = T;
    fn into_syn_result(self, span: Span2) -> Result<T> {
        self.or_else(|e| Err(Error::new(span, e)))
    }
}

macro_rules! unwrap {
    ($x:expr) => {{
        let x = $x;
        let mut is_err = true;
        x.as_ref().map(|_| {
            is_err = false;
        });
        if is_err {
            let s = stringify!($x);
            dbg!(s);
        }
        x.unwrap()
    }};
}

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    match module_impl(attr, item) {
        Ok(x) => x,
        Err(e) => e.into_compile_error().into(),
    }
}

#[derive(Parse, Debug)]
struct ModuleAttributes {
    #[call(Punctuated::parse_terminated)]
    attr: Punctuated<ModuleAttribute, Comma>,
}

#[derive(Parse, Debug)]
struct ModuleAttribute {
    ident: Ident,
    #[parse_if(ident == "dependency")]
    dependency: Option<Dependency>,
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
    eq: SynEq,
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

#[derive(Eq, PartialEq, Hash, Serialize, Deserialize, Debug)]
struct ModuleInfo {
    name: String,
    binary_name: Option<String>,
    dependencies: String,
    source: String,
}

#[derive(Debug)]
struct ModuleCache {
    build: bool,
    manifest_dir: PathBuf,
    krnl_dir: PathBuf,
    cache_path: PathBuf,
    build_dir: PathBuf,
    builder_dir: PathBuf,
    module_names: HashMap<String, Vec<String>>,
    module_spirvs: HashMap<ModuleInfo, HashMap<String, Spirv>>,
}

impl ModuleCache {
    fn load() -> MutexGuard<'static, Self> {
        static CACHE: Lazy<Mutex<ModuleCache>> = Lazy::new(|| {
            let primary_package = var("CARGO_PRIMARY_PACKAGE").as_deref() == Ok("1");
            let build = cfg!(krnl_build) && primary_package;
            let manifest_dir = PathBuf::from(unwrap!(var("CARGO_MANIFEST_DIR")));
            let krnl_dir = manifest_dir.join(".krnl");
            let cache_path = krnl_dir.join("cache.bincode");
            let builder_dir = krnl_dir.join("builder");
            let build_dir = krnl_dir.join("build");
            let cashdir_tag_path = krnl_dir.join("CASHDIR.TAG");
            let cashdir_tag = concat!(
                "Signature: 8a477f597d28d172789f06886806bc55",
                "\n# This file is a cache directory tag created by krnl.",
                "\n# For information about cache directory tags see https://bford.info/cachedir/"
            );
            let module_names = HashMap::new();
            let mut module_spirvs: HashMap<ModuleInfo, HashMap<String, Spirv>> = HashMap::new();
            if build {
                if !krnl_dir.exists() {
                    unwrap!(fs::create_dir(&krnl_dir));
                }
                if !cashdir_tag_path.exists() {
                    unwrap!(fs::write(&cashdir_tag_path, cashdir_tag.as_bytes()));
                } else {
                    let tag = unwrap!(fs::read_to_string(&cashdir_tag_path));
                    assert_eq!(tag, cashdir_tag);
                }
                unwrap!(fs::write(
                    krnl_dir.join(".gitignore"),
                    "# Generated by krnl\nbuilder\nbuild".as_bytes()
                ));
                if !builder_dir.exists() {
                    unwrap!(fs::create_dir(&builder_dir));
                }
                if !build_dir.exists() {
                    unwrap!(fs::create_dir(&build_dir));
                }
                let manifest =
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/builder/Cargo.toml"));
                unwrap!(fs::write(
                    builder_dir.join("Cargo.toml"),
                    manifest.as_bytes()
                ));
                let toolchain = include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/builder/rust-toolchain.toml"
                ));
                unwrap!(fs::write(
                    builder_dir.join("rust-toolchain.toml"),
                    toolchain.as_bytes()
                ));
                let builder_src_dir = builder_dir.join("src");
                if !builder_src_dir.exists() {
                    unwrap!(fs::create_dir(&builder_src_dir));
                }
                let kernel_info =
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernel_info.rs"));
                unwrap!(fs::write(
                    builder_src_dir.join("kernel_info.rs"),
                    kernel_info.as_bytes()
                ));
                let main =
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/builder/src/main.rs"));
                unwrap!(fs::write(builder_src_dir.join("main.rs"), main.as_bytes()));
            } else if cache_path.exists() {
                let bytes = unwrap!(fs::read(&cache_path));
                module_spirvs = unwrap!(bincode::deserialize(&bytes));
            }
            Mutex::new(ModuleCache {
                build,
                manifest_dir,
                krnl_dir,
                cache_path,
                build_dir,
                builder_dir,
                module_names,
                module_spirvs,
            })
        });
        CACHE.lock()
    }
    fn get(&self, info: &ModuleInfo) -> Option<&HashMap<String, Spirv>> {
        self.module_spirvs.get(info)
    }
    fn insert(
        &mut self,
        info: ModuleInfo,
        spirvs: HashMap<String, Spirv>,
    ) -> &HashMap<String, Spirv> {
        self.module_spirvs.entry(info).or_insert(spirvs)
    }
    fn save(&self) {
        let bytes = unwrap!(bincode::serialize(&self.module_spirvs));
        unwrap!(std::fs::write(&self.cache_path, &bytes));
    }
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match kernel_impl(attr, item) {
        Ok(x) => x,
        Err(e) => e.into_compile_error().into(),
    }
}

struct KernelInfoMap {
    path: PathBuf,
    map: HashMap<CompileOptions, Vec<KernelInfo>>,
}

impl KernelInfoMap {
    fn lock() -> MutexGuard<'static, Self> {
        static MAP: Lazy<Mutex<KernelInfoMap>> = Lazy::new(|| {
            let path =
                PathBuf::from(unwrap!(var("CARGO_MANIFEST_DIR"))).join("kernel_infos.bincode");
            Mutex::new(KernelInfoMap {
                path,
                map: HashMap::new(),
            })
        });
        MAP.lock()
    }
    fn insert(&mut self, info: KernelInfo) {
        self.map
            .entry(info.compile_options())
            .or_default()
            .push(info);
    }
    fn save(&self) {
        let bytes = unwrap!(bincode::serialize(&self.map));
        unwrap!(std::fs::write(&self.path, &bytes));
    }
}

fn compile_options_are_enabled(compile_options: &CompileOptions) -> bool {
    if let Ok(target) = var("TARGET") {
        let (major, minor) = compile_options.vulkan;
        let target_prefix = "spirv-unknown-vulkan";
        let vulkan = target.split_at(target_prefix.len()).1;
        if vulkan != &format!("{major}.{minor}") {
            return false;
        }
        let target_feature_prefix = "CARGO_CFG_TARGET_FEATURE_";
        let mut capability_enabled = vec![false; compile_options.capabilities.len()];
        for var in vars().filter_map(|(k, v)| if &v == "" { Some(k) } else { None }) {
            let var = var.split_at(target_feature_prefix.len()).1;
            if let Ok(cap) = Capability::from_str(&var) {
                if let Some(index) = compile_options.capabilities.iter().position(|x| *x == cap) {
                    *unwrap!(capability_enabled.get_mut(index)) = true;
                }
            }
        }
        if capability_enabled.iter().any(|x| !x) {
            return false;
        }
        if !compile_options.extensions.is_empty() {
            todo!()
        }
        true
    } else {
        false
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
    major: LitInt,
    #[inside(paren)]
    #[allow(unused)]
    comma: Comma,
    #[inside(paren)]
    minor: LitInt,
}

impl Vulkan {
    fn version(&self) -> Result<(u32, u32)> {
        let major = self.major.base10_parse()?;
        let minor = self.minor.base10_parse()?;
        Ok((major, minor))
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
    #[call(Attribute::parse_outer)]
    attrs: Vec<Attribute>,
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
    elem_ty: Ident,
    elementwise: bool,
}

impl Parse for KernelSliceArg {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        if input.peek(Ident) {
            let ident1: Ident = input.parse()?;
            if input.peek(Lt) {
                input.parse::<Lt>()?;
                let content;
                let _ = syn::bracketed!(content in input);
                let elem_ty = content.parse()?;
                let _ = input.parse::<Gt>()?;
                Ok(Self {
                    wrapper_ty: Some(ident1),
                    elem_ty,
                    elementwise: false,
                })
            } else {
                Ok(Self {
                    wrapper_ty: None,
                    elem_ty: ident1,
                    elementwise: true,
                })
            }
        } else {
            let content;
            let _ = syn::bracketed!(content in input);
            let elem_ty = content.parse()?;
            Ok(Self {
                wrapper_ty: None,
                elem_ty,
                elementwise: false,
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
    ("global_threads", "UVec*"),
    ("global_id", "UVec*"),
    ("groups", "UVec*"),
    ("group_id", "UVec*"),
    ("subgroup_id", "UVec*"),
    ("subgroups", "UVec*"),
    ("threads", "UVec*"),
    ("thread_id", "UVec*"),
    ("thread_index", "u32"),
    ("elements", "u32"),
    ("element_index", "u32"),
];

fn kernel_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let span = Span2::call_site();
    let kernel_attr: KernelAttributes = syn::parse(attr)?;
    let kernel: Kernel = syn::parse(item)?;
    let mut info = KernelInfo {
        path: unwrap!(var("CARGO_CRATE_NAME")),
        name: kernel.ident.to_string(),
        vulkan: (1, 1),
        capabilities: Vec::new(),
        extensions: Vec::new(),
        safe: kernel.unsafe_token.is_none(),
        threads: [1, 1, 1],
        dimensionality: 0,
        elementwise: false,
        args: Vec::new(),
    };
    let mut has_vulkan = false;
    let mut thread_lits = Punctuated::<LitInt, Comma>::new();
    for attr in kernel_attr.attr.iter() {
        let attr_string = attr.ident.to_string();
        if let Some(vulkan) = attr.vulkan.as_ref() {
            info.vulkan = vulkan.version()?;
            has_vulkan = true;
        } else if attr_string == "elementwise" {
            info.elementwise = true;
        } else if let Some(threads) = attr.threads.as_ref() {
            if info.dimensionality != 0 {
                return Err(Error::new_spanned(&thread_lits, "threads already declared"));
            }
            if threads.threads.is_empty() || threads.threads.len() > 3 {
                return Err(Error::new_spanned(
                    &thread_lits,
                    "expected 1, 2, or 3 dimensional threads, ie `#[kernel(threads(16, 16))]`",
                ));
            }
            info.dimensionality = threads.threads.len() as u32;
            for (i, x) in threads.threads.iter().enumerate() {
                info.threads[i] = x.base10_parse()?;
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
            return Err(Error::new_spanned(&attr.ident, "unknown attribute, expected `vulkan`, `elementwise`, `threads`, `capabilities`, or `extensions`"));
        }
    }
    if !has_vulkan {
        return Err(Error::new(
            span,
            "vulkan (major, minor) must be specified, ie `#[kernel(vulkan(1, 1))]`",
        ));
    }
    if info.dimensionality == 0 {
        return Err(Error::new(
            span,
            "expected 1, 2, or 3 dimensional threads, ie `#[kernel(threads(16, 16))]`",
        ));
    }
    if info.elementwise && info.dimensionality != 1 {
        return Err(Error::new_spanned(
            &thread_lits,
            "can only use 1 dimensional threads in elementwise kernel",
        ));
    }
    let mut builtins = HashSet::<&'static str>::new();
    let mut outer_args = Punctuated::<TypedArg, Comma>::new();
    let mut inner_args = Punctuated::<TypedArg, Comma>::new();
    let mut group_args = Vec::<(Ident, KernelGroupArg)>::new();
    let mut push_consts = Vec::<(ScalarType, usize, TypedArg)>::new();
    let mut set_lit = LitInt::new("0", span);
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
                    let (builtin, mut builtin_ty) = BUILTINS
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
                        .ok_or_else(|| Error::new_spanned(&arg_ident, "expected a builtin type"))?;
                    let (outer_ty, inner_ty) = if builtin_ty == "UVec*" {
                        let inner_ty = match info.dimensionality {
                            1 => "u32",
                            2 => "UVec2",
                            3 => "UVec3",
                            _ => unreachable!(),
                        };
                        (format_ident!("UVec3"), format_ident!("{inner_ty}"))
                    } else {
                        let ident = format_ident!("{builtin_ty}");
                        (ident.clone(), ident)
                    };
                    if arg_ty != &inner_ty {
                        return Err(Error::new_spanned(
                            &arg_ty,
                            format!("expected `{inner_ty}`"),
                        ));
                    }
                    match ident_string.as_str() {
                        "global_threads" => {
                            if !builtins.contains("groups") {
                                outer_args.push(parse_quote! {
                                    #[spirv(num_workgroups)]
                                    groups: krnl_core::glam::UVec3
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
                            let outer_ty = if outer_ty == "UVec3" {
                                quote! {
                                    krnl_core::glam::#outer_ty
                                }
                            } else {
                                quote! {
                                    #outer_ty
                                }
                            };
                            let spirv_attr = Ident::new(spirv_attr, arg.ident.span());
                            outer_args.push(parse_quote! {
                                #[spirv(#spirv_attr)]
                                #arg_ident: #outer_ty
                            });
                            builtins.insert(builtin);
                        }
                    }
                    inner_args.push(parse_quote! {
                        #arg_ident: #inner_ty
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
                let scalar_type = ScalarType::from_str(push_ty_string.as_str())
                    .ok_or_else(|| Error::new_spanned(&push_ty, "expected a scalar"))?;
                let typed_arg: TypedArg = parse_quote! {
                    #arg_ident: #push_ty
                };
                let index = info.args.len();
                push_consts.push((scalar_type, index, typed_arg.clone()));
                inner_args.push(typed_arg);
                info.args.push(Arg::Push(PushInfo {
                    name: ident_string,
                    scalar_type,
                    offset: 0,
                }))
            }
        } else if let Some(slice_arg) = arg.slice_arg.as_ref() {
            let elementwise = slice_arg.elementwise;
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
                    if mutable {
                        inner_args.push(parse_quote! {
                            #arg_ident: & #mut_token #wrapper_ty <[#elem_ty]>
                        })
                    } else {
                        return Err(Error::new_spanned(
                            arg.and.as_ref().unwrap(),
                            "expected `&mut _`",
                        ));
                    }
                } else {
                    return Err(Error::new_spanned(wrapper_ty, "expected `GlobalMut<_>`"));
                }
            } else if elementwise {
                inner_args.push(parse_quote! {
                    #arg_ident: & #mut_token #elem_ty
                })
            } else if mutable {
                return Err(Error::new_spanned(mut_token, "`&mut [_]` can not be used as a kernel argument directly, use `krnl_core::mem::GlobalMut<_>` instead"));
            } else {
                inner_args.push(parse_quote! {
                    #arg_ident: & #mut_token [#elem_ty]
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
            let scalar_type = ScalarType::from_str(elem_ty_string.as_str())
                .ok_or_else(|| Error::new_spanned(&elem_ty, "expected a scalar"))?;
            info.args.push(Arg::Slice(SliceInfo {
                name: arg_ident.to_string(),
                scalar_type,
                mutable,
                elementwise,
            }));
            if !elementwise {
                let index = info.args.len();
                let name = format!("__krnl_len_{}", arg_ident.to_string());
                let ident = format_ident!("{}", name);
                info.args.push(Arg::Push(PushInfo {
                    name,
                    scalar_type: ScalarType::U32,
                    offset: 0,
                }));
                push_consts.push((
                    ScalarType::U32,
                    index,
                    parse_quote! {
                        #ident: u32
                    },
                ));
            }
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
        let index = info.args.len();
        info.args.push(Arg::Push(PushInfo {
            name: "__krnl_elements".to_string(),
            scalar_type: ScalarType::U32,
            offset: 0,
        }));
        push_consts.push((
            ScalarType::U32,
            index,
            parse_quote! {
                __krnl_elements: u32
            },
        ));
    }
    push_consts.sort_by_key(|x| -(x.0.size() as i32));
    let mut offset = 0;
    for (scalar_type, index, field) in push_consts.iter() {
        match unwrap!(info.args.get_mut(*index)) {
            Arg::Push(push_info) => {
                push_info.offset = offset;
            }
            _ => unreachable!(),
        }
        offset += scalar_type.size() as u32;
    }
    let mut push_consts = push_consts
        .into_iter()
        .map(|(_, _, x)| x)
        .collect::<Punctuated<_, Comma>>();
    if offset % 4 != 0 {
        let pad_bytes = 4 - (offset % 4);
        for i in 0..pad_bytes {
            let name = format!("__krnl_pad_{i}");
            let ident = format_ident!("{name}");
            push_consts.push(parse_quote! {
                #ident: u8
            });
            info.args.push(Arg::Push(PushInfo {
                name,
                scalar_type: ScalarType::U8,
                offset,
            }));
            offset += 1;
        }
    }
    let kernel_name = &info.name;
    let kernel_ident = format_ident!("{kernel_name}");
    if let Ok("1") = var("KRNL_DEVICE_CRATE_CHECK").as_deref() {
        assert_eq!(unwrap!(var("CARGO_PRIMARY_PACKAGE")), "1");
        let mut map = KernelInfoMap::lock();
        map.insert(info);
        map.save();
        return Ok(TokenStream::new());
    }
    let kernel_attrs = &kernel.attrs;
    let host_tokens = {
        let kernel_info_bytes = unwrap!(bincode::serialize(&info));
        let dispatch_safety = if info.safe {
            None
        } else {
            Some(Unsafe::default())
        };
        let mut dispatch_args = Punctuated::<FnArg, Comma>::new();
        let mut elementwise_slice_ident = None;
        let dispatch_dim = if info.elementwise {
            let mut dispatch_dim = quote! {
                DispatchDim::groups([0].as_ref())
            };
            for slice_info in info.slice_infos() {
                if slice_info.elementwise {
                    let ident = format_ident!("{}", slice_info.name);
                    dispatch_dim = quote! {
                        DispatchDim::GlobalThreads([#ident.len() as u32].as_ref())
                    };
                    elementwise_slice_ident.replace(ident);
                    break;
                }
            }
            dispatch_dim
        } else {
            let dimensionalty = info.dimensionality as usize;
            dispatch_args.push(parse_quote! {
                dispatch_dim: DispatchDim<[u32; #dimensionalty]>
            });
            quote! {
                dispatch_dim.as_ref()
            }
        };
        let mut dispatch_slices = Punctuated::<TokenStream2, Comma>::new();
        let mut dispatch_push_consts = TokenStream2::new();
        let mut push_const_bytes = 0;
        for arg in info.args.iter() {
            match arg {
                Arg::Slice(slice_info) => {
                    let name = &slice_info.name;
                    let ident = format_ident!("{name}");
                    let slice_ty = if slice_info.mutable {
                        format_ident!("SliceMut")
                    } else {
                        format_ident!("Slice")
                    };
                    let elem_ty = format_ident!("{}", slice_info.scalar_type.name());
                    dispatch_args.push(parse_quote! {
                        #ident: #slice_ty<#elem_ty>
                    });
                    dispatch_slices.push(quote! {
                        (#name, #ident.as_scalar_slice())
                    });
                }
                Arg::Push(push_info) => {
                    let elem_ty = format_ident!("{}", push_info.scalar_type.name());
                    let start = push_info.offset as usize;
                    let end = start + push_info.scalar_type.size();
                    if push_info.name == "__krnl_elements" {
                        let ident = unwrap!(elementwise_slice_ident.as_ref());
                        dispatch_push_consts.extend(quote! {
                            dispatch_push_consts[#start..#end].copy_from_slice((#ident.len() as u32).to_ne_bytes().as_slice());
                        });
                    } else if push_info.name.starts_with("__krnl_len_") {
                        let name = push_info.name.split_at("__krnl_len_".len()).1;
                        let ident = format_ident!("{}", name);
                        dispatch_push_consts.extend(quote! {
                            dispatch_push_consts[#start..#end].copy_from_slice((#ident.len() as u32).to_ne_bytes().as_slice());
                        });
                    } else if push_info.name.starts_with("__krnl_pad_") {
                        // padding
                    } else {
                        let ident = format_ident!("{}", push_info.name);
                        dispatch_push_consts.extend(quote! {
                            dispatch_push_consts[#start..#end].copy_from_slice(#ident.to_ne_bytes().as_slice());
                        });
                        dispatch_args.push(parse_quote! {
                            #ident: #elem_ty
                        });
                    }
                    push_const_bytes += push_info.scalar_type.size();
                }
            }
        }
        let half_import = if info.args.iter().any(|x| {
            use ScalarType::*;
            let scalar_type = match x {
                Arg::Slice(x) => x.scalar_type,
                Arg::Push(x) => x.scalar_type,
            };
            matches!(scalar_type, F16 | BF16)
        }) {
            quote! {
                krnl_core::half::{f16, bf16},
            }
        } else {
            TokenStream2::new()
        };
        quote! {
            #[cfg(not(target_arch = "spirv"))]
            #(#kernel_attrs)*
            pub mod #kernel_ident {
                use super::{__krnl, __krnl_spirv};
                use __krnl::{
                    anyhow::Result,
                    bincode,
                    #half_import
                    __private::once_cell::sync::Lazy,
                    device::Device,
                    buffer::{Slice, SliceMut},
                    kernel::{kernel_info::{KernelInfo, Spirv}, DispatchDim, KernelBase, __build, __dispatch}
                };
                use std::sync::Arc;

                pub struct Kernel {
                    base: KernelBase,
                }
                impl Kernel {
                    pub fn build(device: Device) -> Result<Self> {
                        static INFO: Lazy<Arc<KernelInfo>> = Lazy::new(|| {
                            let mut info: KernelInfo = bincode::deserialize(&[#(#kernel_info_bytes),*]).unwrap();
                            info.path = module_path!().to_string();
                            Arc::new(info)
                        });
                        static SPIRV: Lazy<Arc<Spirv>> = Lazy::new(|| {
                            Arc::new(__krnl_spirv!(#kernel_ident))
                        });
                        let kernel_info = INFO.clone();
                        let spirv = SPIRV.clone();
                        let base = __build(device, kernel_info, spirv)?;
                        Ok(Self {
                            base
                        })
                    }
                    #[forbid(unsafe_op_in_unsafe_fn)]
                    pub #dispatch_safety fn dispatch(&self, #dispatch_args) -> Result<()> {
                        let dispatch_slices = &[
                            #dispatch_slices
                        ];
                        let mut dispatch_push_consts = [0u8; #push_const_bytes];
                        #dispatch_push_consts
                        unsafe {
                            __dispatch(&self.base, #dispatch_dim, dispatch_slices, dispatch_push_consts.as_slice())
                        }
                    }
                }
                pub fn build(device: Device) -> Result<Kernel> {
                    Kernel::build(device)
                }
            }
        }
    };
    let device_tokens = {
        let push_const_idents = push_consts
            .iter()
            .map(|x| &x.ident)
            .collect::<Punctuated<_, Comma>>();
        let push_consts_ident = format_ident!("__{kernel_ident}PushConsts");
        let call_args = inner_args
            .iter()
            .map(|x| &x.ident)
            .collect::<Punctuated<_, Comma>>();
        outer_args.push(parse_quote! {
            #[spirv(push_constant)] push_consts: &#push_consts_ident
        });
        let mut builtin_stmts = Vec::<Stmt>::new();
        if builtins.contains("threads") {
            match info.dimensionality {
                1 => {
                    let thread_lit = unwrap!(thread_lits.first());
                    builtin_stmts.push(parse_quote! {
                        let threads = #thread_lit;
                    });
                }
                2 => {
                    let thread_lits = thread_lits
                        .iter()
                        .cloned()
                        .take(2)
                        .collect::<Punctuated<_, Comma>>();
                    builtin_stmts.push(parse_quote! {
                        let threads = ::krnl_core::glam::UVec2::new(#thread_lits);
                    });
                }
                3 => {
                    builtin_stmts.push(parse_quote! {
                        let threads = ::krnl_core::glam::UVec3::new(#thread_lits);
                    });
                }
                _ => unreachable!(),
            }
        }
        for (builtin, builtin_ty) in BUILTINS.iter() {
            if !matches!(*builtin, "global_threads" | "threads")
                && *builtin_ty == "UVec*"
                && builtins.contains(builtin)
            {
                let ident = format_ident!("{builtin}");
                match info.dimensionality {
                    1 => {
                        builtin_stmts.push(parse_quote! {
                            let #ident = #ident.x;
                        });
                    }
                    2 => {
                        builtin_stmts.push(parse_quote! {
                            let #ident = #ident.truncate();
                        });
                    }
                    3 => (),
                    _ => unreachable!(),
                }
            }
        }
        if builtins.contains("global_threads") {
            builtin_stmts.push(parse_quote! {
                let global_threads = groups * threads;
            });
        }
        builtin_stmts.push(parse_quote! {
            let &#push_consts_ident {
                #push_const_idents
            } = push_consts;
        });
        let mut elementwise_stmts = Vec::<Stmt>::new();
        for slice_info in info.slice_infos() {
            let ident = format_ident!("{}", slice_info.name);
            let mut_token = if slice_info.mutable {
                Some(Mut::default())
            } else {
                None
            };
            if slice_info.elementwise {
                let ident = format_ident!("{}", slice_info.name);
                let index_fn = if slice_info.mutable {
                    format_ident!("index_unchecked_mut")
                } else {
                    format_ident!("index_unchecked")
                };
                elementwise_stmts.push(parse_quote! {
                    let #ident = unsafe {
                        ::krnl_core::spirv_std::arch::IndexUnchecked::#index_fn(#ident, element_index as usize)
                    };
                });
            } else if slice_info.mutable {
                builtin_stmts.push(parse_quote! {
                    let ref mut #ident = ::krnl_core::mem::__global_mut(#ident);
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
        if info.elementwise {
            builtin_stmts.push(parse_quote! {
                let elements = __krnl_elements;
            });
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
        let generated_body = if info.elementwise {
            quote! {
                let element_index = global_id;
                if element_index < elements {
                    #(#elementwise_stmts)*
                    #call
                }
            }
        } else {
            call
        };
        let block = &kernel.block;
        let target_env = format!("vulkan{}.{}", info.vulkan.0, info.vulkan.1);
        let target_features = info
            .capabilities
            .iter()
            .map(|x| format!("{x:?}"))
            .chain(info.extensions.iter().map(|x| format!("ext:{x}")))
            .collect::<Vec<_>>();
        let cfg = quote! {
            #[cfg(all(target_env = #target_env, #(target_feature = #target_features),*))]
        };
        quote! {
            #cfg
            #(#kernel_attrs)*
            #[allow(non_camel_case_types)]
            #[repr(C)]
            pub struct #push_consts_ident {
                #push_consts
            }
            #cfg
            #(#kernel_attrs)*
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
    };
    /*if kernel_name == "axpy_f32" {
        panic!("{:#?}", device_tokens.to_string().split('\n').collect::<Vec<_>>());
    }*/
    Ok(quote! {
        #host_tokens
        #device_tokens
    }
    .into())
}

fn build_or_get_module_spirvs<'a>(
    span: Span,
    module_cache: &'a mut ModuleCache,
    info: ModuleInfo,
) -> Result<&'a HashMap<String, Spirv>> {
    #[cfg(krnl_build)]
    if module_cache.build {
        let file_path: PathBuf = {
            #[cfg(krnl_build)]
            {
                unwrap!(span.source_file().path().canonicalize())
            }
            #[cfg(not(krnl_build))]
            {
                unreachable!()
            }
        };
        let file_path = unwrap!(file_path.strip_prefix(&module_cache.manifest_dir));
        let file_path_string = file_path.to_string_lossy().into_owned();
        let module_name = &info.name;
        if let Some(module_names) = module_cache.module_names.get_mut(&file_path_string) {
            if module_names.iter().find(|x| *x == module_name).is_some() {
                return Err(Error::new(
                    span.into(),
                    format!("module name `{module_name}` already used in {file_path_string:?}"),
                ));
            }
            module_names.push(module_name.clone());
        } else {
            module_cache
                .module_names
                .insert(file_path_string, vec![module_name.clone()]);
        }
        let file_path = file_path.with_extension("");
        let module_crate_dir = module_cache.build_dir.join(&file_path).join(module_name);
        if !module_crate_dir.exists() {
            unwrap!(fs::create_dir_all(&module_crate_dir));
        }
        let module_crate_name = format!(
            "{}_{}",
            file_path.to_string_lossy().replace("/", "_").to_lowercase(),
            module_name
        );
        let mut manifest = format!("[package]\nname = {module_crate_name:?}\n");
        manifest.push_str(
            r#"version = "0.1.0"
    edition = "2021"
    publish = false

    [lib]
    crate-type = ["dylib"]

    [dependencies]
    "#,
        );
        manifest.push_str(&info.dependencies);
        unwrap!(fs::write(
            module_crate_dir.join("Cargo.toml"),
            manifest.as_bytes()
        ));
        let build_script = r#"fn main() { println!("cargo:rustc-cfg=krnl_device_crate") }"#;
        unwrap!(fs::write(
            module_crate_dir.join("build.rs"),
            build_script.as_bytes()
        ));
        let module_crate_src_dir = module_crate_dir.join("src");
        if !module_crate_src_dir.exists() {
            unwrap!(fs::create_dir(&module_crate_src_dir));
        }
        unwrap!(fs::write(
            module_crate_src_dir.join("lib.rs"),
            info.source.as_bytes()
        ));
        let _ = Command::new("cargo")
            .arg("fmt")
            .current_dir(&module_crate_dir)
            .env_remove("RUSTUP_TOOLCHAIN")
            .env_remove("CARGO_CFG_krnl_build")
            .env("RUSTFLAGS", "--cfg krnl_device_crate_check")
            .env("KRNL_DEVICE_CRATE_CHECK", "1")
            .status();
        let status = unwrap!(Command::new("cargo")
            .args(&["+nightly", "check",])
            .current_dir(&module_crate_dir)
            .env_remove("RUSTUP_TOOLCHAIN")
            .env_remove("CARGO_CFG_krnl_build")
            .env("RUSTFLAGS", "--cfg krnl_device_crate_check")
            .env("KRNL_DEVICE_CRATE_CHECK", "1")
            .status());
        assert!(status.success(), "check failed!");
        let status = unwrap!(Command::new("cargo")
            .args(&[
                "run",
                "--release",
                "--",
                &*module_crate_dir.to_string_lossy(),
            ])
            .current_dir(&module_cache.builder_dir)
            .env_remove("RUSTUP_TOOLCHAIN")
            .env_remove("CARGO_CFG_krnl_build")
            .env_remove("RUSTFLAGS")
            .status());
        assert!(status.success(), "build failed!");
        let spirvs_bytes = unwrap!(fs::read(module_crate_dir.join("spirvs.bincode")));
        let spirvs = unwrap!(bincode::deserialize(&spirvs_bytes));
        return Ok(module_cache.insert(info, spirvs));
    }
    module_cache.get(&info).ok_or_else(|| {
        let module_name = &info.name;
        Error::new(span.into(), format!("module `{module_name}` not built or has been modified, rebuild with `RUSTFLAGS=\"--cfg krnl_build\" cargo +nightly check`"))
    })
}

fn module_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let span = Span::call_site();
    let module_attr: ModuleAttributes = syn::parse(attr)?;
    let mut module_item: ModuleItem = syn::parse(item)?;
    let mut new_attr = Vec::with_capacity(module_item.attr.len());
    let mut krnl = parse_quote! {
        krnl
    };
    let mut build = true;
    for attr in module_item.attr {
        if attr.path.segments.len() == 1
            && attr
                .path
                .segments
                .first()
                .map_or(false, |x| x.ident == "krnl")
        {
            let args: ModuleKrnlArgs = syn::parse(attr.tokens.clone().into())?;
            for arg in args.args.iter() {
                if let Some(krnl_crate) = arg.krnl_crate.as_ref() {
                    krnl = krnl_crate.clone();
                } else if let Some(krnl_build) = arg.krnl_build.as_ref() {
                    build = krnl_build.value();
                } else {
                    let ident = arg.ident.as_ref().unwrap();
                    return Err(Error::new_spanned(
                        ident,
                        format!("unknown krnl arg `{ident}`, expected `build` or `crate`"),
                    ));
                }
            }
        } else {
            new_attr.push(attr)
        }
    }
    module_item.attr = new_attr;
    let name = module_item.ident.to_string();
    let binary_name = var("CARGO_BIN_NAME").ok();
    let mut dependencies = String::new();
    let mut has_krnl_core = false;
    for attr in module_attr.attr.iter() {
        if let Some(dependency) = attr.dependency.as_ref() {
            if dependency.name.value() == "krnl-core" {
                has_krnl_core = true;
            }
            dependencies.push_str(&dependency.to_toml_string()?);
            dependencies.push('\n');
        } else {
            todo!()
        }
    }
    if !has_krnl_core {
        let krnl_core_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../krnl-core");
        dependencies.push_str(&format!("\"krnl-core\" = {{ path = {krnl_core_dir:?} }}"));
    }
    let mut source = r#"#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv),
)]
"#
    .to_string();
    source.push_str(&module_item.tokens.to_string());
    let module_info = ModuleInfo {
        name,
        binary_name,
        dependencies,
        source,
    };
    let mut spirv_arms = TokenStream2::new();
    let mut module_cache = ModuleCache::load();
    if build {
        let module_spirvs = build_or_get_module_spirvs(span, &mut module_cache, module_info)?;
        for (kernel, spirv) in module_spirvs.iter() {
            let words = &spirv.words;
            let ident = format_ident!("{}", kernel);
            spirv_arms.extend(quote! {
                (#ident) => (
                    Spirv {
                        words: vec![#(#words),*],
                    }
                );
            });
        }
        if cfg!(krnl_build) {
            module_cache.save();
        }
    } else {
        spirv_arms = quote! {
            ($k:tt) => (
                unimplemented!("`#[module]` has attribute `#[krnl(build = false)]`")
            )
        };
    }
    module_item.tokens.extend(quote! {
        #[doc(hidden)]
        use #krnl as __krnl;
        #[doc(hidden)]
        use __krnl::krnl_core;
        macro_rules! __krnl_spirv {
            #spirv_arms
        }
        #[doc(hidden)]
        use __krnl_spirv;
    });
    Ok(module_item.to_token_stream().into())
}
