#![cfg_attr(feature = "build", feature(proc_macro_span))]
#![cfg_attr(not(feature = "build"), allow(dead_code))]

use derive_syn_parse::Parse;
use krnl_types::{scalar::ScalarType, __private::raw_module::{RawKernelInfo, RawModule, Safety, Mutability, SliceInfo, PushInfo}};
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, format_ident, ToTokens};
use std::{fs, path::PathBuf, sync::Arc, str::FromStr, collections::HashSet};
use syn::{
    parse_quote,
    punctuated::Punctuated,
    token::{And, Bracket, Comma, Eq, Fn, Pub, Mut, Paren, Pound, Colon, Unsafe},
    Block, Error, Ident, ItemFn, ItemMod, LitStr, LitInt, Type, Attribute, Stmt,
    spanned::Spanned,
};
use spirv::Capability;

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

#[derive(Parse, Debug)]
struct KernelAttributes {
    #[call(Punctuated::parse_separated_nonempty)]
    attr: Punctuated<KernelAttribute, Comma>,
}

#[derive(Parse, Debug)]
struct KernelAttribute {
    ident: Ident,
    #[parse_if(ident.to_string() == "target")]
    target: Option<Target>,
    #[parse_if(ident.to_string() == "threads")]
    threads: Option<Threads>,
    #[parse_if(matches!(ident.to_string().as_str(), "capabilities" | "extensions"))]
    features: Option<TargetFeatureList>,
}

#[derive(Parse, Debug)]
struct Target {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    target: LitStr,
}

#[derive(Parse, Debug)]
struct Threads {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_separated_nonempty)]
    threads: Punctuated<LitInt, Comma>,
}

#[derive(Parse, Debug)]
struct TargetFeatureList {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_separated_nonempty)]
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
    #[call(Punctuated::parse_separated_nonempty)]
    args: Punctuated<KernelArg, Comma>,
    block: Block,
}

#[derive(Parse, Debug)]
struct KernelArg {
    #[allow(unused)]
    pound: Option<Pound>,
    #[allow(unused)]
    #[parse_if(pound.is_some())]
    attr: Option<KernelArgAttribute>,
    ident: Ident,
    #[allow(unused)]
    colon: Colon,
    and: Option<And>,
    #[parse_if(and.is_some())]
    mut_token: Option<Mut>,
    ty: Type,
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
    thread_lits: Punctuated<LitInt, Comma>,
    builtins: HashSet<&'static str>,
    outer_args: Punctuated<TypedArg, Comma>,
    inner_args: Punctuated<TypedArg, Comma>,
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
            target: String::new(),
            capabilities: Vec::new(),
            extensions: Vec::new(),
            safety,
            slice_infos: Vec::new(),
            push_infos: Vec::new(),
            elementwise: false,
            threads: Vec::new(),
            spirv: None,
        };
        let mut thread_lits = Punctuated::<LitInt, Comma>::new();
        for attr in kernel_attr.attr.iter() {
            let attr_string = attr.ident.to_string();
            if let Some(target) = attr.target.as_ref() {
                if !info.target.is_empty() {
                    return Err(Error::new_spanned(&attr.ident, "can only have one target"));
                }
                info.target = target.target.value();
            } else if attr_string == "elementwise" {
                info.elementwise = true;
            } else if let Some(threads) = attr.threads.as_ref() {
                if !info.threads.is_empty() {
                    todo!()
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
                                return Err(Error::new_spanned(cap, "unknown capability, see https://docs.rs/spirv/latest/spirv/enum.Capability.html"))
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
            return Err(Error::new(Span::call_site(), "threads(..) must be specified, ie #[kernel(threads(256))]"));
        }
        if info.elementwise && info.threads.len() != 1 {
            return Err(Error::new_spanned(&thread_lits, "can only use 1 dimensional threads in elementwise kernel"));
        }
        if info.elementwise {
            todo!()
        }
        let mut builtins = HashSet::<&'static str>::new();
        let mut outer_args = Punctuated::<TypedArg, Comma>::new();
        let mut inner_args = Punctuated::<TypedArg, Comma>::new();
        let mut push_consts = Vec::<(ScalarType, TypedArg)>::new();
        let mut set_lit = LitInt::new("0", Span::call_site());
        let mut binding = 0;
        for arg in kernel.args.iter() {
            let arg_ident = &arg.ident;
            let ident_string = arg_ident.to_string();
            let arg_ty = &arg.ty;
            let ty_string = arg_ty.to_token_stream().to_string();
            if let Some(arg_attr) = arg.attr.as_ref() {
                let attr_string = arg_attr.ident.to_string();
                match attr_string.as_str() {
                    "builtin" => {
                        if info.elementwise && !matches!(ident_string.as_str(), "elements" | "element_index") {
                            return Err(Error::new_spanned(&arg_ident, "builtin `{ident_string}` can not be used in elementwise kernel, use `elements` or `element_index` instead"));
                        }
                        let (builtin, builtin_ty) = BUILTINS.iter().find(|x| x.0 == ident_string.as_str())
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
                                Error::new_spanned(&arg_attr.ident, msg)
                            })?;
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
                                builtins.extend(["groups", "threads"]);
                            }
                            "threads" => {
                                builtins.insert("threads");
                            }
                            _ => {
                                let spirv_attr = match ident_string.as_str() {
                                    "global_id" => "global_invocation_id",
                                    "groups" => "num_workgroups",
                                    "group_id" => "workgroup_id",
                                    "subgroup_id" => "subgroup_id",
                                    "thread_id" => "local_invocation_id",
                                    "thread_index" => "local_invocation_index",
                                    _ => todo!(),
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
                    }
                    "group" | "subgroup" => {
                        if info.elementwise {
                            return Err(Error::new_spanned(&arg_ident, "#[{attr_string}] can not be used in elementwise kernel"));
                        }
                        let spirv_attr = match attr_string.as_str() {
                            "group" => "workgroup",
                            "subgroup" => "subgroup",
                            _ => unreachable!()
                        };
                        let spirv_attr = Ident::new(spirv_attr, arg.ident.span());
                        outer_args.push(parse_quote! {
                            #[spirv(#spirv_attr)]
                            #arg_ident: #arg_ty
                        });
                        inner_args.push(parse_quote! {
                            #arg_ident: #arg_ty
                        });
                    }
                    _ => todo!(),
                }
            } else if arg.and.is_some() {
                fn get_slice_type(input: &str) -> Option<ScalarType> {
                    let mut iter = input.split(|x| x == '[' || x == ']');
                    if iter.next() == Some("") {
                        if let Some(elem) = iter.next() {
                            if iter.next() == Some("") {
                                return ScalarType::from_str(elem).ok();
                            }
                        }
                    }
                    None
                }
                let mut_token = &arg.mut_token;
                let (scalar_type, elementwise) = if let Ok(scalar_type) = ScalarType::from_str(&ty_string) {
                    (scalar_type, true)
                } else if let Some(scalar_type) = get_slice_type(&ty_string) {
                    (scalar_type, false)
                } else {
                    todo!("invalid slice arg {ty_string}");
                };
                let mutability = if mut_token.is_some() {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                info.slice_infos.push(SliceInfo {
                    name: arg_ident.to_string(),
                    scalar_type,
                    mutability,
                    elementwise,
                });
                inner_args.push(parse_quote! {
                    #arg_ident: & #mut_token #arg_ty
                });
                let elem_ty = Ident::new(scalar_type.name(), arg_ty.span());
                set_lit.set_span(arg.ident.span());
                let binding_lit = LitInt::new(&binding.to_string(), arg.ident.span());
                outer_args.push(parse_quote! {
                    #[spirv(descriptor_set = #set_lit, binding = #binding_lit, storage_buffer)]
                    #arg_ident: & #mut_token [#elem_ty]
                });
                binding += 1;
            } else {
                if let Ok(scalar_type) = ScalarType::from_str(&ty_string) {
                    let typed_arg: TypedArg = parse_quote! {
                        #arg_ident: #arg_ty
                    };
                    push_consts.push((scalar_type, typed_arg.clone()));
                    inner_args.push(typed_arg);
                } else {
                    return Err(Error::new_spanned(&arg_ty, "expected a scalar for push constant type"));
                }
            }
        }
        push_consts.sort_by_key(|x| x.0.size());
        let mut offset = 0;
        for (scalar_type, typed_arg) in push_consts.iter() {
            info.push_infos.push(PushInfo {
                name: typed_arg.ident.to_string(),
                scalar_type: *scalar_type,
                offset: offset,
            });
            offset += scalar_type.size() as u32;
        }
        let mut push_consts = push_consts.into_iter().rev()
            .map(|x| x.1)
            .collect::<Punctuated<_, Comma>>();
        for i in 0 .. offset % 4 {
            let field = format_ident!("{}", i);
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
            thread_lits,
            builtins,
            outer_args,
            inner_args,
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
    let kernel_ident = &kernel.ident;
    let block = &kernel.block;
    let thread_lits = &meta.thread_lits;
    let full_thread_lits = thread_lits.iter()
        .cloned()
        .chain([LitInt::new( "1", span), LitInt::new("1", span)])
        .take(3)
        .collect::<Punctuated<_, Comma>>();
    let mut builtin_stmts = Vec::<Stmt>::new();
    if meta.builtins.contains("threads") {
        builtin_stmts.push(parse_quote! {
            let threads = ::krnl_core::glam::UVec3::new(#full_thread_lits);
        });
    }
    if meta.builtins.contains("global_threads") {
        builtin_stmts.push(parse_quote! {
            let global_threads = groups * threads;
        });
    }
    let outer_args = meta.outer_args;
    let inner_args = meta.inner_args;
    let call_args = inner_args.iter().map(|x| &x.ident).collect::<Punctuated<_, Comma>>();
    let push_consts = meta.push_consts;
    let push_const_idents = push_consts.iter().map(|x| &x.ident).collect::<Punctuated<_, Comma>>();
    let features = meta.info.capabilities.iter()
        .map(|x| format_ident!("{}", format!("{x:?}")))
        .chain(meta.info.extensions.iter().map(|x| format_ident!("ext:{}", x)))
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
        }.to_tokens(&mut output);
        builtin_stmts.push(parse_quote! {
            let &#push_consts_ident {
                #push_const_idents
            } = push_consts;
        });
    }
    quote! {
        #[cfg(all(target_arch = "spirv", #features))]
        #[spirv(compute(threads(#thread_lits)))]
        pub fn #kernel_ident (
            #outer_args
        ) {
            #(#builtin_stmts)*
            fn #kernel_ident (
                #inner_args
            ) {
                #block
            }
            #kernel_ident (
                #call_args
            )
        }
    }.to_tokens(&mut output);
    let module_path = std::env::var("KRNL_MODULE_PATH").ok();
    if let Some(module_path) = module_path.as_ref() {

        let bytes = fs::read(&module_path).into_syn_result(span)?;
        let mut raw_module: RawModule = bincode::deserialize(&bytes).into_syn_result(span)?;
        let mut info = meta.info;
        if info.target.is_empty() {
            info.target = raw_module.target.clone();
        }
        raw_module
            .kernels
            .insert(info.name.clone(), Arc::new(info));
        let bytes = bincode::serialize(&raw_module).into_syn_result(span)?;
        fs::write(&module_path, &bytes).into_syn_result(span)?;
    }
    dbg!(output.to_string());
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
struct ModuleArgs {
    #[call(Punctuated::parse_terminated)]
    args: Punctuated<ModuleArg, Comma>,
}

#[derive(Parse, Debug)]
struct ModuleArg {
    ident: Ident,
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[parse_if(ident.to_string() == "target")]
    target: Option<LitStr>,
    #[inside(paren)]
    #[parse_if(ident.to_string() == "dependency")]
    dependency: Option<Dependency>,
    #[inside(paren)]
    #[parse_if(ident.to_string() == "attr")]
    attr: Option<TokenStream2>,
}

#[derive(Parse, Debug)]
struct Dependency {
    name: LitStr,
    #[allow(unused)]
    comma: Comma,
    #[call(Punctuated::parse_separated_nonempty)]
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
    #[call(Punctuated::parse_separated_nonempty)]
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

#[cfg(feature = "build")]
fn build_module(module_args: ModuleArgs, module: &ItemMod, invocation: &str) -> Result<()> {
    use std::process::Command;

    let source_path = proc_macro::Span::call_site().source_file().path();
    let span = Span::call_site();
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").into_syn_result(span)?);
    let module_name = module.ident.to_string();
    let krnl_dir = manifest_dir.join("target").join(".krnl");
    fs::create_dir_all(&krnl_dir).into_syn_result(span)?;
    let builder_dir = krnl_dir.join("builder");
    if !builder_dir.exists() {
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

        let src_dir = builder_dir.join("src");
        if !src_dir.exists() {
            fs::create_dir(&src_dir).into_syn_result(span)?;
        }
        let main = r#"use krnl_builder::ModuleBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let crate_path = &args[1];
    let target = &args[2];
    let module_path = &args[3];
    let module = ModuleBuilder::new(crate_path, target)
        .build()?;
    let bytes = bincode::serialize(&module.__raw())?;
    std::fs::write(module_path, &bytes)?;
    Ok(())
}"#;
        fs::write(src_dir.join("main.rs"), main).into_syn_result(span)?;
    }

    let crate_dir = krnl_dir.join(&source_path);
    if !crate_dir.exists() {
        fs::create_dir_all(&crate_dir).into_syn_result(span)?;
    }

    let mut manifest = format!("[package]\nname = {module_name:?}");
    manifest.push_str(
        r#"
version = "0.1.0"
edition = "2021"
publish = false

[lib]
crate-type = ["dylib"]

[dependencies]
"#,
    );

    let mut target = None;
    let mut tokens = TokenStream2::new();

    for arg in module_args.args.iter() {
        if let Some(target_arg) = arg.target.as_ref() {
            if target.is_some() {
                return Err(Error::new_spanned(&arg.ident, "can only have one target"));
            }
            target.replace(target_arg.value());
        } else if let Some(dep) = arg.dependency.as_ref() {
            manifest.push_str(&dep.to_toml_string()?);
            manifest.push('\n');
        } else if let Some(attr) = arg.attr.as_ref() {
            quote! {
                #![#attr]
            }
            .to_tokens(&mut tokens);
        }
    }
    if let Some(content) = module.content.as_ref() {
        for item in content.1.iter() {
            item.to_tokens(&mut tokens);
        }
    }
    let source = tokens.to_string();

    fs::write(crate_dir.join("Cargo.toml"), manifest).into_syn_result(span)?;

    let target = target.ok_or_else(|| Error::new(span, "expected a default target, ie vulkan"))?;

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
        .join("src")
        .join(".krnl")
        .join(&source_path)
        .join(&module_name);
    fs::create_dir_all(&module_dir).into_syn_result(span)?;
    let module_path = module_dir.join("module").with_extension("bincode");
    let status = Command::new("cargo")
        .args(&[
            "+nightly-2022-04-18",
            "run",
            "--",
            &*crate_dir.to_string_lossy(),
            &target,
            &*module_path.to_string_lossy(),
        ])
        .current_dir(&builder_dir)
        .status()
        .into_syn_result(span)?;
    if !status.success() {
        return Err(Error::new(
            span,
            format!("Failed to compile module {module_name:?}!"),
        ));
    }
    fs::copy(src_dir.join("lib.rs"), module_dir.join("lib.rs")).into_syn_result(span)?;
    fs::write(module_dir.join("invocation.rs"), &invocation).into_syn_result(span)?;
    Ok(())
}

fn module_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    #[allow(unused)]
    let module_args: ModuleArgs = syn::parse(attr.clone())?;
    let mut module: ItemMod = syn::parse(item.clone())?;
    let attr = TokenStream2::from(attr);
    let item = TokenStream2::from(item);
    let invocation = quote! {
        #[module(#attr)]
        #item
    }
    .to_string();
    let module_name = module.ident.to_string();
    #[cfg(feature = "build")]
    {
        build_module(module_args, &module, &invocation)?;
    }
    let krnl = quote! {
        ::krnl
    };
    let module_fn: ItemFn = parse_quote! {
        pub fn module() -> #krnl::result::Result<#krnl::kernel::Module> {
            use #krnl::{kernel::Module, __private::bincode, krnl_core::__private::raw_module::RawModule};
            use ::std::sync::Arc;
            if cfg!(debug_assertions) {
                let invocation = #invocation;
                let saved_invocation = include_str!(concat!(".krnl/", file!(), "/", #module_name, "/invocation.rs"));
                assert_eq!(invocation, saved_invocation, "module was modified, rebuild with cargo +nightly build --features krnl/build");
            }
            let bytes = include_bytes!(concat!(".krnl/", file!(), "/", #module_name, "/module.bincode"));
            let raw_module: RawModule = bincode::deserialize(bytes)?;
            Ok(Module::__from_raw(Arc::new(raw_module)))
        }
    };
    if let Some(content) = module.content.as_mut() {
        content.1.push(module_fn.into());
    }
    Ok(module.to_token_stream().into())
}

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    match module_impl(attr, item) {
        Ok(x) => x,
        Err(e) => e.into_compile_error().into(),
    }
}
