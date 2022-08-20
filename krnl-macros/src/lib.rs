#![cfg_attr(feature = "build", feature(proc_macro_span))]
#![cfg_attr(not(feature = "build"), allow(dead_code))]

use krnl_types::__private::{
    raw_module::{RawModule, RawKernelInfo, Safety}
};
use std::{fs, sync::Arc, path::PathBuf};
use proc_macro::TokenStream;
use proc_macro2::{TokenStream as TokenStream2, Span};
use quote::{quote, ToTokens};
use syn::{Error, parse_quote, ItemMod, ItemFn, LitStr, punctuated::Punctuated, Ident, token::{Comma, Paren, Bracket, Eq}};
use derive_syn_parse::Parse;

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

#[allow(unused_variables)]
fn kernel_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let span = Span::call_site();
    let module_path = std::env::var("KRNL_MODULE_PATH").ok();
    if let Some(module_path) = module_path.as_ref() {
        let bytes = fs::read(&module_path).into_syn_result(span)?;
        let mut raw_module: RawModule = bincode::deserialize(&bytes).into_syn_result(span)?;
        let kernel_info = Arc::new(RawKernelInfo {
            name: "foo".into(),
            target: raw_module.target.clone(),
            capabilities: Vec::new(),
            extensions: Vec::new(),
            safety: Safety::Safe,
            slice_infos: Vec::new(),
            push_infos: Vec::new(),
            threads: vec![1],
            spirv: None,
        });
        raw_module.kernels.insert(kernel_info.name.clone(), kernel_info);
        let bytes = bincode::serialize(&raw_module).into_syn_result(span)?;
        fs::write(&module_path, &bytes).into_syn_result(span)?;
    }
    Ok(quote! {
        #[cfg(target_arch = "spirv")]
        #[spirv(compute(threads(1)))] pub fn foo() {}
    }.into())
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
    key_values: Punctuated<DependencyKeyValue, Comma>
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
                let value = self.value.as_ref().ok_or_else(|| Error::new_spanned(&self.key, "expected a str, found a list"))?;
                let mut value_string = value.value();
                if key_string.as_str() == "path" {
                    value_string = PathBuf::from(value_string).canonicalize().into_syn_result(value.span())?
                        .to_string_lossy().into_owned();
                }
                s.push('"');
                s.push_str(&value_string);
                s.push('"');
            }
            "features" => {
                let value_list = self.value_list.as_ref().ok_or_else(|| Error::new_spanned(&self.key, "expected a list, found a str"))?;
                s.push_str(&value_list.to_toml_string());
            }
            _ => {
                return Err(Error::new_spanned(&self.key, format!("unexpected dependency key {key_string}")));
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
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")
        .into_syn_result(span)?);
    let module_name = module.ident.to_string();
    let krnl_dir = manifest_dir.join("target")
        .join(".krnl");
    fs::create_dir_all(&krnl_dir).into_syn_result(span)?;
    let builder_dir = krnl_dir.join("builder");
    if !builder_dir.exists() {
        if !builder_dir.exists() {
            fs::create_dir(&builder_dir).into_syn_result(span)?;
        }
        let manifest =
r#"[package]
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
        let main =
r#"use krnl_builder::ModuleBuilder;

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
"#);

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
            }.to_tokens(&mut tokens);
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

    let module_dir = manifest_dir.join("src")
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
        return Err(Error::new(span, format!("Failed to compile module {module_name:?}!")));
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
    }.to_string();
    let module_name = module.ident.to_string();
    #[cfg(feature = "build")] {
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
