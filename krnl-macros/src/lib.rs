
use std::collections::HashMap;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
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
use derive_syn_parse::Parse;

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = syn::parse_macro_input!(attr as ModuleAttr);
    let mut dependencies = None;
    for arg in attr.args {
        if let Some(deps) = arg.dependencies {
            if dependencies.is_some() {
                return Error::new_spanned(
                    &arg.ident,
                    format!("`dependencies` already provided"),
                ).into_compile_error().into();
            }
            dependencies.replace(deps.value.value());
        }
    }
    let dependencies = dependencies.unwrap();
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
                    ).into_compile_error().into();
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
        let module_src_indices = (0 .. module_src.len()).into_iter().collect::<Vec<_>>();
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
        TokenStream2::new()
    };
    item.tokens.extend(quote! {
        #[automatically_derived]
        mod module {
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
