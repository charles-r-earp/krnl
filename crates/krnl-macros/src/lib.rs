#![allow(unused_imports, dead_code)]
use derive_syn_parse::Parse;
use proc_macro::TokenStream;
use proc_macro2::{Span as Span2, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
use std::hash::{Hash, Hasher};
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

#[derive(Parse, Debug)]
struct InsideParen<T> {
    #[paren]
    paren: Paren,
    #[inside(paren)]
    value: T,
}

#[proc_macro_attribute]
pub fn module(_: TokenStream, item: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(item as ModuleItem);
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
        use #krnl as __krnl;
    });
    if build {
        let file = syn::parse2(item.tokens.clone()).expect("unable to parse module tokens");
        let source = prettyplease::unparse(&file);
        let mut hasher = std::collections::hash_map::DefaultHasher::default();
        source.hash(&mut hasher);
        let hash = hasher.finish();
        let name_with_hash = format_ident!("{ident}_{hash:x}", ident = item.ident);
        item.tokens.extend(quote! {
            mod __krnl_module_data {
                #[allow(non_upper_case_globals)]
                const __krnl_module_source: &'static str = #source;
            }
            include!(concat!(env!("CARGO_MANIFEST_DIR"), "/krnl-cache.rs"));
            __krnl_module!(#name_with_hash);
        });
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
