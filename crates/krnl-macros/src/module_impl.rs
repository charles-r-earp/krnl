use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{ItemMod, Result};

pub fn module(item: TokenStream) -> Result<TokenStream> {
    let item: ItemMod = syn::parse2(item)?;
    match std::env::var("KRNLC").as_deref() {
        Ok("expand") => module_host(item, true),
        Ok("device") => module_device(item),
        _ => module_host(item, false),
    }
}

fn module_host(item: ItemMod, expand: bool) -> Result<TokenStream> {
    let content = if let Some((_, content)) = item.content {
        content
            .into_iter()
            .flat_map(|x| x.to_token_stream())
            .collect()
    } else {
        TokenStream::new()
    };
    let source = content.to_string();
    let ident = item.ident;
    let attrs = item.attrs;
    let attrs = if !attrs.is_empty() {
        quote! { #[#(#attrs),*] }
    } else {
        TokenStream::new()
    };
    let module_data = if expand {
        quote! {
            mod __krnl_module_data {
                #[allow(non_upper_case_globals)]
                const __krnl_module_source: &'static str = #source;
            }
        }
    } else {
        TokenStream::new()
    };
    let tokens = quote! {
        #attrs
        mod #ident {
            #module_data
            use krnl::krnl_core;
            use krnl_core::macros::kernel;
            #content
        }
    };
    Ok(tokens)
}

fn module_device(item: ItemMod) -> Result<TokenStream> {
    todo!()
}
