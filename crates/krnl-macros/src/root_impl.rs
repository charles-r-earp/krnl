use fxhash::FxHashMap;
use proc_macro2::{Literal, Span, TokenStream};
use quote::{ToTokens, quote};
use std::path::PathBuf;
use syn::{ItemMod, Result, punctuated::Punctuated};

pub fn root() -> Result<TokenStream> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let krnl_cache_path = manifest_dir.join(".krnl-cache");
    let kernels_bytes = std::fs::read(&krnl_cache_path).unwrap_or_default();
    let kernels: FxHashMap<String, Vec<u8>> = if !kernels_bytes.is_empty() {
        bitcode::decode(&kernels_bytes).unwrap_or_default()
    } else {
        FxHashMap::default()
    };
    let kernel_tokens: TokenStream = kernels
        .into_iter()
        .flat_map(|(name, binary)| {
            dbg!(&name);
            let binary: Punctuated<Literal, syn::token::Comma> =
                binary.into_iter().map(Literal::u8_unsuffixed).collect();
            let (module, name) = name.rsplit_once("::").unwrap();
            quote! {
                (#module, #name) => Some([#binary].as_ref()),
            }
        })
        .collect();
    let tokens = quote! {
        mod __krnl_root {
            pub fn __kernel(module: &str, name: &str) -> Option<&'static [u8]> {
                match (module, name) {
                    #kernel_tokens
                    _ => None,
                }
            }
        }
    };
    Ok(tokens)
}
