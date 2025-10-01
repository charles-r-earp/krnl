use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{ItemFn, Result, parse_quote, parse2};

pub fn krnl_intrinisic(input: TokenStream) -> Result<TokenStream> {
    let mut item: ItemFn = parse2(input)?;
    let stmts = item.block.stmts;
    let block = parse_quote! {
        {
            #[cfg(all(target_arch = "spirv", not(krnlc)))] {
                compile_error!("krnl intrinisc");
            }
            #[cfg(all(target_arch = "spirv", krnlc))] {
                #(#stmts)*
            }
            #[cfg(not(target_arch = "spirv"))] {
                unreachable!("krnl intrinsic")
            }
        }
    };
    item.block = block;
    Ok(item.to_token_stream())
}
