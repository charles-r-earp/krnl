use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{Block, File, Result, Stmt, parse_quote};

pub enum Context {
    Host,
    Device,
}

impl ToTokens for Context {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Host => {
                quote! {
                    #[cfg(not(target_arch = "spirv"))]
                }
                .to_tokens(tokens);
            }
            Self::Device => {
                quote! {
                    #[cfg(all(krnlc, target_arch = "spirv"))]
                }
                .to_tokens(tokens);
            }
        }
    }
}

pub fn only(input: TokenStream, context: Context) -> Result<TokenStream> {
    let Block { mut stmts, .. } = parse_quote!({ #input });
    Ok(quote! {
        #(
            #context
            #stmts
        )*
    })
}
