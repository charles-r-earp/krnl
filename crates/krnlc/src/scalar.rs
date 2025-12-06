#[cfg(feature = "bindings")]
use quote::format_ident;
#[cfg(feature = "bindings")]
use syn::{Ident, TypePath, parse_quote};

#[derive(Clone, Copy, PartialEq, Eq, derive_more::Display, Debug)]
#[repr(u8)]
pub(crate) enum ScalarType {
    #[display("u8")]
    U8 = 1,
    #[display("i8")]
    I8 = 2,
    #[display("u16")]
    U16 = 3,
    #[display("i16")]
    I16 = 4,
    #[display("f16")]
    F16 = 5,
    #[display("bf16")]
    BF16 = 6,
    #[display("u32")]
    U32 = 7,
    #[display("i32")]
    I32 = 8,
    #[display("u64")]
    U64 = 10,
    #[display("i64")]
    I64 = 11,
    #[display("f32")]
    F32 = 9,
    #[display("f64")]
    F64 = 12,
}

#[cfg(feature = "bindings")]
impl ScalarType {
    pub(crate) fn ident(&self) -> Ident {
        format_ident!("{self}")
    }
    pub(crate) fn type_path(&self) -> TypePath {
        let ident = self.ident();
        if matches!(self, Self::F16 | Self::BF16) {
            parse_quote! {
                krnl::scalar::#ident
            }
        } else {
            parse_quote! {
                #ident
            }
        }
    }
}
