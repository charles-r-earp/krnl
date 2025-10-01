#![allow(warnings)]

use proc_macro::TokenStream;

mod kernel_impl;
mod krnl_intrinsic_impl;
mod only_impl;

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match kernel_impl::kernel(attr.into(), item.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro]
pub fn host_only(input: TokenStream) -> TokenStream {
    match only_impl::only(input.into(), only_impl::Context::Host) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro]
pub fn device_only(input: TokenStream) -> TokenStream {
    match only_impl::only(input.into(), only_impl::Context::Device) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn krnl_intrinsic(_: TokenStream, input: TokenStream) -> TokenStream {
    match krnl_intrinsic_impl::krnl_intrinisic(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
