use krnl_core::__private::{
    KRNL_MODULE_PATH,
    raw_module::{RawModule, KernelInfo, Spirv, Target}
};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Error;

type Result<T, E = syn::Error> = Result<T, E>;

fn kernel_impl() -> Result<TokenStream2> {
    let krnl_module_path_result = std::env::var(KRNL_MODULE_PATH);
    let mut raw_module = if let Ok(krnl_module_path) = krnl_module_path_result.as_ref() {
        bincode::deserialize(&fs::read(krnl_module_path)?)?;
    } else {
        todo!()
    };
    let kernl_info = KernelInfo {
        name: "foo".into(),
        target: raw_module.target,
        capabilities: Vec::new(),
        safety: Safety::Safe,
        slice_infos: Vec::new(),
        push_infos: Vec::new(),
        threads: vec![1],
        spirv: None,
    };
    raw_module.insert(kernel_info.name.clone(), kernl_info);
    if let Ok(krnl_module_path) = krnl_module_path_result.as_ref() {
        fs::write(&krnl_module_path, bincode::serialize(&raw_module)?)?;
    }
    Ok(quote! {
        #[spirv(compute(threads(1)))] pub fn foo() {}
    })
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match kernel_impl() {
        Ok(x) => x.into(),
        Err(e) => e.into_compile_error().into(),
    }
}
