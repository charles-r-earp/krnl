use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, Block, Error, Expr, File, FnArg, Ident, LitInt, PatType, Result, Stmt, Type,
    TypeParamBound, TypePath, Visibility, WhereClause, bracketed, parenthesized,
    parse::Parse,
    parse_quote, parse2,
    punctuated::Punctuated,
    spanned::Spanned,
    token::{
        And, Brace, Bracket, Colon, Comma, Gt, Impl, Let, Lt, Mut, Paren, PathSep, Plus, Pound,
        Semi, Unsafe,
    },
    visit_mut::VisitMut,
};
use syn_derive::{Parse, ToTokens};

mod kw {
    syn::custom_keyword!(kernel);
    syn::custom_keyword!(spec);
    syn::custom_keyword!(item);
    syn::custom_keyword!(group);
    syn::custom_keyword!(len);
    syn::custom_keyword!(UnsafeCell);
    syn::custom_keyword!(debug);
    syn::custom_keyword!(debug_pretty);
    syn::custom_keyword!(no_build);
    syn::custom_keyword!(global_thread_id);
    syn::custom_keyword!(global_threads);
    syn::custom_keyword!(threads);
    syn::custom_keyword!(thread_id);
}

#[derive(Clone, Copy, PartialEq, Eq, derive_more::Display, Debug)]
#[repr(u8)]
enum ScalarType {
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

impl ScalarType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "u8" => Some(Self::U8),
            "i8" => Some(Self::I8),
            "u16" => Some(Self::U16),
            "i16" => Some(Self::I16),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "u32" => Some(Self::U32),
            "i32" => Some(Self::I32),
            "f32" => Some(Self::F32),
            "u64" => Some(Self::U64),
            "i64" => Some(Self::I64),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }
    fn iter() -> impl Iterator<Item = Self> + Clone {
        use ScalarType::*;
        [U8, I8, U16, I16, F16, BF16, U32, I32, F32, U64, I64, F64].into_iter()
    }
    fn ident(&self) -> Ident {
        format_ident!("{self}")
    }
    fn type_path(&self) -> TypePath {
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
    fn expr(&self) -> Expr {
        let ident = format_ident!("{self:?}");
        parse_quote! {
            krnl::scalar::ScalarType::#ident
        }
    }
    fn traits(&self) -> ScalarTraits {
        use ScalarType::*;
        match self {
            U8 | U16 | U32 | U64 => ScalarTraits {
                prim_int: true,
                unsigned: true,
                ..ScalarTraits::default()
            },
            I8 | I16 | I32 | I64 => ScalarTraits {
                prim_int: true,
                signed: true,
                ..ScalarTraits::default()
            },
            F16 | BF16 | F32 | F64 => ScalarTraits {
                signed: true,
                float: true,
                ..ScalarTraits::default()
            },
        }
    }
}

#[derive(Clone)]
struct ScalarTypeIdent {
    ident: Ident,
}

impl Parse for ScalarTypeIdent {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let ident: Ident = input
            .parse()
            .map_err(|e| Error::new(e.span(), "expected scalar"))?;
        if ScalarType::from_str(&ident.to_string()).is_none() {
            return Err(Error::new(ident.span(), "expected scalar"));
        }
        Ok(Self { ident })
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct ScalarTraits {
    prim_int: bool,
    unsigned: bool,
    signed: bool,
    float: bool,
}

impl ScalarTraits {
    fn from_bounds(bounds: &Punctuated<TypeParamBound, Plus>) -> Self {
        let mut traits = Self::default();
        for bound in bounds {
            if let TypeParamBound::Trait(bound) = bound {
                if let Some(ident) = bound.path.get_ident() {
                    if ident == "PrimInt" {
                        traits.prim_int = true;
                    } else if ident == "Unsigned" {
                        traits.unsigned = true;
                    } else if ident == "Signed" {
                        traits.signed = true;
                    } else if ident == "Float" {
                        traits.float = true;
                    }
                }
            }
        }
        traits
    }
    fn scalars(self) -> impl Iterator<Item = ScalarType> + Clone {
        ScalarType::iter().filter(move |x| {
            let traits = x.traits();
            if self.prim_int && !traits.prim_int {
                return false;
            }
            if self.unsigned && !traits.unsigned {
                return false;
            }
            if self.signed && !traits.signed {
                return false;
            }
            if self.float && !traits.float {
                return false;
            }
            true
        })
    }
}

fn monomorphize(ty: &mut Type, generics: &[Ident], scalars: &[ScalarType]) {
    struct ElemVisitor<'a> {
        generics: &'a [Ident],
        scalars: &'a [ScalarType],
    }

    let mut visitor = ElemVisitor { generics, scalars };

    impl VisitMut for ElemVisitor<'_> {
        fn visit_type_path_mut(&mut self, i: &mut TypePath) {
            if let Some(path_ident) = i.path.get_ident() {
                if let Some(index) = self.generics.iter().position(|x| x == path_ident) {
                    *i = self.scalars[index].type_path();
                }
            }
            syn::visit_mut::visit_type_path_mut(self, i);
        }
    }

    syn::visit_mut::visit_type_mut(&mut visitor, ty);
}

trait FnArgExt {
    fn unwrap_pat_type(self) -> PatType;
}

impl FnArgExt for FnArg {
    fn unwrap_pat_type(self) -> PatType {
        if let Self::Typed(x) = self {
            x
        } else {
            unreachable!()
        }
    }
}

struct Kernel {
    attrs: Vec<Attribute>,
    kernel_attr: KernelAttr,
    vis: Visibility,
    sig: KernelSignature,
    group_inputs: Vec<GroupInput>,
    block: Box<Block>,
}

impl Parse for Kernel {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let mut attrs = Attribute::parse_outer(input)?;
        let mut kernel_attr = KernelAttr::default();
        for attr in attrs.iter() {
            if attr.meta.path().is_ident("kernel") {
                let kattr: KernelAttr = parse2(attr.to_token_stream())?;
                kernel_attr.args.extend(kattr.args);
            }
        }
        attrs.retain(|attr| !attr.meta.path().is_ident("kernel"));
        let vis = input.parse()?;
        let sig: KernelSignature = input.parse()?;
        let mut block: Box<Block> = Box::new(input.parse()?);
        let group_inputs = GroupInput::parse_body(&mut block.stmts)?;
        Ok(Self {
            attrs,
            kernel_attr,
            vis,
            sig,
            group_inputs,
            block,
        })
    }
}

impl Kernel {
    fn tokens(&self) -> Result<TokenStream> {
        let host_tokens = self.host_tokens()?;
        let device_tokens = self.device_tokens()?;
        let tokens = quote! {
            #host_tokens
            #device_tokens
        };
        if self
            .kernel_attr
            .args
            .iter()
            .any(|x| matches!(x, KernelAttrArg::Debug(_)))
        {
            eprintln!("{tokens}");
        } else if self
            .kernel_attr
            .args
            .iter()
            .any(|x| matches!(x, KernelAttrArg::DebugPretty(_)))
        {
            let file: File = syn::parse2(tokens.clone()).unwrap();
            eprintln!("{}", prettyplease::unparse(&file));
        }
        Ok(tokens)
    }
    fn host_tokens(&self) -> Result<TokenStream> {
        let Self { attrs: _, vis, .. } = self;
        let ident = &self.sig.ident;
        let ty_generic_idents: Vec<Ident> = self
            .sig
            .generics
            .params
            .iter()
            .map(|x| x.ident.clone())
            .collect();
        let (impl_generics, ty_generics, _where_clause) = self.sig.generics.split_for_impl();
        let kernel_struct = if !ty_generic_idents.is_empty() {
            quote! {
                struct #ident <#(#ty_generic_idents),*> {
                    _m: std::marker::PhantomData<(#(#ty_generic_idents,)*)>
                }
            }
        } else {
            quote! {
                struct #ident {}
            }
        };
        let safety = if self.sig.unsafety.is_none() {
            quote! { krnl::kernel::Safe }
        } else {
            quote! { () }
        };
        let build_args = self
            .sig
            .inputs
            .iter()
            .filter(|x| x.is_spec())
            .map(|x| &x.ty);
        let args = self.sig.inputs.iter().filter_map(|x| x.host_arg_ty());
        let kernel_import_visit = self.kernel_import_visit();
        let visit_build_args = self
            .sig
            .inputs
            .iter()
            .filter(|x| x.is_spec())
            .enumerate()
            .map(|(i, x)| {
                let ident = &x.ident;
                let name = ident.to_string();
                let index = LitInt::new(&i.to_string(), Span::call_site());
                let stmt: Stmt = parse_quote! {
                    v.__visit_spec(#name, &args.#index);
                };
                stmt
            });
        let visit_args = {
            let mut index = 0;
            self.sig.inputs.iter().filter_map(move |x| {
                let arg = x.visit(index)?;
                index += 1;
                Some(arg)
            })
        };
        let type_check = self.input_type_check();
        Ok(quote! {
            #[cfg(not(target_arch = "spirv"))]
            #[allow(non_camel_case_types)]
            #vis #kernel_struct
            #type_check
            #[cfg(not(target_arch = "spirv"))]
            unsafe impl #impl_generics krnl::kernel::KernelDef for #ident #ty_generics {
                type Safety = #safety;
                type BuildArgs = (#(#build_args,)*);
                type Args<'a> = (#(#args,)*);
                fn __visit_build_args<V: krnl::kernel::__private::__BuildArgsVisitor>(
                    args: &Self::BuildArgs,
                    v: &mut V,
                ) {
                    #kernel_import_visit
                    #(#visit_build_args)*
                }
                fn __visit_args<V: krnl::kernel::__private::__ArgsVisitor>(args: &mut Self::Args<'_>, v: &mut V) -> krnl::Result<()> {
                    #(#visit_args)*
                    Ok(())
                }
            }
        })
    }
    fn device_tokens(&self) -> Result<TokenStream> {
        let ident = &self.sig.ident;
        let unsafety = self.sig.unsafety;
        let (impl_generics, ty_generics, where_clause) = self.sig.generics.split_for_impl();
        let kernel_args = self.kernel_args();
        let kernel_arg_idents = self.kernel_args().map(|x| x.pat);
        let kernel_block = &self.block;
        let kernel = quote! {
            #unsafety fn __krnl_kernel #impl_generics (
                #(#kernel_args,)*
            ) #where_clause
            #kernel_block
        };
        let ty_generics_turbofish = ty_generics.as_turbofish();
        let kernel_call = quote! {
            #unsafety {
                __krnl_kernel #ty_generics_turbofish (#(#kernel_arg_idents,)*);
            }
        };
        let is_item = self.sig.inputs.iter().any(|x| x.is_item());
        let device_decls = self.device_decls();
        let body = if is_item {
            let item_loads_stores = self.sig.inputs.iter().filter_map(|x| {
                if x.is_item() {
                    let ident = &x.ident;
                    if x.ty.is_scalar_mut() {
                        Some(quote! {
                            let #ident = unsafe { &mut *#ident[__krnl_item_id].get() };
                        })
                    } else {
                        Some(quote! {
                            let #ident = #ident[__krnl_item_id];
                        })
                    }
                } else {
                    None
                }
            });
            quote! {
                let mut __krnl_item_id = __krnl_global_thread_id;
                while __krnl_item_id < __krnl_items {
                    #(#item_loads_stores)*
                    #kernel_call
                    __krnl_item_id += __krnl_global_threads;
                }
                #kernel
            }
        } else {
            quote! {
                #kernel_call
                #kernel
            }
        };
        let tokens = if self.sig.generics.params.is_empty() {
            let entry_point_args = self.entry_point_args();
            quote! {
                #[cfg(target_arch = "spirv")]
                #[krnl::spirv_std::spirv(compute(threads(1)))]
                pub #unsafety fn #ident(
                    #(#entry_point_args),*
                ) {
                    #device_decls
                    #body
                }
            }
        } else {
            let type_param_idents: Vec<_> = self
                .sig
                .generics
                .params
                .iter()
                .map(|x| x.ident.clone())
                .collect();
            self.sig
                .generics
                .scalar_permutations()
                .map(|scalar_type_idents| {
                    let scalars: Vec<_> = scalar_type_idents
                        .iter()
                        .map(|x| ScalarType::from_str(&x.ident.to_string()).unwrap())
                        .collect();
                    let type_defs =
                        self.sig
                            .generics
                            .params
                            .iter()
                            .zip(scalars.iter())
                            .map(|(p, s)| {
                                let ident = &p.ident;
                                let ty = s.type_path();
                                quote! {
                                    type #ident = #ty;
                                }
                            });
                    let entry_point_name = self.entry_point_name(&scalars);
                    let entry_point_args = self.entry_point_args().map(|mut x| {
                        monomorphize(&mut x.ty, &type_param_idents, &scalars);
                        x
                    });
                    let entry_point_ident = Ident::new(&entry_point_name, self.sig.ident.span());
                    quote! {
                        #[cfg(target_arch = "spirv")]
                        #[krnl::spirv_std::spirv(compute(threads(1)))]
                        pub #unsafety fn #entry_point_ident(
                            #(#entry_point_args),*
                        ) {
                            #(#type_defs)*
                            #device_decls
                            #body
                        }
                    }
                })
                .collect()
        };
        Ok(tokens)
    }
    fn entry_point_name(&self, scalars: &[ScalarType]) -> String {
        let ident = &self.sig.ident;
        let generic_ident_name = self
            .sig
            .generics
            .params
            .iter()
            .zip(scalars.iter())
            .map(|(p, s)| format!("{}{s}", p.ident.to_string().to_lowercase()))
            .join("_");
        format!("{ident}_{generic_ident_name}")
    }
    fn entry_point_args(&self) -> impl Iterator<Item = PatType> + '_ {
        let uvec3 = quote! {
            krnl::spirv_std::glam::UVec3
        };
        let builtins = [
            parse_quote!(#[spirv(num_workgroups)] __krnl_groups: #uvec3),
            parse_quote!(#[spirv(workgroup_id)] __krnl_group_id: #uvec3),
            parse_quote!(#[spirv(local_invocation_id)] __krnl_thread_id: #uvec3),
        ]
        .into_iter()
        .map(FnArg::unwrap_pat_type);
        let mut binding = 0;
        let args = self
            .sig
            .inputs
            .iter()
            .filter_map(move |x| x.entry_point_arg(&mut binding));
        let mut binding = 0;
        let group_buffers = self
            .group_inputs
            .iter()
            .map(move |x| x.entry_point_arg(&mut binding));
        builtins.chain(args).chain(group_buffers)
    }
    fn kernel_args(&self) -> impl Iterator<Item = PatType> + '_ {
        let inputs = self.sig.inputs.iter().map(|x| x.kernel_arg());
        let group_inputs = self.group_inputs.iter().map(|x| x.kernel_arg());
        inputs.chain(group_inputs)
    }
    fn input_type_check(&self) -> TokenStream {
        let (impl_generics, _ty_generics, _where_clause) = self.sig.generics.split_for_impl();
        let inputs = self.sig.inputs.iter().map(
            |KernelInput {
                 ident,
                 colon_token,
                 ty,
                 ..
             }| quote! { #ident #colon_token #ty },
        );
        let group_inputs = self.group_inputs.iter().map(
            |GroupInput {
                 ident,
                 colon_token,
                 ty,
                 ..
             }| quote! { #ident #colon_token #ty },
        );
        let inputs = inputs.chain(group_inputs);
        let input_checks = self.sig.inputs.iter().map(|x| {
            let KernelInput { ident, ty, .. } = x;
            if x.is_builtin() {
                quote! {
                    __krnl_builtin::<#ty>(#ident);
                }
            } else if x.is_spec() {
                quote! {
                    __krnl_spec::<#ty>(#ident);
                }
            } else if x.is_item() {
                if let KernelInputType::ScalarMut { elem, .. } = ty {
                    quote! {
                        __krnl_item_mut::<#elem>(#ident);
                    }
                } else {
                    quote! {
                        __krnl_item::<#ty>(#ident);
                    }
                }
            } else if let KernelInputType::Buffer { elem, .. } = ty {
                quote! {
                    __krnl_slice::<#elem>(#ident);
                }
            } else if let KernelInputType::UnsafeBuffer { elem, .. } = ty {
                quote! {
                    __krnl_unsafe_slice::<#elem>(#ident);
                }
            } else {
                quote! {
                    __krnl_push::<#ty>(#ident);
                }
            }
        });
        let group_checks = self.group_inputs.iter().map(|x| {
            let GroupInput { ident, ty, .. } = x;
            let elem = &ty.elem;
            quote! {
                __krnl_unsafe_slice::<#elem>(#ident);
            }
        });
        quote! {
            const _: () = {
                fn __krnl_kernel #impl_generics (#(#inputs),*) {
                    fn __krnl_builtin<T>(_: usize) {}
                    fn __krnl_spec<T: krnl::scalar::Scalar>(_: T) {}
                    fn __krnl_push<T: krnl::scalar::Element>(_: T) {}
                    fn __krnl_slice<T: krnl::scalar::Element>(_: &[T]) {}
                    fn __krnl_unsafe_slice<T: krnl::scalar::Element>(_: &[::core::cell::UnsafeCell<T>]) {}
                    fn __krnl_item<T: krnl::scalar::Element>(_: T) {}
                    fn __krnl_item_mut<T: krnl::scalar::Element>(_: &mut T) {}

                    #(#input_checks)*
                    #(#group_checks)*
                }
            };
        }
    }
    fn device_decls(&self) -> TokenStream {
        let safe = if self.sig.unsafety.is_none() {
            quote! {
                unsafe { krnl::kernel::__private::__safe() };
            }
        } else {
            TokenStream::new()
        };
        let builtins = quote! {
            let __krnl_groups = __krnl_groups.x as usize;
            let __krnl_group_id = __krnl_group_id.x as usize;
            let __krnl_threads = unsafe { krnl::kernel::__private::__threads() } as usize;
            let __krnl_thread_id = __krnl_thread_id.x as usize;
            let __krnl_global_threads = __krnl_groups * __krnl_threads;
            let __krnl_global_thread_id = __krnl_group_id * __krnl_threads + __krnl_thread_id;
        };
        let items = {
            let mut items = self
                .sig
                .inputs
                .iter()
                .filter(|x| x.is_item())
                .map(|x| x.ident.clone())
                .peekable();
            if let Some(first) = items.next() {
                if items.peek().is_some() {
                    quote! {
                        let __krnl_items = #first.len() #(.min(#items.len()))*;
                    }
                } else {
                    quote! {
                        let __krnl_items = #first.len();
                    }
                }
            } else {
                TokenStream::new()
            }
        };
        let inputs = self.sig.inputs.iter().map(move |x| x.device_decl());
        quote! {
            #safe
            #builtins
            #items
            #(#inputs)*
        }
    }
    fn kernel_import_visit(&self) -> TokenStream {
        let no_build = self
            .kernel_attr
            .args
            .iter()
            .any(|x| matches!(x, KernelAttrArg::NoBuild(_)));

        fn import(
            name: &str,
            no_build: bool,
            safety: TokenStream,
            inputs: impl Iterator<Item = KernelImportInput>,
        ) -> TokenStream {
            let var = quote! {
                concat!("KRNL_INCLUDE_", module_path!())
            };
            let ident = Ident::new(name, Span::call_site());
            if no_build {
                quote! {
                    {
                        mod #ident {
                            const _ENV: &'static str = #var;
                        }
                    }
                }
            } else {
                quote! {
                    {
                        mod #ident {
                            use super::*;
                            include!(concat!(env!("OUT_DIR"), "/", env!(#var)));

                            pub(super) fn visit<V: krnl::kernel::__private::__BuildArgsVisitor>(v: &mut V) {
                                __krnl_Kernel {
                                    __krnl_safety: krnl::kernel::__private::__Safety::<#safety>::__new(),
                                    #(#inputs),*
                                }.visit(v);
                            }
                        }
                        #ident::visit(v);
                    }
                }
            }
        }
        let inputs = self.sig.inputs.iter().filter_map(|x| x.kernel_import());
        let is_generic = !self.sig.generics.params.is_empty();
        let safety = if self.sig.unsafety.is_some() {
            quote!(())
        } else {
            quote!(krnl::kernel::Safe)
        };
        if !is_generic {
            return import(&self.sig.ident.to_string(), no_build, safety, inputs);
        }
        let inputs: Vec<_> = inputs.collect();
        let type_param_idents: Vec<_> = self
            .sig
            .generics
            .params
            .iter()
            .map(|x| x.ident.clone())
            .collect();
        let arms = self
            .sig
            .generics
            .scalar_permutations()
            .map(|scalar_type_idents| {
                let scalars: Vec<_> = scalar_type_idents
                    .iter()
                    .map(|x| ScalarType::from_str(&x.ident.to_string()).unwrap())
                    .collect();
                let name = self.entry_point_name(&scalars);
                let inputs = inputs.iter().cloned().map(|mut x| {
                    monomorphize(&mut x.ty, &type_param_idents, &scalars);
                    x
                });
                let import = import(&name, no_build, safety.clone(), inputs);
                let generics = scalars.iter().map(|x| x.expr());
                quote! {
                    (#(#generics,)*) => #import
                }
            });
        let generic_types = self.sig.generics.params.iter().map(|x| {
            let ident = &x.ident;
            quote! {
                krnl::scalar::ScalarType::of::<#ident>()
            }
        });
        quote! {
            match (#(#generic_types,)*) {
                #(#arms,)*
                _ => (),
            }
        }
    }
}

#[derive(Default)]
struct KernelAttr {
    pound_token: Pound,
    bracket: Bracket,
    kernel: kw::kernel,
    paren: Paren,
    args: Punctuated<KernelAttrArg, Comma>,
}

impl Parse for KernelAttr {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let pound_token = input.parse()?;
        let in_bracket;
        let bracket = bracketed!(in_bracket in input);
        let kernel = in_bracket.parse()?;
        let in_paren;
        let paren = parenthesized!(in_paren in in_bracket);
        let args = Punctuated::parse_terminated(&in_paren)?;
        Ok(Self {
            pound_token,
            bracket,
            kernel,
            paren,
            args,
        })
    }
}

enum KernelAttrArg {
    Debug(kw::debug),
    DebugPretty(kw::debug_pretty),
    NoBuild(kw::no_build),
}

impl Parse for KernelAttrArg {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        if input.peek(kw::debug) {
            input.parse().map(Self::Debug)
        } else if input.peek(kw::debug_pretty) {
            input.parse().map(Self::DebugPretty)
        } else if input.peek(kw::no_build) {
            input.parse().map(Self::NoBuild)
        } else {
            Err(Error::new(
                input.span(),
                "expected debug, debug_pretty, no_build",
            ))
        }
    }
}

impl ToTokens for KernelAttrArg {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Debug(x) => {
                x.to_tokens(tokens);
            }
            Self::DebugPretty(x) => {
                x.to_tokens(tokens);
            }
            Self::NoBuild(x) => {
                x.to_tokens(tokens);
            }
        }
    }
}

#[derive(Parse)]
struct KernelSignature {
    unsafety: Option<Unsafe>,
    fn_token: syn::token::Fn,
    ident: Ident,
    generics: KernelGenerics,
    #[syn(parenthesized)]
    paren_token: Paren,
    #[syn(in = paren_token)]
    #[parse(Punctuated::parse_terminated)]
    inputs: Punctuated<KernelInput, Comma>,
}

#[derive(Default)]
struct KernelGenerics {
    lt: Option<Lt>,
    params: Punctuated<KernelTypeParam, Comma>,
    gt: Option<Gt>,
    where_clause: Option<WhereClause>,
}

struct KernelImplGenerics<'a>(&'a KernelGenerics);

impl ToTokens for KernelImplGenerics<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.0.lt.is_none() {
            return;
        }
        self.0.lt.to_tokens(tokens);
        self.0.params.to_tokens(tokens);
        self.0.gt.to_tokens(tokens);
    }
}

struct KernelTypeGenerics<'a>(&'a KernelGenerics);

impl ToTokens for KernelTypeGenerics<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.0.lt.is_none() {
            return;
        }
        self.0.lt.to_tokens(tokens);
        for (i, param) in self.0.params.iter().enumerate() {
            if i > 0 {
                Comma(Span::call_site()).to_tokens(tokens);
            }
            param.ident.to_tokens(tokens);
        }
        self.0.gt.to_tokens(tokens);
    }
}

impl<'a> KernelTypeGenerics<'a> {
    fn as_turbofish(&self) -> KernelTypeGenericsTurboFish<'a> {
        KernelTypeGenericsTurboFish(self.0)
    }
}

struct KernelTypeGenericsTurboFish<'a>(&'a KernelGenerics);

impl ToTokens for KernelTypeGenericsTurboFish<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.0.lt.is_none() {
            return;
        }
        PathSep::default().to_tokens(tokens);
        self.0.lt.to_tokens(tokens);
        for (i, param) in self.0.params.iter().enumerate() {
            if i > 0 {
                Comma(Span::call_site()).to_tokens(tokens);
            }
            param.ident.to_tokens(tokens);
        }
        self.0.gt.to_tokens(tokens);
    }
}

impl KernelGenerics {
    fn split_for_impl(
        &self,
    ) -> (
        KernelImplGenerics<'_>,
        KernelTypeGenerics<'_>,
        Option<&'_ WhereClause>,
    ) {
        (
            KernelImplGenerics(self),
            KernelTypeGenerics(self),
            self.where_clause.as_ref(),
        )
    }
    fn scalar_permutations(&self) -> impl Iterator<Item = Vec<ScalarTypeIdent>> {
        self.params
            .iter()
            .map(|x| {
                let scalar_type_idents: Vec<ScalarTypeIdent> =
                    if let Some(attr) = x.kernel_attr.as_ref() {
                        attr.arg.types.iter().cloned().collect()
                    } else {
                        ScalarTraits::from_bounds(&x.bounds)
                            .scalars()
                            .map(|x| ScalarTypeIdent { ident: x.ident() })
                            .collect()
                    };
                scalar_type_idents
            })
            .multi_cartesian_product()
    }
}

impl Parse for KernelGenerics {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        if input.peek(Lt) {
            let lt = input.parse()?;
            let mut params = Punctuated::new();
            loop {
                params.push_value(input.parse()?);
                if input.peek(Comma) {
                    params.push_punct(input.parse()?);
                } else {
                    break;
                }
                if input.peek(Gt) {
                    break;
                }
            }
            let gt = input.parse()?;
            Ok(Self {
                lt: Some(lt),
                params,
                gt: Some(gt),
                where_clause: None,
            })
        } else {
            Ok(Self::default())
        }
    }
}

struct KernelTypeParamAttr {
    pound_token: Pound,
    bracket_token: Bracket,
    kernel: kw::kernel,
    paren_token: Paren,
    arg: KernelTypeParamAttrArg,
}

impl KernelTypeParamAttr {
    fn from_attrs(attrs: &mut Vec<Attribute>) -> Result<Option<Self>> {
        for (i, attr) in attrs.iter().enumerate() {
            if attr.path().is_ident("kernel") {
                let attr = attrs.remove(i);
                return Ok(Some(syn::parse2(attr.to_token_stream())?));
            }
        }
        Ok(None)
    }
}

impl Parse for KernelTypeParamAttr {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let pound_token = input.parse()?;
        let in_bracket;
        let bracket_token = bracketed!(in_bracket in input);
        let kernel = in_bracket.parse()?;
        let in_paren;
        let paren_token = parenthesized!(in_paren in in_bracket);
        let arg = in_paren.parse()?;
        Ok(Self {
            pound_token,
            bracket_token,
            kernel,
            paren_token,
            arg,
        })
    }
}

struct KernelTypeParamAttrArg {
    impl_token: Impl,
    eq_token: syn::token::Eq,
    bracket_token: Bracket,
    types: Punctuated<ScalarTypeIdent, Comma>,
}

impl Parse for KernelTypeParamAttrArg {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let impl_token = input.parse()?;
        let eq_token = input.parse()?;
        let in_bracket;
        let bracket_token = bracketed!(in_bracket in input);
        let types = Punctuated::parse_separated_nonempty(&in_bracket)?;
        Ok(Self {
            impl_token,
            eq_token,
            bracket_token,
            types,
        })
    }
}

struct KernelTypeParam {
    attrs: Vec<Attribute>,
    kernel_attr: Option<KernelTypeParamAttr>,
    ident: Ident,
    colon_token: Colon,
    bounds: Punctuated<TypeParamBound, Plus>,
}

impl Parse for KernelTypeParam {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let mut attrs = Attribute::parse_outer(input)?;
        let kernel_attr = KernelTypeParamAttr::from_attrs(&mut attrs)?;
        let ident = input.parse()?;
        let colon_token = input.parse()?;
        let mut bounds = Punctuated::new();
        loop {
            bounds.push_value(input.parse()?);
            if input.peek(Plus) {
                bounds.push_punct(input.parse::<Plus>()?);
            } else {
                break;
            }
        }
        if bounds.is_empty() {
            return Err(Error::new(input.span(), "expected generic bound"));
        }
        Ok(Self {
            attrs,
            kernel_attr,
            ident,
            colon_token,
            bounds,
        })
    }
}

impl ToTokens for KernelTypeParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for attr in self.attrs.iter() {
            attr.to_tokens(tokens);
        }
        self.ident.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
    }
}

/*
struct KernelSpecParam {
    attrs: Vec<Attribute>,
    spec: kw::spec,
    ident: Ident,
    colon_token: Colon,
    ty: Type,
}
*/

struct KernelInputAttr {
    pound_token: Pound,
    bracket_token: Bracket,
    kernel: kw::kernel,
    paren_token: Paren,
    arg: KernelInputAttrArg,
}

impl Parse for KernelInputAttr {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let pound_token = input.parse()?;
        let in_bracket;
        let bracket_token = bracketed!(in_bracket in input);
        let kernel = in_bracket.parse()?;
        let in_paren;
        let paren_token = parenthesized!(in_paren in in_bracket);
        let arg = in_paren.parse()?;
        Ok(Self {
            pound_token,
            bracket_token,
            kernel,
            paren_token,
            arg,
        })
    }
}

#[derive(derive_more::IsVariant)]
enum Builtin {
    GlobalThreads(kw::global_threads),
    GlobalThreadId(kw::global_thread_id),
    Threads(kw::threads),
    ThreadId(kw::thread_id),
}

impl Builtin {
    fn to_ident(&self) -> Ident {
        match self {
            Self::GlobalThreads(x) => Ident::new("__krnl_global_threads", x.span),
            Self::GlobalThreadId(x) => Ident::new("__krnl_global_thread_id", x.span),
            Self::Threads(x) => Ident::new("__krnl_threads", x.span),
            Self::ThreadId(x) => Ident::new("__krnl_thread_id", x.span),
        }
    }
}

impl ToTokens for Builtin {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::GlobalThreads(x) => {
                x.to_tokens(tokens);
            }
            Self::GlobalThreadId(x) => {
                x.to_tokens(tokens);
            }
            Self::Threads(x) => {
                x.to_tokens(tokens);
            }
            Self::ThreadId(x) => {
                x.to_tokens(tokens);
            }
        }
    }
}

#[derive(derive_more::IsVariant)]
enum KernelInputAttrArg {
    Spec(kw::spec),
    Item(kw::item),
    Builtin(Builtin),
}

impl Parse for KernelInputAttrArg {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        if input.peek(kw::spec) {
            Ok(Self::Spec(input.parse()?))
        } else if input.peek(kw::item) {
            Ok(Self::Item(input.parse()?))
        } else if input.peek(kw::global_threads) {
            Ok(Self::Builtin(Builtin::GlobalThreads(input.parse()?)))
        } else if input.peek(kw::global_thread_id) {
            Ok(Self::Builtin(Builtin::GlobalThreadId(input.parse()?)))
        } else if input.peek(kw::threads) {
            Ok(Self::Builtin(Builtin::Threads(input.parse()?)))
        } else if input.peek(kw::thread_id) {
            Ok(Self::Builtin(Builtin::ThreadId(input.parse()?)))
        } else {
            Err(Error::new(
                input.span(),
                "expected `spec`, `item`, or builtin",
            ))
        }
    }
}

impl ToTokens for KernelInputAttrArg {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Spec(x) => {
                x.to_tokens(tokens);
            }
            Self::Item(x) => {
                x.to_tokens(tokens);
            }
            Self::Builtin(x) => {
                x.to_tokens(tokens);
            }
        }
    }
}

struct KernelImportInput {
    ident: Ident,
    colon_token: Colon,
    kind: Ident,
    ty: Type,
}

/*
impl KernelImportInput {
    fn spec_constant(const_param: &ConstParam) -> Self {
        Self {
            ident: const_param.ident.clone(),
            colon_token: const_param.colon_token,
            kind: format_ident!("__Spec"),
            ty: parse2(const_param.ty.to_token_stream()).unwrap(),
        }
    }
}
*/

impl Clone for KernelImportInput {
    fn clone(&self) -> Self {
        Self {
            ident: self.ident.clone(),
            colon_token: self.colon_token,
            kind: self.kind.clone(),
            ty: parse2(self.ty.to_token_stream()).unwrap(),
        }
    }
}

impl ToTokens for KernelImportInput {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        let kind = &self.kind;
        let ty = &self.ty;
        quote! {
            krnl::kernel::__private::#kind::<#ty>::__new()
        }
        .to_tokens(tokens);
    }
}

struct KernelInput {
    attrs: Vec<Attribute>,
    kernel_attr: Option<KernelInputAttr>,
    ident: Ident,
    colon_token: Colon,
    ty: KernelInputType,
}

impl Parse for KernelInput {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let mut attrs = Attribute::parse_outer(input)?;
        let mut kernel_attr = None;
        for (i, attr) in attrs.iter().enumerate() {
            if attr.meta.path().is_ident("kernel") {
                let attr = attrs.remove(i);
                let attr: KernelInputAttr = parse2(attr.to_token_stream())?;
                kernel_attr.replace(attr);
                break;
            }
        }
        if kernel_attr.is_some() {
            for attr in attrs.iter() {
                if attr.meta.path().is_ident("kernel") {
                    return Err(Error::new(attr.span(), "duplicate `kernel(..)` attribute"));
                }
            }
        }
        let ident = input.parse()?;
        let colon_token = input.parse()?;
        let ty = if let Some(kernel_attr) = kernel_attr.as_ref() {
            match &kernel_attr.arg {
                KernelInputAttrArg::Spec(_) | KernelInputAttrArg::Builtin(_) => {
                    KernelInputType::parse_scalar(input)?
                }
                KernelInputAttrArg::Item(_) => KernelInputType::parse_item(input)?,
            }
        } else if input.peek(And) {
            KernelInputType::parse_buffer(input)?
        } else {
            KernelInputType::parse_scalar(input)?
        };
        Ok(Self {
            attrs,
            kernel_attr,
            ident,
            colon_token,
            ty,
        })
    }
}

impl KernelInput {
    fn is_spec(&self) -> bool {
        self.kernel_attr
            .as_ref()
            .map(|x| x.arg.is_spec())
            .unwrap_or_default()
    }
    fn is_item(&self) -> bool {
        self.kernel_attr
            .as_ref()
            .map(|x| x.arg.is_item())
            .unwrap_or_default()
    }
    fn is_builtin(&self) -> bool {
        self.kernel_attr
            .as_ref()
            .map(|x| x.arg.is_builtin())
            .unwrap_or_default()
    }
    fn is_push(&self) -> bool {
        self.kernel_attr.is_none() && self.ty.is_scalar()
    }
    fn host_arg_ty(&self) -> Option<Type> {
        if self.is_builtin() || self.is_spec() {
            return None;
        }
        match &self.ty {
            KernelInputType::Scalar { elem } if self.is_push() => {
                Some(parse2(elem.to_token_stream()).unwrap())
            }
            KernelInputType::Scalar { elem } | KernelInputType::Buffer { elem, .. } => {
                Some(parse_quote! {
                    krnl::buffer::Slice<'a, #elem>
                })
            }
            KernelInputType::ScalarMut { elem, .. }
            | KernelInputType::UnsafeBuffer { elem, .. } => Some(parse_quote! {
                krnl::buffer::SliceMut<'a, #elem>
            }),
        }
    }
    fn device_decl(&self) -> Stmt {
        let Self {
            attrs: _,
            ident,
            ty: _,
            ..
        } = self;
        if let Some(KernelInputAttrArg::Builtin(builtin)) =
            self.kernel_attr.as_ref().map(|x| &x.arg)
        {
            let builtin = builtin.to_ident();
            parse_quote! {
                let #ident = #builtin;
            }
        } else if self.is_spec() {
            parse_quote! {
                let #ident = {
                    unsafe {
                        krnl::kernel::__private::__spec_constant(#ident);
                    }
                    *#ident
                };
            }
        } else if self.is_push() {
            parse_quote! {
                let #ident = {
                    unsafe {
                        krnl::kernel::__private::__push_constant(#ident);
                    }
                    *#ident
                };
            }
        } else if self.is_item() {
            if self.ty.is_scalar_mut() {
                parse_quote! {
                    unsafe {
                        krnl::kernel::__private::__item_mut(#ident);
                    }
                }
            } else {
                parse_quote! {
                    unsafe {
                        krnl::kernel::__private::__item(#ident);
                    }
                }
            }
        } else if self.ty.is_unsafe_buffer() {
            parse_quote! {
                unsafe {
                    krnl::kernel::__private::__unsafe_slice(#ident);
                }
            }
        } else {
            parse_quote! {
                unsafe {
                    krnl::kernel::__private::__slice(#ident);
                }
            }
        }
    }
    fn entry_point_arg(&self, binding: &mut usize) -> Option<PatType> {
        if self.is_builtin() {
            return None;
        }
        let Self {
            attrs: _,
            ident,
            ty,
            colon_token,
            ..
        } = self;
        if self.is_spec() || self.is_push() {
            return Some(FnArg::unwrap_pat_type(parse_quote! {
                #[spirv(push_constant)]
                #ident #colon_token &#ty
            }));
        }
        let binding_lit = LitInt::new(&binding.to_string(), Span::call_site());
        *binding += 1;
        let attr = quote! {
            #[spirv(storage_buffer, descriptor_set = 0, binding = #binding_lit)]
        };
        match &self.ty {
            KernelInputType::Scalar { elem } => Some(FnArg::unwrap_pat_type(parse_quote! {
                #attr
                #ident #colon_token &[#elem]
            })),
            KernelInputType::ScalarMut { elem, .. } => Some(FnArg::unwrap_pat_type(parse_quote! {
                #attr
                #ident #colon_token &[::core::cell::UnsafeCell<#elem>]
            })),
            KernelInputType::Buffer { .. } | KernelInputType::UnsafeBuffer { .. } => {
                Some(FnArg::unwrap_pat_type(parse_quote! {
                    #attr
                    #ident #colon_token #ty
                }))
            }
        }
    }
    fn kernel_arg(&self) -> PatType {
        let Self {
            attrs,
            ident,
            colon_token,
            ty,
            ..
        } = self;
        let arg = parse_quote! {
            #(#attrs)*
            #ident
            #colon_token
            #ty
        };
        if let FnArg::Typed(pat_ty) = arg {
            pat_ty
        } else {
            unreachable!()
        }
    }
    fn kernel_import(&self) -> Option<KernelImportInput> {
        if self.is_builtin() {
            return None;
        }
        let Self {
            attrs: _,
            ident,
            ty,
            colon_token,
            ..
        } = self;
        let ident = ident.clone();
        let colon_token = *colon_token;
        if self.is_spec() {
            let ty = parse2(ty.to_token_stream()).unwrap();
            Some(KernelImportInput {
                ident,
                colon_token,
                kind: format_ident!("__Spec"),
                ty,
            })
        } else if self.is_push() {
            let ty = parse2(ty.to_token_stream()).unwrap();
            Some(KernelImportInput {
                ident,
                colon_token,
                kind: format_ident!("__Push"),
                ty,
            })
        } else if self.is_item() {
            if let KernelInputType::ScalarMut { elem, .. } = ty {
                Some(KernelImportInput {
                    ident,
                    colon_token,
                    kind: format_ident!("__ItemMut"),
                    ty: parse2(elem.to_token_stream()).unwrap(),
                })
            } else {
                let ty = parse2(ty.to_token_stream()).unwrap();
                Some(KernelImportInput {
                    ident,
                    colon_token,
                    kind: format_ident!("__Item"),
                    ty,
                })
            }
        } else {
            let mut tokens = TokenStream::new();
            if let KernelInputType::Buffer {
                and: _,
                bracket,
                elem,
                len,
            } = ty
            {
                bracket.surround(&mut tokens, |tokens| {
                    elem.to_tokens(tokens);
                    if let Some((semi, len)) = len {
                        semi.to_tokens(tokens);
                        len.to_tokens(tokens);
                    }
                });
            } else if let KernelInputType::UnsafeBuffer {
                and: _,
                bracket,
                unsafe_cell,
                lt,
                elem,
                gt,
                len,
            } = ty
            {
                bracket.surround(&mut tokens, |tokens| {
                    quote!(::core::cell::#unsafe_cell #lt #elem #gt).to_tokens(tokens);
                    if let Some((semi, len)) = len {
                        semi.to_tokens(tokens);
                        len.to_tokens(tokens);
                    }
                });
            };
            let ty = parse2(tokens).unwrap();
            Some(KernelImportInput {
                ident,
                colon_token,
                kind: format_ident!("__Buffer"),
                ty,
            })
        }
    }
    fn visit(&self, index: usize) -> Option<Stmt> {
        if self.is_builtin() || self.is_spec() {
            return None;
        }
        let name = self.ident.to_string();
        let index = LitInt::new(&index.to_string(), Span::call_site());
        let visit = if self.is_push() {
            "__visit_push"
        } else if self.ty.is_scalar() {
            "__visit_item"
        } else if self.ty.is_scalar_mut() {
            "__visit_item_mut"
        } else if self.ty.is_buffer() {
            "__visit_slice"
        } else {
            "__visit_slice_mut"
        };
        let visit = format_ident!("{visit}");
        let mut_token = match &self.ty {
            KernelInputType::ScalarMut { mut_token, .. } => Some(*mut_token),
            KernelInputType::UnsafeBuffer { unsafe_cell, .. } => Some(Mut(unsafe_cell.span)),
            _ => None,
        };
        let question = if !self.is_push() {
            Some(syn::token::Question::default())
        } else {
            None
        };
        Some(parse_quote! {
            v.#visit(#name, & #mut_token args.#index) #question;
        })
    }
}

#[derive(derive_more::IsVariant)]
enum KernelInputType {
    Scalar {
        elem: Type,
    },
    ScalarMut {
        and: And,
        mut_token: Mut,
        elem: Type,
    },
    Buffer {
        and: And,
        bracket: Bracket,
        elem: Type,
        len: Option<(Semi, Expr)>,
    },
    UnsafeBuffer {
        and: And,
        bracket: Bracket,
        unsafe_cell: kw::UnsafeCell,
        lt: Lt,
        elem: Type,
        gt: Gt,
        len: Option<(Semi, Expr)>,
    },
}

impl KernelInputType {
    fn parse_scalar(input: syn::parse::ParseStream) -> Result<Self> {
        let elem = input.parse()?;
        Ok(Self::Scalar { elem })
    }
    fn parse_item(input: syn::parse::ParseStream) -> Result<Self> {
        if input.peek(And) {
            let and = input.parse()?;
            let mut_token = input.parse()?;
            let elem = input.parse()?;
            Ok(Self::ScalarMut {
                and,
                mut_token,
                elem,
            })
        } else {
            let elem = input.parse()?;
            Ok(Self::Scalar { elem })
        }
    }
    fn parse_buffer(input: syn::parse::ParseStream) -> Result<Self> {
        let and = input.parse()?;
        if input.peek(Mut) {
            return Err(input.error("unexpected `&mut[_]`, try `&[UnsafeCell<_>]`"));
        }
        let in_bracket;
        let bracket = bracketed!(in_bracket in input);
        if in_bracket.peek(kw::UnsafeCell) {
            let unsafe_cell = in_bracket.parse()?;
            let lt = in_bracket.parse()?;
            let elem = in_bracket.parse()?;
            let gt = in_bracket.parse()?;
            let len = if in_bracket.peek(Semi) {
                let semi = in_bracket.parse()?;
                let len = in_bracket.parse()?;
                Some((semi, len))
            } else {
                None
            };
            Ok(Self::UnsafeBuffer {
                and,
                bracket,
                unsafe_cell,
                lt,
                elem,
                gt,
                len,
            })
        } else {
            let elem = in_bracket.parse()?;
            let len = if in_bracket.peek(Semi) {
                let semi = in_bracket.parse()?;
                let len = in_bracket.parse()?;
                Some((semi, len))
            } else {
                None
            };
            Ok(Self::Buffer {
                and,
                bracket,
                elem,
                len,
            })
        }
    }
}

impl ToTokens for KernelInputType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Scalar { elem } => {
                elem.to_tokens(tokens);
            }
            Self::ScalarMut {
                and,
                mut_token,
                elem,
            } => {
                and.to_tokens(tokens);
                mut_token.to_tokens(tokens);
                elem.to_tokens(tokens);
            }
            Self::Buffer {
                and,
                bracket,
                elem,
                len,
            } => {
                and.to_tokens(tokens);
                bracket.surround(tokens, |tokens| {
                    elem.to_tokens(tokens);
                    if let Some((semi, len)) = len {
                        semi.to_tokens(tokens);
                        len.to_tokens(tokens);
                    }
                });
            }
            Self::UnsafeBuffer {
                and,
                bracket,
                unsafe_cell,
                lt,
                elem,
                gt,
                len,
            } => {
                and.to_tokens(tokens);
                bracket.surround(tokens, |tokens| {
                    unsafe_cell.to_tokens(tokens);
                    lt.to_tokens(tokens);
                    elem.to_tokens(tokens);
                    gt.to_tokens(tokens);
                    if let Some((semi, len)) = len {
                        semi.to_tokens(tokens);
                        len.to_tokens(tokens);
                    }
                });
            }
        }
    }
}

struct GroupInput {
    attrs: Vec<Attribute>,
    kernel_attr: GroupInputAttr,
    let_token: Let,
    ident: Ident,
    colon_token: Colon,
    ty: GroupInputType,
    semi: Semi,
}

impl Parse for GroupInput {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let mut attrs = Attribute::parse_outer(input)?;
        let mut kernel_attr = None;
        for (i, attr) in attrs.iter().enumerate() {
            if attr.meta.path().is_ident("kernel") {
                let attr = attrs.remove(i);
                let attr: GroupInputAttr = parse2(attr.to_token_stream())?;
                kernel_attr.replace(attr);
                break;
            }
        }
        let kernel_attr =
            kernel_attr.ok_or(Error::new(input.span(), "expected #[kernel(group)]"))?;
        let let_token = input.parse()?;
        let ident = input.parse()?;
        let colon_token = input.parse()?;
        let ty = if kernel_attr.slice_len().is_some() {
            GroupInputType::parse_slice(input)?
        } else {
            GroupInputType::parse_array(input)?
        };
        let semi = input.parse()?;
        Ok(Self {
            attrs,
            kernel_attr,
            let_token,
            ident,
            colon_token,
            ty,
            semi,
        })
    }
}

impl ToTokens for GroupInput {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for attr in self.attrs.iter() {
            attr.to_tokens(tokens);
        }
        self.kernel_attr.to_tokens(tokens);
        self.let_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}

impl GroupInput {
    fn parse_body(body: &mut Vec<Stmt>) -> Result<Vec<Self>> {
        let mut output = Vec::new();
        for stmt in body.iter_mut() {
            if let Stmt::Local(stmt_local) = &stmt {
                if let Some(i) = stmt_local
                    .attrs
                    .iter()
                    .position(|attr| attr.path().is_ident("kernel"))
                {
                    let group_input: GroupInput = parse2(stmt.to_token_stream())?;
                    let ident = &group_input.ident;
                    if let Some(len) = group_input.kernel_attr.slice_len() {
                        *stmt = parse_quote! {
                            unsafe {
                                krnl::kernel::__private::__group_slice(#ident, #len);
                            }
                        }
                    } else {
                        *stmt = parse_quote!({});
                    }
                    output.push(group_input);
                }
            }
        }
        Ok(output)
    }
    fn entry_point_arg(&self, binding: &mut usize) -> PatType {
        let Self {
            attrs,
            ident,
            kernel_attr,
            ty,
            ..
        } = self;
        let arg = if kernel_attr.slice_len().is_some() {
            let binding_lit = LitInt::new(&binding.to_string(), Span::call_site());
            *binding += 1;
            parse_quote! {
                #(#attrs)*
                #[spirv(storage_buffer, descriptor_set = 1, binding = #binding_lit)]
                #ident: #ty
            }
        } else {
            parse_quote! {
                #(#attrs)*
                #[spirv(workgroup)]
                #ident: #ty
            }
        };
        if let FnArg::Typed(pat_type) = arg {
            pat_type
        } else {
            unreachable!()
        }
    }
    fn kernel_arg(&self) -> PatType {
        let Self {
            attrs, ident, ty, ..
        } = self;
        let arg = parse_quote! {
            #(#attrs)*
            #ident: #ty
        };
        if let FnArg::Typed(pat_type) = arg {
            pat_type
        } else {
            unreachable!()
        }
    }
}

#[derive(ToTokens)]
struct GroupInputType {
    and: And,
    #[syn(bracketed)]
    bracket_token: Bracket,
    #[syn(in = bracket_token)]
    unsafe_cell: kw::UnsafeCell,
    #[syn(in = bracket_token)]
    lt: Lt,
    #[syn(in = bracket_token)]
    elem: Type,
    #[syn(in = bracket_token)]
    gt: Gt,
    #[syn(in = bracket_token)]
    #[to_tokens(|tokens, val: &Option<(Semi, Expr)>| {
        if let Some((semi, len)) = val {
            semi.to_tokens(tokens);
            len.to_tokens(tokens);
        }
    })]
    len: Option<(Semi, Expr)>,
}

impl GroupInputType {
    fn parse_array(input: syn::parse::ParseStream) -> Result<Self> {
        let and = input.parse()?;
        let in_bracket;
        let bracket_token = bracketed!(in_bracket in input);
        let unsafe_cell = in_bracket.parse()?;
        let lt = in_bracket.parse()?;
        let elem = in_bracket.parse()?;
        let gt = in_bracket.parse()?;
        let semi = in_bracket.parse()?;
        let len = in_bracket.parse()?;
        Ok(Self {
            and,
            bracket_token,
            unsafe_cell,
            lt,
            elem,
            gt,
            len: Some((semi, len)),
        })
    }
    fn parse_slice(input: syn::parse::ParseStream) -> Result<Self> {
        let and = input.parse()?;
        let in_bracket;
        let bracket_token = bracketed!(in_bracket in input);
        let unsafe_cell = in_bracket.parse()?;
        let lt = in_bracket.parse()?;
        let elem = in_bracket.parse()?;
        let gt = in_bracket.parse()?;
        Ok(Self {
            and,
            bracket_token,
            unsafe_cell,
            lt,
            elem,
            gt,
            len: None,
        })
    }
}

struct GroupInputAttr {
    pound_token: Pound,
    bracket_token: Bracket,
    kernel: kw::kernel,
    paren_token: Paren,
    group: kw::group,
    comma: Option<Comma>,
    args: Punctuated<GroupInputAttrArg, Comma>,
}

impl GroupInputAttr {
    fn slice_len(&self) -> Option<&GroupSliceLen> {
        self.args.iter().find_map(|x| {
            if let GroupInputAttrArg::Len(_, _, len) = x {
                Some(len)
            } else {
                None
            }
        })
    }
}

impl Parse for GroupInputAttr {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let pound_token = input.parse()?;
        let in_bracket;
        let bracket_token = bracketed!(in_bracket in input);
        let kernel = in_bracket.parse()?;
        let in_paren;
        let paren_token = parenthesized!(in_paren in in_bracket);
        let group = in_paren.parse()?;
        let comma = in_paren.parse().ok();
        let args = if comma.is_some() {
            Punctuated::parse_terminated(&in_paren)?
        } else {
            Punctuated::new()
        };
        Ok(Self {
            pound_token,
            bracket_token,
            kernel,
            paren_token,
            group,
            comma,
            args,
        })
    }
}

impl ToTokens for GroupInputAttr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.pound_token.to_tokens(tokens);
        self.bracket_token.surround(tokens, |tokens| {
            self.kernel.to_tokens(tokens);
            self.paren_token.surround(tokens, |tokens| {
                self.group.to_tokens(tokens);
                self.comma.to_tokens(tokens);
                self.args.to_tokens(tokens);
            });
        });
    }
}

enum GroupInputAttrArg {
    Len(kw::len, syn::token::Eq, GroupSliceLen),
}

impl Parse for GroupInputAttrArg {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        Ok(Self::Len(input.parse()?, input.parse()?, input.parse()?))
    }
}

impl ToTokens for GroupInputAttrArg {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Len(len, eq, slice_len) => {
                len.to_tokens(tokens);
                eq.to_tokens(tokens);
                slice_len.to_tokens(tokens);
            }
        }
    }
}

#[derive(Debug)]
enum GroupSliceLen {
    LitInt(LitInt),
    Ident(Ident),
    Block(Block),
}

impl Parse for GroupSliceLen {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        if input.peek(LitInt) {
            let lit_int = input.parse()?;
            Ok(Self::LitInt(lit_int))
        } else if input.peek(Ident) {
            let ident = input.parse()?;
            Ok(Self::Ident(ident))
        } else if input.peek(Brace) {
            let block = input.parse()?;
            Ok(Self::Block(block))
        } else {
            Err(Error::new(
                input.span(),
                "expected literal, expression, or block",
            ))
        }
    }
}

impl ToTokens for GroupSliceLen {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::LitInt(x) => x.to_tokens(tokens),
            Self::Ident(x) => x.to_tokens(tokens),
            Self::Block(x) => x.to_tokens(tokens),
        }
    }
}

fn is_rust_analyzer() -> bool {
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        out_dir.contains("rust-analyzer")
    } else {
        false
    }
}

pub fn kernel(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let ra_no_build = if is_rust_analyzer() {
        quote! {
            #[kernel(no_build)]
        }
    } else {
        TokenStream::new()
    };
    let kernel: Kernel = parse2(quote! {
        #ra_no_build
        #[kernel(#attr)]
        #item
    })?;
    kernel.tokens()
}
