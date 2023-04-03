use dry::macro_for;
use krnl::macros::module;
use paste::paste;

#[module]
pub mod kernels {
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{
        buffer::UnsafeIndex,
        half::{bf16, f16},
    };
    use paste::paste;

    #[kernel(threads(X))]
    fn spec_threads_1d<#[spec] const X: u32>() {}

    #[kernel(threads(X, Y))]
    fn spec_threads_2d<#[spec] const X: u32, #[spec] const Y: u32>() {}

    #[kernel(threads(X, Y, Z))]
    fn spec_threads_3d<#[spec] const X: u32, #[spec] const Y: u32, #[spec] const Z: u32>() {}

    macro_for!($A in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            #[kernel(threads(256))]
            fn [<basic_ $A>]<#[spec] const A: $A>(
                #[item] a: &mut $A,
                a_push: $A
            ) {
                *a = a_push + A;
            }
        }
    });

    macro_rules! impl_group_kernel {
        ($($k:ident(|$n:ident| $e:expr)),* $(,)?) => {
            $(
                paste! {
                    #[kernel(threads(1))]
                    unsafe fn [<group_$k>]<#[spec] const N: u32>(
                        #[global] x: Slice<f32>,
                        #[group] x_group: UnsafeSlice<f32, { let $n = N; $e }>,
                        #[global] y: UnsafeSlice<f32>,
                    ) {
                        use krnl_core::spirv_std::arch::workgroup_memory_barrier;
                        unsafe {
                            *x_group.unsafe_index_mut(0) = x[0];
                            workgroup_memory_barrier();
                            *y.unsafe_index_mut(0) = *x_group.unsafe_index(0);
                        }
                    }
                }
            )*
        };
    }

    impl_group_kernel!(
        n(|n| n as usize),
        n_times_4_plus_1(|n| (n * 4 + 1) as usize),
        n_div_2(|n| (n / 2) as usize),
    );
}

macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    paste! {
        #[module]
        mod [<kernels_ $T>] {
            #[cfg(not(target_arch = "spirv"))]
            use krnl::krnl_core;
            use krnl_core::macros::kernel;

           #[kernel(threads(1))]
           fn foo() {}
        }
    }
});
