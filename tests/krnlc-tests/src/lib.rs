use dry::macro_for;
use krnl::macros::module;
use paste::paste;

/**
```no_run
use krnl::macros::module;

#[module]
#[krnl(no_build)]
mod kernels {
    use krnl::{macros::kernel, device::Device, anyhow::Result};

    #[kernel]
    fn specialization<const X: i32>() {}

    fn test_specialization(device: Device) -> Result<()> {
        specialization::builder()?.specialize(1).build(device)?;
        Ok(())
    }
}
```
```compile_fail
use krnl::macros::module;

#[module]
#[krnl(no_build)]
mod kernels {
    use krnl::{macros::kernel, device::Device, anyhow::Result};

    #[kernel]
    fn specialization<const X: i32>() {}

    fn test_specialization(device: Device) -> Result<()> {
        specialization::builder()?.build(device)?;
        Ok(())
    }
}
```
*/
#[allow(dead_code)]
enum Specialization {}

/**
```no_run
use krnl::macros::module;

#[module]
#[krnl(no_build)]
mod kernels {
    use krnl::{macros::kernel, device::Device, buffer::SliceMut, anyhow::Result};

    #[kernel]
    fn with_groups() {}

    fn test_with_groups(device: Device) -> Result<()> {
        with_groups::builder()?.build(device)?.with_groups(1).dispatch()
    }

    #[kernel]
    fn with_groups_item(
        #[item] y: &mut u32,
    ) {}

    fn test_with_groups_item(y: SliceMut<u32>) -> Result<()> {
        with_groups_item::builder()?.build(y.device())?.dispatch(y)
    }
}
```
```compile_fail
use krnl::macros::module;

#[module]
#[krnl(no_build)]
mod kernels {
    use krnl::{macros::kernel, device::Device, anyhow::Result};

    #[kernel]
    fn with_groups() {}

    fn test_with_groups(device: Device) -> Result<()> {
        with_groups::builder()?.build(device)?.dispatch()
    }
}
```
*/
#[allow(dead_code)]
enum WithGroups {}

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

    #[kernel]
    fn specs<const X: u32, const Y: f32>() {}

    #[cfg(test)]
    #[test]
    fn test_specs() {
        specs::builder().unwrap().specialize(10u32, 1.5f32);
    }

    macro_for!($A in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            #[kernel]
            fn [<basic_ $A>]<const A: $A>(
                #[item] a: &mut $A,
                a_push: $A
            ) {
                *a = a_push + A;
            }

            #[cfg(test)]
            #[test]
            fn [<test_basic_ $A>]() {
                #[allow(unused_imports)]
                use krnl::krnl_core::{num_traits::FromPrimitive, half::{f16, bf16}};
                [<basic_ $A>]::builder().unwrap().specialize($A::from_u32(16).unwrap());
            }
        }
    });

    macro_rules! impl_group_kernel {
        ($($k:ident(|$n:ident| $e:expr)),* $(,)?) => {
            $(
                paste! {
                    #[kernel]
                    unsafe fn [<group_$k>]<const N: u32>(
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
                    #[cfg(test)]
                    #[test]
                    fn [<test_group_ $k>]() {
                        [<group_ $k>]::builder().unwrap().specialize(11);
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

    #[allow(non_snake_case)]
    #[kernel]
    fn attribute(fooBar: u32) {}
}

macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    paste! {
        #[module]
        mod [<kernels_ $T>] {
            pub mod foo {
                #[cfg(not(target_arch = "spirv"))]
                use krnl::krnl_core;
                use krnl_core::macros::kernel;

                #[kernel]
                pub fn foo() {}

                #[cfg(test)]
                #[test]
                fn test_foo() {
                    foo::builder().unwrap();
                }
            }
        }
    }
});

mod parent {
    use krnl::macros::module;

    #[module]
    pub mod functional {
        #[cfg(target_arch = "spirv")]
        pub fn foo() -> i32 {
            1
        }
    }
}

#[module]
mod use_functional {
    #[cfg(target_arch = "spirv")]
    use crate::parent::functional::foo;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    #[kernel]
    fn bar(#[item] y: &mut i32) {
        *y = foo();
    }
}

#[module]
mod dependency {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use test_dependency::add_one;

    #[kernel]
    fn add_one_i32(#[item] x: i32, #[item] y: &mut i32) {
        *y = add_one(x);
    }
}
