use bytemuck::Pod;
use core::{
    any::TypeId,
    fmt::{Debug, Display},
    num::NonZeroU8,
};
use derive_more::Display;
use dry::macro_for;
pub use half::{bf16, f16};
use paste::paste;

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

pub unsafe trait DeviceCopy: Copy + Send + Sync + Pod + Sealed {
    #[cfg(target_arch = "spirv")]
    unsafe fn __data_type<V>(var: *const V) {}
}

unsafe impl<T: Scalar, const N: usize> DeviceCopy for [T; N] {
    #[cfg(target_arch = "spirv")]
    unsafe fn __data_type<V>(var: *const V) {
        unsafe { T::__data_type(var) }
    }
}

pub unsafe trait Scalar:
    Copy + Send + Sync + Pod + 'static + Element + DeviceCopy + Sealed
{
    const SCALAR_TYPE: ScalarType;
}

#[derive(Clone, Copy, PartialEq, Eq, Display, Debug)]
#[cfg_attr(not(target_arch = "spirv"), repr(u8))]
#[cfg_attr(target_arch = "spirv", repr(u32))]
pub enum ScalarType {
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
    pub const fn of<T: Scalar>() -> Self {
        T::SCALAR_TYPE
    }
    pub const fn size(&self) -> usize {
        use ScalarType::*;
        match *self {
            U8 | I8 => 1,
            U16 | I16 | F16 | BF16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }
}

pub unsafe trait Element: DeviceCopy + Pod + Send + Sync + Sealed {
    type Scalar: Scalar;
}

macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    impl Sealed for $T {}
});

impl<T: Scalar, const N: usize> Sealed for [T; N] {}

macro_for!($T in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64] {
    paste! {
        unsafe impl DeviceCopy for $T {}
    }
});

macro_for!($T in [f16, bf16] {
    paste! {
        unsafe impl DeviceCopy for $T {
            #[cfg(target_arch = "spirv")]
            unsafe fn __data_type<V>(var: *const V) {
                unsafe {
                    krnl_core::__private::[<__data_type_$T>](var);
                }
            }
        }
    }
});

macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    paste! {
        unsafe impl Scalar for $T {
            const SCALAR_TYPE: ScalarType = ScalarType::[<$T:upper>];
        }

        unsafe impl Element for $T {
            type Scalar = Self;
        }
    }
});

macro_for!($N in [1, 2, 4, 8, 16] {
    unsafe impl<T: Scalar> Element for [T; $N] {
        type Scalar = T;
    }
});
