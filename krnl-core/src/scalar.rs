#[cfg(not(target_arch = "spirv"))]
use bytemuck::Pod;
#[cfg(not(target_arch = "spirv"))]
use derive_more::Display;
#[cfg(feature = "half")]
use half::{bf16, f16};
use num_traits::{NumAssign, NumCast};
#[cfg(not(target_arch = "spirv"))]
use std::fmt::{Debug, Display};
#[cfg(not(target_arch = "spirv"))]
use serde::{Serialize, Deserialize};

mod sealed {
    #[cfg(feature = "half")]
    use half::{bf16, f16};

    #[doc(hidden)]
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($t:ty),+) => {
            $(
                impl Sealed for $t {}
            )+
        };
    }

    impl_sealed! {u8, i8, u16, i16, u32, i32, f32, u64, i64, f64}
    #[cfg(feature = "half")]
    impl_sealed! {f16, bf16}
}
use sealed::Sealed;

/// Numerical types supported in autograph.
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Eq, PartialEq)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug, Display, Serialize, Deserialize))]
#[cfg_attr(target_arch = "spirv", repr(u32))]
pub enum ScalarType {
    U8,
    I8,
    U16,
    I16,
    #[cfg(feature = "half")]
    F16,
    #[cfg(feature = "half")]
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl ScalarType {
    /// Size of the type in bytes.
    pub fn size(&self) -> usize {
        use ScalarType::*;
        match self {
            U8 | I8 => 1,
            U16 | I16 => 2,
            #[cfg(feature = "half")]
            F16 | BF16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }
    /// Name of the type.
    ///
    /// Lowercase, ie "f32", "f16", etc.
    #[cfg(not(target_arch = "spirv"))]
    pub fn name(&self) -> &'static str {
        use ScalarType::*;
        match self {
            U8 => "u8",
            I8 => "i8",
            U16 => "u16",
            I16 => "i16",
            #[cfg(feature = "half")]
            F16 => "f16",
            #[cfg(feature = "half")]
            BF16 => "bf16",
            U32 => "u32",
            I32 => "i32",
            F32 => "f32",
            U64 => "u64",
            I64 => "i64",
            F64 => "f64",
        }
    }
}

/// Base trait for numerical types.
#[cfg(not(target_arch = "spirv"))]
pub trait Scalar:
    Default + NumCast + NumAssign + Pod + Debug + Display + Serialize + for<'de> Deserialize<'de> + PartialEq + Sealed
{
    /// The [`ScalarType`] of the scalar.
    fn scalar_type() -> ScalarType;
    /// Name of the type.
    ///
    /// See [`ScalarType::name()`].
    fn scalar_name() -> &'static str {
        Self::scalar_type().name()
    }
}

#[cfg(target_arch = "spirv")]
pub trait Scalar: Default + NumCast + NumAssign + PartialEq + Sealed {
    /// The [`ScalarType`] of the scalar.
    fn scalar_type() -> ScalarType;
}

impl Scalar for u8 {
    fn scalar_type() -> ScalarType {
        ScalarType::U8
    }
}

impl Scalar for i8 {
    fn scalar_type() -> ScalarType {
        ScalarType::I8
    }
}

impl Scalar for u16 {
    fn scalar_type() -> ScalarType {
        ScalarType::U16
    }
}

impl Scalar for i16 {
    fn scalar_type() -> ScalarType {
        ScalarType::I16
    }
}

#[cfg(feature = "half")]
impl Scalar for f16 {
    fn scalar_type() -> ScalarType {
        ScalarType::F16
    }
}

#[cfg(feature = "half")]
impl Scalar for bf16 {
    fn scalar_type() -> ScalarType {
        ScalarType::BF16
    }
}

impl Scalar for u32 {
    fn scalar_type() -> ScalarType {
        ScalarType::U32
    }
}

impl Scalar for i32 {
    fn scalar_type() -> ScalarType {
        ScalarType::I32
    }
}

impl Scalar for f32 {
    fn scalar_type() -> ScalarType {
        ScalarType::F32
    }
}

impl Scalar for u64 {
    fn scalar_type() -> ScalarType {
        ScalarType::U64
    }
}

impl Scalar for i64 {
    fn scalar_type() -> ScalarType {
        ScalarType::I64
    }
}

impl Scalar for f64 {
    fn scalar_type() -> ScalarType {
        ScalarType::F64
    }
}
