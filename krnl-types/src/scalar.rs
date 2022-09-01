#[cfg(not(target_arch = "spirv"))]
use bytemuck::Pod;
#[cfg(not(target_arch = "spirv"))]
use derive_more::Display;
#[cfg(feature = "half")]
use half::{bf16, f16};
use num_traits::{NumAssign, NumCast};
#[cfg(not(target_arch = "spirv"))]
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "spirv"))]
use std::{
    fmt::{Debug, Display},
    str::FromStr,
};

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

#[cfg(not(target_arch = "spirv"))]
pub mod error {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("unknown ScalarType `{}`", .input)]
    pub struct ScalarTypeFromStrError {
        pub(super) input: String,
    }
}
#[cfg(not(target_arch = "spirv"))]
use error::*;

/// Numerical types supported in krnl.
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Eq, PartialEq)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, Display, Serialize, Deserialize)
)]
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

#[cfg(not(target_arch = "spirv"))]
impl FromStr for ScalarType {
    type Err = ScalarTypeFromStrError;
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        use ScalarType::*;
        match input {
            "U8" | "u8" => Ok(U8),
            "I8" | "i8" => Ok(I8),
            "U16" | "u16" => Ok(U16),
            "I16" | "i16" => Ok(I16),
            #[cfg(feature = "half")]
            "F16" | "f16" => Ok(F16),
            #[cfg(feature = "half")]
            "BF16" | "bf16" => Ok(BF16),
            "U32" | "u32" => Ok(U32),
            "I32" | "i32" => Ok(I32),
            "F32" | "f32" => Ok(F32),
            "U64" | "u64" => Ok(U64),
            "I64" | "i64" => Ok(I64),
            "F64" | "f64" => Ok(F64),
            _ => Err(ScalarTypeFromStrError {
                input: input.to_string(),
            }),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum ScalarElem {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    #[cfg(feature = "half")]
    F16(f16),
    #[cfg(feature = "half")]
    BF16(bf16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[cfg(not(target_arch = "spirv"))]
impl ScalarElem {
    pub fn scalar_type(&self) -> ScalarType {
        use ScalarType as T;
        use ScalarElem::*;
        match self {
            U8(_) => T::U8,
            I8(_) => T::I8,
            /*U16(u16),
            I16(i16),
            #[cfg(feature = "half")]
            F16(f16),
            #[cfg(feature = "half")]
            BF16(bf16),*/
            U32(_) => T::U32,
            /*I32(i32),
            F32(f32),
            U64(u64),
            I64(i64),
            F64(f64),*/
            _ => todo!()
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<u8> for ScalarElem {
    fn from(x: u8) -> Self {
        Self::U8(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<i8> for ScalarElem {
    fn from(x: i8) -> Self {
        Self::I8(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<u16> for ScalarElem {
    fn from(x: u16) -> Self {
        Self::U16(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<i16> for ScalarElem {
    fn from(x: i16) -> Self {
        Self::I16(x)
    }
}

#[cfg(all(not(target_arch = "spirv"), feature = "half"))]
impl From<f16> for ScalarElem {
    fn from(x: f16) -> Self {
        Self::F16(x)
    }
}

#[cfg(all(not(target_arch = "spirv"), feature = "half"))]
impl From<bf16> for ScalarElem {
    fn from(x: bf16) -> Self {
        Self::BF16(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<u32> for ScalarElem {
    fn from(x: u32) -> Self {
        Self::U32(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<i32> for ScalarElem {
    fn from(x: i32) -> Self {
        Self::I32(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<f32> for ScalarElem {
    fn from(x: f32) -> Self {
        Self::F32(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<u64> for ScalarElem {
    fn from(x: u64) -> Self {
        Self::U64(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<i64> for ScalarElem {
    fn from(x: i64) -> Self {
        Self::I64(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl From<f64> for ScalarElem {
    fn from(x: f64) -> Self {
        Self::F64(x)
    }
}

/// Base trait for numerical types.
#[cfg(not(target_arch = "spirv"))]
pub trait Scalar:
    Default
    + Copy
    + 'static
    + Into<ScalarElem>
    + NumCast
    + NumAssign
    + Pod
    + Debug
    + Display
    + Serialize
    + for<'de> Deserialize<'de>
    + PartialEq
    + Sealed
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
pub trait Scalar: Default + Copy + 'static + NumCast + NumAssign + PartialEq + Sealed {
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
