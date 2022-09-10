#[cfg(not(target_arch = "spirv"))]
use bytemuck::Pod;
#[cfg(not(target_arch = "spirv"))]
use derive_more::Display;
#[cfg(feature = "half")]
use half::{bf16, f16};
use num_traits::{FromPrimitive, NumAssign, NumCast};
#[cfg(not(target_arch = "spirv"))]
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "spirv"))]
use std::{
    borrow::Cow,
    fmt::{Debug, Display},
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

pub mod error {
    #[cfg(not(target_arch = "spirv"))]
    use super::*;

    #[cfg(not(target_arch = "spirv"))]
    #[derive(Debug, thiserror::Error)]
    #[error("ScalarTypeFromStrError: unknown scalar type {input:?}!")]
    pub struct ScalarTypeFromStrError<'a> {
        pub(super) input: Cow<'a, str>,
    }

    #[cfg_attr(
        not(target_arch = "spirv"),
        derive(Debug, thiserror::Error),
        error("ScalarTypeFromU32Error: unknown scalar type {input}!")
    )]
    pub struct ScalarTypeFromU32Error {
        #[cfg(not(target_arch = "spirv"))]
        pub(super) input: u32,
    }
}
use error::*;

/// Numerical types supported in krnl.
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Eq, PartialEq)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, Display, Serialize, Deserialize),
    serde(into = "u32", try_from = "u32")
)]
#[cfg_attr(target_arch = "spirv", repr(u32))]
pub enum ScalarType {
    U8 = 1,
    I8 = 2,
    U16 = 3,
    I16 = 4,
    #[cfg(feature = "half")]
    F16 = 5,
    #[cfg(feature = "half")]
    BF16 = 6,
    U32 = 7,
    I32 = 8,
    F32 = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
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
    /// Lowercase, ie "f16", "i32", etc.
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

impl From<ScalarType> for u32 {
    fn from(scalar_type: ScalarType) -> u32 {
        scalar_type as u32
    }
}

impl TryFrom<u32> for ScalarType {
    type Error = ScalarTypeFromU32Error;
    fn try_from(input: u32) -> Result<Self, Self::Error> {
        use ScalarType::*;
        match input {
            1 => Ok(U8),
            2 => Ok(I8),
            3 => Ok(U16),
            4 => Ok(I16),
            #[cfg(feature = "half")]
            5 => Ok(F16),
            #[cfg(feature = "half")]
            6 => Ok(BF16),
            7 => Ok(U32),
            8 => Ok(I32),
            9 => Ok(F32),
            10 => Ok(U64),
            11 => Ok(I64),
            12 => Ok(F64),
            _ => Err(ScalarTypeFromU32Error {
                #[cfg(not(target_arch = "spirv"))]
                input,
            }),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl<'a> TryFrom<&'a str> for ScalarType {
    type Error = ScalarTypeFromStrError<'a>;
    fn try_from(input: &'a str) -> Result<Self, Self::Error> {
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
                input: input.into(),
            }),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
#[derive(Clone, Copy, PartialEq, Debug)]
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
        use ScalarElem::*;
        use ScalarType as T;
        match self {
            U8(_) => T::U8,
            I8(_) => T::I8,
            U16(_) => T::U16,
            I16(_) => T::I16,
            #[cfg(feature = "half")]
            F16(_) => T::F16,
            #[cfg(feature = "half")]
            BF16(_) => T::BF16,
            U32(_) => T::U32,
            I32(_) => T::I32,
            F32(_) => T::F32,
            U64(_) => T::U64,
            I64(_) => T::I64,
            F64(_) => T::F64,
        }
    }
    pub fn to_scalar_bits(&self) -> Self {
        use ScalarElem::*;
        match self {
            U8(_) => *self,
            I8(x) => (*x as u8).into(),
            U16(_) => *self,
            I16(x) => (*x as u16).into(),
            #[cfg(feature = "half")]
            F16(x) => x.to_bits().into(),
            #[cfg(feature = "half")]
            BF16(x) => x.to_bits().into(),
            U32(_) => *self,
            I32(x) => (*x as u32).into(),
            F32(x) => x.to_bits().into(),
            U64(_) => *self,
            I64(x) => (*x as u64).into(),
            F64(x) => x.to_bits().into(),
        }
    }
    /*
    pub fn bitcast(&self, scalar_type: ScalarType) -> Self {
        use ScalarType as T;
        use ScalarElem::*;
        fn bcs<X: Scalar>(x: &X, s: ScalarType) -> ScalarElem {
            fn bc<X: Scalar, Y: Scalar>(x: &X) -> Y {
                *bytemuck::cast_ref(x)
            }
            match s {
                T::U8 => U8(bc(x)),
                T::I8 => I8(bc(x)),
                T::U16 => U16(bc(x)),
                T::I16 => I16(bc(x)),
                #[cfg(feature = "half")]
                T::F16 => F16(bc(x)),
                #[cfg(feature = "half")]
                T::BF16 => BF16(bc(x)),
                T::U32 => U32(bc(x)),
                T::I32 => I32(bc(x)),
                T::F32 => F32(bc(x)),
                T::U64 => U64(bc(x)),
                T::I64 => I64(bc(x)),
                T::F64 => F64(bc(x)),
            }
        }
        let s = scalar_type;
        match self {
            U8(x) => bcs(x, s),
            U8(x) => bcs(x, s),
            I8(x) => bcs(x, s),
            U16(x) => bcs(x, s),
            I16(x) => bcs(x, s),
            #[cfg(feature = "half")]
            F16(x) => bcs(x, s),
            #[cfg(feature = "half")]
            BF16(x) => bcs(x, s),
            U32(x) => bcs(x, s),
            I32(x) => bcs(x, s),
            F32(x) => bcs(x, s),
            U64(x) => bcs(x, s),
            I64(x) => bcs(x, s),
            F64(x) => bcs(x, s),
        }
    }*/
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
    + FromPrimitive
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
pub trait Scalar:
    Default + Copy + 'static + NumCast + FromPrimitive + NumAssign + PartialEq + Sealed
{
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
