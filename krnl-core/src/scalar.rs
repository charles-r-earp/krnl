#[cfg(not(target_arch = "spirv"))]
use bytemuck::Pod;
#[cfg(not(target_arch = "spirv"))]
use derive_more::Display;
use dry::macro_for;
use half::{bf16, f16};
use num_traits::{AsPrimitive, FromPrimitive, NumAssign, NumCast};
use paste::paste;
#[cfg(not(target_arch = "spirv"))]
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "spirv"))]
use std::{
    borrow::Cow,
    fmt::{Debug, Display},
};

mod sealed {
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

    impl_sealed!(u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64);
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
    F16 = 5,
    BF16 = 6,
    U32 = 7,
    I32 = 8,
    F32 = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ScalarType {
    /*pub fn iter() -> impl Iterator<Item = Self> {
        use ScalarType::*;
        fn half_types() -> impl Iterator<Item = ScalarType> {
            #[cfg(feature = "half")]
            {
                return [F16, BF16].into_iter();
            }
            [].into_iter()
        }
        [U8, I8, U16, I16]
            .into_iter()
            .chain(half_types())
            .chain([U32, I32, F32, U64, I64, F64])
    }*/
    /// Size of the type in bytes.
    pub fn size(&self) -> usize {
        use ScalarType::*;
        match self {
            U8 | I8 => 1,
            U16 | I16 | F16 | BF16 => 2,
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
            F16 => "f16",
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
            5 => Ok(F16),
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
            "F16" | "f16" => Ok(F16),
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
    F16(f16),
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
    pub fn zero(scalar_type: ScalarType) -> Self {
        Self::U8(1).scalar_cast(scalar_type)
    }
    pub fn one(scalar_type: ScalarType) -> Self {
        Self::U8(1).scalar_cast(scalar_type)
    }
    pub fn scalar_cast(self, scalar_type: ScalarType) -> Self {
        use ScalarElem as E;
        use ScalarType as S;
        let x = self;
        match scalar_type {
            S::U8 => E::U8(x.cast()),
            S::I8 => E::I8(x.cast()),
            S::U16 => E::U16(x.cast()),
            S::I16 => E::I16(x.cast()),
            S::F16 => E::F16(x.cast()),
            S::BF16 => E::BF16(x.cast()),
            S::U32 => E::U32(x.cast()),
            S::I32 => E::I32(x.cast()),
            S::F32 => E::F32(x.cast()),
            S::U64 => E::U64(x.cast()),
            S::I64 => E::I64(x.cast()),
            S::F64 => E::F64(x.cast()),
        }
    }
    pub fn cast<T: Scalar>(self) -> T {
        use ScalarElem::*;
        match self {
            U8(x) => x.cast(),
            I8(x) => x.cast(),
            U16(x) => x.cast(),
            I16(x) => x.cast(),
            F16(x) => x.cast(),
            BF16(x) => x.cast(),
            U32(x) => x.cast(),
            I32(x) => x.cast(),
            F32(x) => x.cast(),
            U64(x) => x.cast(),
            I64(x) => x.cast(),
            F64(x) => x.cast(),
        }
    }
    pub fn scalar_type(&self) -> ScalarType {
        use ScalarElem::*;
        use ScalarType as T;
        match self {
            U8(_) => T::U8,
            I8(_) => T::I8,
            U16(_) => T::U16,
            I16(_) => T::I16,
            F16(_) => T::F16,
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
            F16(x) => x.to_bits().into(),
            BF16(x) => x.to_bits().into(),
            U32(_) => *self,
            I32(x) => (*x as u32).into(),
            F32(x) => x.to_bits().into(),
            U64(_) => *self,
            I64(x) => (*x as u64).into(),
            F64(x) => x.to_bits().into(),
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

#[cfg(not(target_arch = "spirv"))]
impl From<f16> for ScalarElem {
    fn from(x: f16) -> Self {
        Self::F16(x)
    }
}

#[cfg(not(target_arch = "spirv"))]
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

trait AsScalar<T>: Scalar {
    fn as_scalar(self) -> T;
}

macro_for!($X in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64] {
    macro_for!($Y in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64] {
        impl AsScalar<$Y> for $X {
            fn as_scalar(self) -> $Y {
                self as _
            }
        }
    });
});

macro_for!($X in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64] {
    macro_for!($Y in [f16, bf16] {
        impl AsScalar<$Y> for $X {
            fn as_scalar(self) -> $Y {
                self.as_()
            }
        }
    });
});

macro_for!($X in [f16, bf16] {
    macro_for!($Y in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64]  {
        impl AsScalar<$Y> for $X {
            fn as_scalar(self) -> $Y {
                self.as_()
            }
        }
    });
});

macro_for!($X in [f16, bf16] {
    macro_for!($Y in [f16, bf16]  {
        impl AsScalar<$Y> for $X {
            fn as_scalar(self) -> $Y {
                $Y::from_f32(self.to_f32())
            }
        }
    });
});
/*
trait TryAs<T> {
    fn try_as(self) -> Option<T>;
}

impl<Y: Scalar, X: AsScalar<Y>> TryAs<Y> for X {
    fn try_as(self) -> Option<Y> {
        if X::scalar_type() == Y::scalar_type() {
            Some(self.as_scalar())
        } else {
            None
        }
    }
}*/

#[cfg(target_arch = "spirv")]
/// Base trait for numerical types.
pub trait Scalar:
    Default + Copy + 'static + Send + Sync + NumCast + FromPrimitive + NumAssign + PartialEq + Sealed
{
    /// The [`ScalarType`] of the scalar.
    fn scalar_type() -> ScalarType;
    fn cast<T: Scalar>(self) -> T;
}
/*
macro_rules! cast {
    ($self:ident $(:$x:ty)? => $T:ident as $($y:ty),*) => {
        $(
            if <$T>::scalar_type() == <$y>::scalar_type() {
                return NumCast::from(AsPrimitive::<$y>::as_($self)).unwrap();
            }
        )*
    };
}*/

#[cfg(not(target_arch = "spirv"))]
/// Base trait for numerical types.
pub trait Scalar:
    Default
    + Copy
    + 'static
    + Send
    + Sync
    + NumCast
    + FromPrimitive
    + NumAssign
    + PartialEq
    + Into<ScalarElem>
    + Pod
    + Debug
    + Display
    + Serialize
    + for<'de> Deserialize<'de>
    + Sealed
{
    /// The [`ScalarType`] of the scalar.
    fn scalar_type() -> ScalarType;
    fn cast<T: Scalar>(self) -> T;
}

macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    paste! {
        impl Scalar for $X {
            fn scalar_type() -> ScalarType {
                ScalarType::[<$X:upper>]
            }
            fn cast<T: Scalar>(self) -> T {
                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    if T::scalar_type() == $Y::scalar_type() {
                        let y: $Y = self.as_scalar();
                        return NumCast::from(y).unwrap();
                    }
                });
                unreachable!()
            }
        }
    }
});
/*
impl Scalar for u8 {
    fn scalar_type() -> ScalarType {
        ScalarType::U8
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
        {
            cast!(self: u8 => T as u8, i8, u32, i32, f32);
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
            {
                cast!(self: u8 => T as u16, i16);
                #[cfg(feature = "half")]
                {
                    cast!(self: u8 => T as f16, bf16);
                }
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
            {
                cast!(self: u8 => T as u64, i64);
                #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
                {
                    cast!(self: u8 => T as f64);
                }
            }
        }
        unreachable!()
    }
}

impl Scalar for i8 {
    fn scalar_type() -> ScalarType {
        ScalarType::I8
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
        {
            cast!(self: i8 => T as u8, i8, u32, i32, f32);
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
            {
                cast!(self: i8 => T as u16, i16);
                #[cfg(feature = "half")]
                {
                    cast!(self: i8 => T as f16, bf16);
                }
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
            {
                cast!(self: i8 => T as u64, i64);
                #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
                {
                    cast!(self: i8 => T as f64);
                }
            }
        }
        unreachable!()
    }
}

impl Scalar for u16 {
    fn scalar_type() -> ScalarType {
        ScalarType::U16
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
        {
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
            {
                cast!(self: u16 => T as u8, i8);
            }
            cast!(self: u16 => T as u16, i16, u32, i32, f32);
            #[cfg(feature = "half")]
            {
                cast!(self: u16 => T as f16, bf16);
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
            {
                cast!(self: u16 => T as u64, i64);
                #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
                {
                    cast!(self: u16 => T as f64);
                }
            }
        }
        unreachable!()
    }
}

impl Scalar for i16 {
    fn scalar_type() -> ScalarType {
        ScalarType::I16
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
        {
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
            {
                cast!(self: i16 => T as u8, i8);
            }
            cast!(self: i16 => T as u16, i16, u32, i32, f32);
            #[cfg(feature = "half")]
            {
                cast!(self: i16 => T as f16, bf16);
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
            {
                cast!(self: i16 => T as u64, i64);
                #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
                {
                    cast!(self: i16 => T as f64);
                }
            }
        }
        unreachable!()
    }
}

impl Scalar for f16 {
    fn scalar_type() -> ScalarType {
        ScalarType::F16
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
        {
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
            {
                cast!(self: f16 => T as u8, i8);
            }
            cast!(self: f16 => T as u16, i16, f16, f32, u32, i32, f32);
            if T::scalar_type() == ScalarType::BF16 {
                return NumCast::from(bf16::from_f32(self.as_())).unwrap();
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
            {
                cast!(self: u16 => T as u64, i64);
                #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
                {
                    cast!(self: u16 => T as f64);
                }
            }
        }
        unreachable!()
    }
}

impl Scalar for bf16 {
    fn scalar_type() -> ScalarType {
        ScalarType::BF16
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
        {
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
            {
                cast!(self: f16 => T as u8, i8);
            }
            cast!(self: f16 => T as u16, i16, bf16, f32, u32, i32, f32);
            if T::scalar_type() == ScalarType::F16 {
                return NumCast::from(f16::from_f32(self.as_())).unwrap();
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
            {
                cast!(self: u16 => T as u64, i64);
                #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
                {
                    cast!(self: u16 => T as f64);
                }
            }
        }
        unreachable!()
    }
}

impl Scalar for u32 {
    fn scalar_type() -> ScalarType {
        ScalarType::U32
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
        {
            cast!(self => T as u8, i8);
        }
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
        {
            cast!(self => T as u16, i16);
        }
        #[cfg_attr(
            target_arch = "spirv",
            cfg(all(target_feature = "Int8", target_feature = "Int16"))
        )]
        {
            #[cfg(feature = "half")]
            {
                cast!(self => T as f16, bf16);
            }
        }
        cast!(self => T as u32, i32, f32);
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
        {
            cast!(self => T as u64, i64);
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
            {
                cast!(self => T as f64);
            }
        }
        unreachable!()
    }
}

impl Scalar for i32 {
    fn scalar_type() -> ScalarType {
        ScalarType::I32
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
        {
            cast!(self => T as u8, i8);
        }
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
        {
            cast!(self => T as u16, i16);
        }
        #[cfg_attr(
            target_arch = "spirv",
            cfg(all(target_feature = "Int8", target_feature = "Int16"))
        )]
        {
            #[cfg(feature = "half")]
            {
                cast!(self => T as f16, bf16);
            }
        }
        cast!(self => T as u32, i32, f32);
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
        {
            cast!(self => T as u64, i64);
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
            {
                cast!(self => T as f64);
            }
        }
        unreachable!()
    }
}

impl Scalar for f32 {
    fn scalar_type() -> ScalarType {
        ScalarType::F32
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
        {
            cast!(self => T as u8, i8);
        }
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
        {
            cast!(self => T as u16, i16);
        }
        #[cfg_attr(
            target_arch = "spirv",
            cfg(all(target_feature = "Int8", target_feature = "Int16"))
        )]
        {
            #[cfg(feature = "half")]
            {
                cast!(self => T as f16, bf16);
            }
        }
        cast!(self => T as u32, i32, f32);
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
        {
            cast!(self => T as u64, i64);
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
            {
                cast!(self => T as f64);
            }
        }
        unreachable!()
    }
}

impl Scalar for u64 {
    fn scalar_type() -> ScalarType {
        ScalarType::U64
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
        {
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
            {
                cast!(self: u64 => T as u8, i8);
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
            {
                cast!(self: u64 => T as u16, i16);
            }
            #[cfg_attr(
                target_arch = "spirv",
                cfg(all(target_feature = "Int8", target_feature = "Int16"))
            )]
            {
                #[cfg(feature = "half")]
                {
                    cast!(self => T as f16, bf16);
                }
            }
            cast!(self: u64 => T as u32, i32, f32, u64, i64);
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
            {
                cast!(self => T as f64);
            }
        }
        unreachable!()
    }
}

impl Scalar for i64 {
    fn scalar_type() -> ScalarType {
        ScalarType::I64
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int64"))]
        {
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
            {
                cast!(self: u64 => T as u8, i8);
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
            {
                cast!(self: u64 => T as u16, i16);
            }
            #[cfg_attr(
                target_arch = "spirv",
                cfg(all(target_feature = "Int8", target_feature = "Int16"))
            )]
            {
                #[cfg(feature = "half")]
                {
                    cast!(self => T as f16, bf16);
                }
            }
            cast!(self => T as u32, i32, f32, u64, i64);
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Float64"))]
            {
                cast!(self => T as f64);
            }
        }
        unreachable!()
    }
}

impl Scalar for f64 {
    fn scalar_type() -> ScalarType {
        ScalarType::F64
    }
    fn cast<T: Scalar>(self) -> T {
        #[cfg_attr(
            target_arch = "spirv",
            cfg(all(target_feature = "Int64", target_feature = "Float64"))
        )]
        {
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int8"))]
            {
                cast!(self => T as u8, i8);
            }
            #[cfg_attr(target_arch = "spirv", cfg(target_feature = "Int16"))]
            {
                cast!(self => T as u16, i16);
            }
            #[cfg_attr(
                target_arch = "spirv",
                cfg(all(target_feature = "Int8", target_feature = "Int16"))
            )]
            {
                #[cfg(feature = "half")]
                {
                    cast!(self => T as f16, bf16);
                }
            }
            cast!(self => T as u32, i32, f32, u64, i64, f64);
        }
        unreachable!()
    }
}*/
