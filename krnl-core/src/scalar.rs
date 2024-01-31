#[cfg(not(target_arch = "spirv"))]
use bytemuck::Pod;
#[cfg(not(target_arch = "spirv"))]
use derive_more::Display;
use dry::macro_for;
#[cfg(not(target_arch = "spirv"))]
use dry::macro_wrap;
use half::{bf16, f16};
use num_traits::{AsPrimitive, FromPrimitive, NumAssign, NumCast};
use paste::paste;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(not(target_arch = "spirv"))]
use std::{
    fmt::{Debug, Display},
    str::FromStr,
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

/// Numerical types supported in **krnl**.
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug, Display))]
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
    #[cfg(not(target_arch = "spirv"))]
    #[inline]
    fn iter() -> impl Iterator<Item = Self> {
        use ScalarType::*;
        [U8, I8, U16, I16, F16, BF16, U32, I32, F32, U64, I64, F64].into_iter()
    }
    /// Size of the type in bytes.
    #[inline]
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
    #[inline]
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
    /// Name of the variant.
    ///
    /// Uppercase, ie "F16", "I32", etc.
    #[cfg(not(target_arch = "spirv"))]
    #[inline]
    pub fn as_str(&self) -> &'static str {
        use ScalarType::*;
        match self {
            U8 => "U8",
            I8 => "I8",
            U16 => "U16",
            I16 => "I16",
            F16 => "F16",
            BF16 => "BF16",
            U32 => "U32",
            I32 => "I32",
            F32 => "F32",
            U64 => "U64",
            I64 => "I64",
            F64 => "F64",
        }
    }
}

impl From<ScalarType> for u32 {
    #[inline]
    fn from(scalar_type: ScalarType) -> u32 {
        scalar_type as u32
    }
}

impl TryFrom<u32> for ScalarType {
    type Error = ();
    fn try_from(input: u32) -> Result<Self, ()> {
        use ScalarType::*;
        let output = match input {
            1 => U8,
            2 => I8,
            3 => U16,
            4 => I16,
            5 => F16,
            6 => BF16,
            7 => U32,
            8 => I32,
            9 => F32,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => {
                return Err(());
            }
        };
        Ok(output)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl FromStr for ScalarType {
    type Err = ();
    fn from_str(input: &str) -> Result<Self, ()> {
        Self::iter()
            .find(|x| x.as_str() == input || x.name() == input)
            .ok_or(())
    }
}

#[cfg(feature = "serde")]
impl Serialize for ScalarType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for ScalarType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Visitor;

        struct ScalarTypeVisitor;

        impl Visitor<'_> for ScalarTypeVisitor {
            type Value = ScalarType;
            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(formatter, "a scalar type")
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if let Ok(scalar_type) = ScalarType::from_str(v) {
                    Ok(scalar_type)
                } else {
                    Err(E::custom(format!("unknown ScalarType {v}")))
                }
            }
        }
        deserializer.deserialize_str(ScalarTypeVisitor)
    }
}

/// Enumeration of all scalars.
#[allow(missing_docs)]
#[cfg(not(target_arch = "spirv"))]
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    /// Zero.
    #[inline]
    pub const fn zero(scalar_type: ScalarType) -> Self {
        use ScalarElem as E;
        use ScalarType as S;
        match scalar_type {
            S::U8 => E::U8(0),
            S::I8 => E::I8(0),
            S::U16 => E::U16(0),
            S::I16 => E::I16(0),
            S::F16 => E::F16(f16::ZERO),
            S::BF16 => E::BF16(bf16::ZERO),
            S::U32 => E::U32(0),
            S::I32 => E::I32(0),
            S::F32 => E::F32(0.),
            S::U64 => E::U64(0),
            S::I64 => E::I64(0),
            S::F64 => E::F64(0.),
        }
    }
    /// One.
    #[inline]
    pub const fn one(scalar_type: ScalarType) -> Self {
        use ScalarElem as E;
        use ScalarType as S;
        match scalar_type {
            S::U8 => E::U8(1),
            S::I8 => E::I8(1),
            S::U16 => E::U16(1),
            S::I16 => E::I16(1),
            S::F16 => E::F16(f16::ONE),
            S::BF16 => E::BF16(bf16::ONE),
            S::U32 => E::U32(1),
            S::I32 => E::I32(1),
            S::F32 => E::F32(1.),
            S::U64 => E::U64(1),
            S::I64 => E::I64(1),
            S::F64 => E::F64(1.),
        }
    }
    /// Casts to `scalar_type`.
    ///
    /// See [`Scalar::cast`].
    #[inline]
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
    /// Casts to `T`.
    ///
    /// See [`Scalar::cast`].
    #[inline]
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
    /// The [`ScalarType`].
    #[inline]
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
    /// The bits of the elem, ie u8, u16, u32, or u64.
    #[inline]
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
    /// The bytes as as slice.
    ///
    /// See [`bytemuck::bytes_of`].
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        use ScalarElem::*;
        macro_wrap!(match self {
            macro_for!($E in [U8, I8, U16, I16, F16, BF16, U32, I32, F32, U64, I64, F64] {
                $E(x) => bytemuck::bytes_of(x),
            })
        })
    }
}

#[cfg(not(target_arch = "spirv"))]
impl<T: Scalar> From<T> for ScalarElem {
    fn from(x: T) -> Self {
        x.scalar_elem()
    }
}

macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    #[cfg(not(target_arch = "spirv"))]
    impl TryFrom<ScalarElem> for $T {
        type Error = ();
        #[inline]
        fn try_from(elem: ScalarElem) -> Result<Self, ()> {
            if Self::SCALAR_TYPE == elem.scalar_type() {
                Ok(elem.cast())
            } else {
                Err(())
            }
        }
    }
});

trait AsScalar<T>: Sized {
    #[allow(clippy::wrong_self_convention)]
    fn as_scalar(self) -> T;
}

// For spirv arch, use min_specialization to strip branches before
// hitting rust-gpu backend, reducing compile times.
#[cfg(target_arch = "spirv")]
impl<X, Y> AsScalar<Y> for X {
    #[rustfmt::skip]
    default fn as_scalar(self) -> Y {
        unreachable!()
    }
}

macro_for!($X in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64] {
    macro_for!($Y in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64] {
        impl AsScalar<$Y> for $X {
            #[inline]
            fn as_scalar(self) -> $Y {
                self as _
            }
        }
    });
});

macro_for!($X in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64] {
    macro_for!($Y in [f16, bf16] {
        impl AsScalar<$Y> for $X {
            #[inline]
            fn as_scalar(self) -> $Y {
                self.as_()
            }
        }
    });
});

macro_for!($X in [f16, bf16] {
    macro_for!($Y in [u8, i8, u16, i16, u32, i32, f32, u64, i64, f64]  {
        impl AsScalar<$Y> for $X {
            #[inline]
            fn as_scalar(self) -> $Y {
                self.as_()
            }
        }
    });
});

macro_for!($X in [f16, bf16] {
    macro_for!($Y in [f16, bf16]  {
        impl AsScalar<$Y> for $X {
            #[inline]
            fn as_scalar(self) -> $Y {
                $Y::from_f32(self.to_f32())
            }
        }
    });
});

#[cfg(target_arch = "spirv")]
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
    + PartialOrd
    + Sealed
{
    /// The [`ScalarType`] of the scalar.
    const SCALAR_TYPE: ScalarType;
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced by Scalar::SCALAR_TYPE")]
    fn scalar_type() -> ScalarType;
    /// Casts `self as T`.
    fn cast<T: Scalar>(self) -> T;
}

#[cfg(all(not(target_arch = "spirv"), not(feature = "serde")))]
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
    + PartialOrd
    + Pod
    + Debug
    + Display
    + Sealed
{
    /// The [`ScalarType`] of the scalar.
    const SCALAR_TYPE: ScalarType;
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced by Scalar::SCALAR_TYPE")]
    fn scalar_type() -> ScalarType;
    /// Converts to [`ScalarElem`].
    fn scalar_elem(self) -> ScalarElem;
    /// Casts `self as T`.
    fn cast<T: Scalar>(self) -> T;
}

#[cfg(all(not(target_arch = "spirv"), feature = "serde"))]
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
    + PartialOrd
    + Pod
    + Debug
    + Display
    + Serialize
    + for<'de> Deserialize<'de>
    + Sealed
{
    /// The [`ScalarType`] of the scalar.
    const SCALAR_TYPE: ScalarType;
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced by Scalar::SCALAR_TYPE")]
    fn scalar_type() -> ScalarType;
    /// Converts to [`ScalarElem`].
    fn scalar_elem(self) -> ScalarElem;
    /// Casts `self as T`.
    fn cast<T: Scalar>(self) -> T;
}

macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    paste! {
        impl Scalar for $X {
            const SCALAR_TYPE: ScalarType = ScalarType::[<$X:upper>];
            #[inline(always)]
            fn scalar_type() -> ScalarType {
                Self::SCALAR_TYPE
            }
            #[cfg(not(target_arch = "spirv"))]
            #[inline(always)]
            fn scalar_elem(self) -> ScalarElem {
                ScalarElem::[<$X:upper>](self)
            }
            #[cfg(not(target_arch = "spirv"))]
            #[inline]
            fn cast<T: Scalar>(self) -> T {
                macro_wrap!(match T::SCALAR_TYPE {
                    macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                        $Y::SCALAR_TYPE => bytemuck::cast(AsScalar::<$Y>::as_scalar(self)),
                    })
                })
            }
            #[cfg(target_arch = "spirv")]
            #[inline]
            fn cast<T: Scalar>(self) -> T {
                AsScalar::<T>::as_scalar(self)
            }
        }
    }
});
