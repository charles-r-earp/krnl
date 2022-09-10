use serde::{Deserialize, Serialize};
use std::{borrow::Cow, convert::TryFrom, str::FromStr};

pub mod error {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("VersionFromStrError: expected \"major[.minor.patch]\", found {:?}!", .input)]
    pub struct VersionFromStrError<'a> {
        pub(super) input: Cow<'a, str>,
    }
}
use error::*;

#[derive(
    Clone,
    Copy,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Debug,
    derive_more::Display,
)]
#[display(fmt = "{}.{}.{}", major, minor, patch)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn from_major_minor(major: u32, minor: u32) -> Self {
        Self {
            major,
            minor,
            patch: 0,
        }
    }
}

impl<'a> TryFrom<&'a str> for Version {
    type Error = VersionFromStrError<'a>;
    fn try_from(input: &'a str) -> Result<Self, Self::Error> {
        let mut iter = input.split(".");
        let major = if let Some(x) = iter.next() {
            if let Ok(x) = u32::from_str(x) {
                x
            } else {
                return Err(VersionFromStrError {
                    input: input.into(),
                });
            }
        } else {
            0
        };
        let minor = if let Some(x) = iter.next() {
            if let Ok(x) = u32::from_str(x) {
                x
            } else {
                return Err(VersionFromStrError {
                    input: input.into(),
                });
            }
        } else {
            0
        };
        let patch = if let Some(x) = iter.next() {
            if let Ok(x) = u32::from_str(x) {
                x
            } else {
                return Err(VersionFromStrError {
                    input: input.into(),
                });
            }
        } else {
            0
        };
        if iter.next().is_some() {
            return Err(VersionFromStrError {
                input: input.into(),
            });
        }
        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}
