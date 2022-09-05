use serde::{Serialize, Deserialize};
use std::{str::FromStr, borrow::Cow};

pub mod error {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("VersionFromStrError: expected \"major[.minor.patch]\", found {:?}!", .input)]
    pub struct VersionFromStrError {
        pub(super) input: Cow<'static, str>,
    }
}
use error::*;

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, Serialize, Deserialize, Debug, derive_more::Display)]
#[display(fmt="{}.{}.{}", major, minor, patch)]
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

impl FromStr for Version {
    type Err = VersionFromStrError;
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let mut iter = input.split(".");
        let major = if let Some(x) = iter.next() {
            if let Ok(x) = u32::from_str(x) {
                x
            } else {
                return Err(VersionFromStrError {
                    input: input.to_string().into(),
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
                    input: input.to_string().into(),
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
                    input: input.to_string().into(),
                });
            }
        } else {
            0
        };
        if iter.next().is_some() {
            return Err(VersionFromStrError {
                input: input.to_string().into(),
            });
        }
        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}
