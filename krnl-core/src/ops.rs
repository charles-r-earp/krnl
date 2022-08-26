#[cfg(target_arch = "spirv")]
use core::arch::asm;
use core::ops::{Index, IndexMut};

// adapted from https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-std/src/arch.rs IndexUnchecked
pub trait IndexUnchecked<Idx>: Index<Idx> {
    unsafe fn index_unchecked(&self, index: Idx) -> &Self::Output;
}

// adapted from https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-std/src/arch.rs IndexUnchecked
pub trait IndexUncheckedMut<Idx>: IndexMut<Idx> + IndexUnchecked<Idx> {
    unsafe fn index_unchecked_mut(&mut self, index: Idx) -> &mut Self::Output;
}

// adapted from https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-std/src/arch.rs IndexUnchecked
impl<T> IndexUnchecked<usize> for [T] {
    #[cfg(target_arch = "spirv")]
    unsafe fn index_unchecked(&self, index: usize) -> &T {
        unsafe {
            asm! {
                "%slice_ptr = OpLoad _ {slice_ptr_ptr}",
                "%data_ptr = OpCompositeExtract _ %slice_ptr 0",
                "%val_ptr = OpAccessChain _ %data_ptr {index}",
                "OpReturnValue %val_ptr",
                slice_ptr_ptr = in(reg) &self,
                index = in(reg) index,
                options(noreturn)
            }
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    unsafe fn index_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len());
        unsafe { self.get_unchecked(index) }
    }
}

// adapted from https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-std/src/arch.rs IndexUnchecked
impl<T> IndexUncheckedMut<usize> for [T] {
    #[cfg(target_arch = "spirv")]
    unsafe fn index_unchecked_mut(&mut self, index: usize) -> &mut T {
        unsafe {
            asm! {
                "%slice_ptr = OpLoad _ {slice_ptr_ptr}",
                "%data_ptr = OpCompositeExtract _ %slice_ptr 0",
                "%val_ptr = OpAccessChain _ %data_ptr {index}",
                "OpReturnValue %val_ptr",
                slice_ptr_ptr = in(reg) &self,
                index = in(reg) index,
                options(noreturn)
            }
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    unsafe fn index_unchecked_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len());
        unsafe { self.get_unchecked_mut(index) }
    }
}

// adapted from https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-std/src/arch.rs IndexUnchecked
impl<T, const N: usize> IndexUnchecked<usize> for [T; N] {
    #[cfg(target_arch = "spirv")]
    unsafe fn index_unchecked(&self, index: usize) -> &T {
        unsafe {
            asm! {
                "%val_ptr = OpAccessChain _ {array_ptr} {index}",
                "OpReturnValue %val_ptr",
                array_ptr = in(reg) &self,
                index = in(reg) index,
                options(noreturn)
            }
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    unsafe fn index_unchecked(&self, index: usize) -> &T {
        unsafe { self.as_slice().index_unchecked(index) }
    }
}

// adapted from https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/spirv-std/src/arch.rs IndexUnchecked
impl<T, const N: usize> IndexUncheckedMut<usize> for [T; N] {
    #[cfg(target_arch = "spirv")]
    unsafe fn index_unchecked_mut(&mut self, index: usize) -> &mut T {
        unsafe {
            asm! {
                "%val_ptr = OpAccessChain _ {array_ptr} {index}",
                "OpReturnValue %val_ptr",
                array_ptr = in(reg) self,
                index = in(reg) index,
                options(noreturn)
            }
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    unsafe fn index_unchecked_mut(&mut self, index: usize) -> &mut T {
        unsafe { self.as_mut_slice().index_unchecked_mut(index) }
    }
}
