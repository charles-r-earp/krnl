#[cfg(not(target_arch = "spirv"))]
use core::marker::PhantomData;
#[cfg(target_arch = "spirv")]
use core::ops::Index;
use glam::{UVec2, UVec3, Vec3Swizzles};
#[cfg(target_arch = "spirv")]
use spirv_std::arch::IndexUnchecked;

#[derive(Copy, Clone)]
pub struct KernelArgs {
    pub groups: UVec3,
    pub group_id: UVec3,
    pub threads: UVec3,
    pub thread_id: UVec3,
    pub items: u32,
}

impl KernelArgs {
    fn kernel1(self) -> Kernel<u32> {
        let Kernel {
            global_threads,
            global_id,
            global_index,
            groups,
            group_id,
            group_index,
            threads,
            thread_id,
            thread_index,
            items,
            item_id,
            item_stride,
        } = self.kernel3();
        Kernel {
            global_threads: global_threads.x,
            global_id: global_id.x,
            global_index,
            groups: groups.x,
            group_id: group_id.x,
            group_index,
            threads: threads.x,
            thread_id: thread_id.x,
            thread_index,
            items,
            item_id,
            item_stride,
        }
    }
    fn kernel2(self) -> Kernel<UVec2> {
        let Kernel {
            global_threads,
            global_id,
            global_index,
            groups,
            group_id,
            group_index,
            threads,
            thread_id,
            thread_index,
            items,
            item_id,
            item_stride,
        } = self.kernel3();
        Kernel {
            global_threads: global_threads.xy(),
            global_id: global_id.xy(),
            global_index,
            groups: groups.xy(),
            group_id: group_id.xy(),
            group_index,
            threads: threads.xy(),
            thread_id: thread_id.xy(),
            thread_index,
            items,
            item_id,
            item_stride,
        }
    }
    fn kernel3(self) -> Kernel<UVec3> {
        let Self {
            groups,
            group_id,
            threads,
            thread_id,
            items,
        } = self;
        let global_threads = groups * threads;
        let global_id = group_id * threads + thread_id;
        let global_index = global_id.x
            + global_id.y * global_threads.x
            + global_id.z * global_threads.x * global_threads.y;
        let group_index = group_id.x + group_id.y * groups.x + group_id.z * groups.x * groups.y;
        let thread_index =
            thread_id.x + thread_id.y * threads.x + thread_id.z * threads.x * threads.y;
        let item_stride = global_threads.x * global_threads.y * global_threads.z;
        Kernel {
            global_threads,
            global_id,
            global_index,
            groups,
            group_id,
            group_index,
            threads,
            thread_id,
            thread_index,
            items,
            item_id: global_index,
            item_stride,
        }
    }
}

pub struct Kernel<D> {
    global_threads: D,
    global_id: D,
    global_index: u32,
    groups: D,
    group_id: D,
    group_index: u32,
    threads: D,
    thread_id: D,
    thread_index: u32,
    items: u32,
    item_id: u32,
    item_stride: u32,
}

impl<D: Copy> Kernel<D> {
    pub fn global_threads(&self) -> D {
        self.global_threads
    }
    pub fn global_id(&self) -> D {
        self.global_id
    }
    pub fn global_index(&self) -> u32 {
        self.global_index
    }
    pub fn groups(&self) -> D {
        self.groups
    }
    pub fn group_id(&self) -> D {
        self.group_id
    }
    pub fn group_index(&self) -> u32 {
        self.group_index
    }
    pub fn threads(&self) -> D {
        self.threads
    }
    pub fn thread_id(&self) -> D {
        self.thread_id
    }
    pub fn thread_index(&self) -> u32 {
        self.thread_index
    }
    pub fn items(&self) -> u32 {
        self.items
    }
    pub fn item_id(&self) -> u32 {
        self.item_id
    }
    pub fn next_item(&mut self) {
        self.item_id += self.item_stride;
    }
}

impl From<KernelArgs> for Kernel<u32> {
    fn from(args: KernelArgs) -> Self {
        args.kernel1()
    }
}

impl From<KernelArgs> for Kernel<UVec2> {
    fn from(args: KernelArgs) -> Self {
        args.kernel2()
    }
}

impl From<KernelArgs> for Kernel<UVec3> {
    fn from(args: KernelArgs) -> Self {
        args.kernel3()
    }
}

pub trait Length {
    fn len(&self) -> usize;
}

impl<T> Length for [T] {
    fn len(&self) -> usize {
        Self::len(self)
    }
}

pub trait UnsafeIndexMut<I> {
    type Output;
    unsafe fn unsafe_index_mut(&self, index: I) -> &mut Self::Output;
}

#[cfg(target_arch = "spirv")]
fn debug_index_out_of_bounds(index: usize, len: usize) {
    unsafe {
        spirv_std::macros::debug_printfln!(
            "index out of bounds: the len is %u but the index is %u",
            len as u32,
            index as u32,
        );
    }
}

#[cfg(target_arch = "spirv")]
trait IndexUncheckedMutExt<T> {
    unsafe fn index_unchecked_mut_ext(&self, index: usize) -> &mut T;
}

#[cfg(target_arch = "spirv")]
impl<T> IndexUncheckedMutExt<T> for [T] {
    unsafe fn index_unchecked_mut_ext(&self, index: usize) -> &mut T {
        // https://docs.rs/spirv-std/0.5.0/src/spirv_std/arch.rs.html#237-248
        unsafe {
            ::core::arch::asm!(
                "%slice_ptr = OpLoad _ {slice_ptr_ptr}",
                "%data_ptr = OpCompositeExtract _ %slice_ptr 0",
                "%val_ptr = OpAccessChain _ %data_ptr {index}",
                "OpReturnValue %val_ptr",
                slice_ptr_ptr = in(reg) &self,
                index = in(reg) index,
                options(noreturn)
            )
        }
    }
}

#[doc(hidden)]
#[cfg(target_arch = "spirv")]
pub struct Slice<'a, T> {
    inner: &'a [T],
    offset: usize,
    len: usize,
}

#[cfg(target_arch = "spirv")]
impl<'a, T> Slice<'a, T> {
    pub unsafe fn from_raw_parts(inner: &'a [T], offset: usize, len: usize) -> Self {
        Self { inner, offset, len }
    }
}

#[cfg(target_arch = "spirv")]
impl<T> Length for Slice<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(target_arch = "spirv")]
impl<T> Index<usize> for Slice<'_, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        if index < self.len {
            unsafe { self.inner.index_unchecked(self.offset + index) }
        } else {
            debug_index_out_of_bounds(index, self.len);
            panic!();
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
#[derive(Copy, Clone)]
pub struct UnsafeSliceMut<'a, T> {
    ptr: *mut T,
    len: usize,
    _m: PhantomData<&'a ()>,
}

#[cfg(target_arch = "spirv")]
pub struct UnsafeSliceMut<'a, T> {
    inner: &'a mut [T],
    offset: usize,
    len: usize,
}

#[cfg(not(target_ach = "spirv"))]
unsafe impl<T: Send> Send for UnsafeSliceMut<'_, T> {}
#[cfg(not(target_ach = "spirv"))]
unsafe impl<T: Sync> Sync for UnsafeSliceMut<'_, T> {}

#[cfg(not(target_arch = "spirv"))]
impl<'a, T> From<&'a mut [T]> for UnsafeSliceMut<'a, T> {
    fn from(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _m: PhantomData::default(),
        }
    }
}

#[cfg(target_arch = "spirv")]
impl<'a, T> UnsafeSliceMut<'a, T> {
    pub unsafe fn from_raw_parts(inner: &'a mut [T], offset: usize, len: usize) -> Self {
        Self { inner, offset, len }
    }
}

impl<T> Length for UnsafeSliceMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> UnsafeIndexMut<usize> for UnsafeSliceMut<'_, T> {
    type Output = T;
    #[cfg(not(target_arch = "spirv"))]
    unsafe fn unsafe_index_mut(&self, index: usize) -> &mut Self::Output {
        if index < self.len {
            unsafe { &mut *self.ptr.add(index) }
        } else {
            panic!(
                "index out of bounds: the len is {index} but the index is {len}",
                len = self.len
            );
        }
    }
    #[cfg(target_arch = "spirv")]
    unsafe fn unsafe_index_mut(&self, index: usize) -> &mut Self::Output {
        if index < self.len {
            unsafe { self.inner.index_unchecked_mut_ext(self.offset + index) }
        } else {
            debug_index_out_of_bounds(index, self.len);
            panic!();
        }
    }
}
