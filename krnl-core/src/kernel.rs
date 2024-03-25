#[doc(hidden)]
#[cfg(target_arch = "spirv")]
pub mod __private {
    use super::{ItemKernel, Kernel};
    use core::mem::size_of;

    pub struct KernelArgs {
        pub global_id: u32,
        pub groups: u32,
        pub group_id: u32,
        pub subgroups: u32,
        pub subgroup_id: u32,
        //pub subgroup_threads: u32,
        pub subgroup_thread_id: u32,
        pub threads: u32,
        pub thread_id: u32,
    }

    #[allow(deprecated)]
    impl KernelArgs {
        #[inline]
        pub unsafe fn into_kernel(self) -> Kernel {
            let Self {
                global_id,
                groups,
                group_id,
                subgroups,
                subgroup_id,
                //subgroup_threads,
                subgroup_thread_id,
                threads,
                thread_id,
            } = self;
            Kernel {
                global_threads: groups * threads,
                global_id,
                groups,
                group_id,
                subgroups,
                subgroup_id,
                //subgroup_threads,
                subgroup_thread_id,
                threads,
                thread_id,
            }
        }
    }

    // ensures __krnl_kernel_data is used, and not optimized away
    // removed by krnlc
    #[inline]
    pub unsafe fn kernel_data(data: &mut [u32]) {
        use spirv_std::arch::IndexUnchecked;

        unsafe {
            *data.index_unchecked_mut(0) = 1;
        }
    }

    // passes the length (constant, spec constant, or spec const expr) to krnlc
    // the array is changed from len 1 to the constant
    #[inline]
    pub unsafe fn group_buffer_len(data: &mut [u32], index: usize, len: usize) {
        use spirv_std::arch::IndexUnchecked;

        unsafe {
            *data.index_unchecked_mut(index) = if len > 0 { len as u32 } else { 1 };
        }
    }

    #[inline]
    pub unsafe fn zero_group_buffer<T: Default + Copy>(
        kernel: &Kernel,
        buffer: &mut [T; 1],
        len: usize,
    ) {
        use spirv_std::arch::IndexUnchecked;

        let stride = {
            if size_of::<T>() == 1 {
                4
            } else if size_of::<T>() == 2 {
                2
            } else {
                1
            }
        };

        let mut index = kernel.thread_id() * stride;
        if index < kernel.threads() * stride {
            while index < len {
                unsafe {
                    *buffer.index_unchecked_mut(index) = T::default();
                }
                if stride >= 2 {
                    if index + 1 < len {
                        unsafe {
                            *buffer.index_unchecked_mut(index + 1) = T::default();
                        }
                    }
                }
                if stride == 4 {
                    if index + 2 < len {
                        unsafe {
                            *buffer.index_unchecked_mut(index + 2) = T::default();
                        }
                    }
                    if index + 3 < len {
                        unsafe {
                            *buffer.index_unchecked_mut(index + 3) = T::default();
                        }
                    }
                }
                index += kernel.threads() * stride;
            }
        }
    }

    pub struct ItemKernelArgs {
        pub items: u32,
        pub item_id: u32,
    }

    #[allow(deprecated)]
    impl ItemKernelArgs {
        #[inline]
        pub unsafe fn into_item_kernel(self) -> ItemKernel {
            let Self { items, item_id } = self;
            ItemKernel { items, item_id }
        }
    }
}

pub struct Kernel {
    global_threads: u32,
    global_id: u32,
    groups: u32,
    group_id: u32,
    subgroups: u32,
    subgroup_id: u32,
    //subgroup_threads: u32,
    subgroup_thread_id: u32,
    threads: u32,
    thread_id: u32,
}

impl Kernel {
    /// The number of global threads.
    ///
    /// `global_threads = groups * threads`
    #[inline]
    pub fn global_threads(&self) -> usize {
        self.global_threads as usize
    }
    /// The global thread id.
    ///
    /// `global_id = group_id * threads + thread_id`
    #[inline]
    pub fn global_id(&self) -> usize {
        self.global_id as usize
    }
    /// The number of thread groups.
    #[inline]
    pub fn groups(&self) -> usize {
        self.groups as usize
    }
    /// The group id.
    #[inline]
    pub fn group_id(&self) -> usize {
        self.group_id as usize
    }
    /// The number of subgroups per group.
    #[inline]
    pub fn subgroups(&self) -> usize {
        self.subgroups as usize
    }
    /// The subgroup id.
    #[inline]
    pub fn subgroup_id(&self) -> usize {
        self.subgroup_id as usize
    }
    // TODO: Potentially implement via subgroup ballot / reduce operation
    /*
    /// The number of threads per subgroup.
    #[inline]
    pub fn subgroup_threads(&self) -> usize {
        self.subgroup_threads as usize
    }
    */
    /// The subgroup thread id.
    #[inline]
    pub fn subgroup_thread_id(&self) -> usize {
        self.subgroup_thread_id as usize
    }
    /// The number of threads per group.
    #[inline]
    pub fn threads(&self) -> usize {
        self.threads as usize
    }
    /// The thread id.
    #[inline]
    pub fn thread_id(&self) -> usize {
        self.thread_id as usize
    }
}

pub struct ItemKernel {
    items: u32,
    item_id: u32,
}

impl ItemKernel {
    /// The number of items.
    ///
    /// This will be the minimum length of buffers with `#[item]`.
    #[inline]
    pub fn items(&self) -> usize {
        self.items as usize
    }
    /// The id of the item.
    #[inline]
    pub fn item_id(&self) -> usize {
        self.item_id as usize
    }
}
