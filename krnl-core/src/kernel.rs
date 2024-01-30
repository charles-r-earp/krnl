#[doc(hidden)]
#[cfg(target_arch = "spirv")]
pub mod __private {
    use super::{ItemKernel, Kernel};

    pub struct KernelArgs {
        pub global_threads: u32,
        pub global_id: u32,
        pub groups: u32,
        pub group_id: u32,
        pub subgroups: u32,
        pub subgroup_id: u32,
        pub subgroup_threads: u32,
        pub subgroup_thread_id: u32,
        pub threads: u32,
        pub thread_id: u32,
    }

    #[allow(deprecated)]
    impl KernelArgs {
        pub unsafe fn into_kernel(self) -> Kernel {
            let Self {
                global_threads,
                global_id,
                groups,
                group_id,
                subgroups,
                subgroup_id,
                subgroup_threads,
                subgroup_thread_id,
                threads,
                thread_id,
            } = self;
            Kernel {
                global_threads,
                global_id,
                groups,
                group_id,
                subgroups,
                subgroup_id,
                subgroup_threads,
                subgroup_thread_id,
                threads,
                thread_id,
            }
        }
    }

    pub unsafe fn zero_group_buffer<T: Default + Copy>(
        kernel: &Kernel,
        buffer: &mut [T; 1],
        len: usize,
    ) {
        use spirv_std::arch::IndexUnchecked;

        let mut index = kernel.thread_id();
        let threads = kernel.threads();
        if index < threads {
            while index < len {
                unsafe {
                    *buffer.index_unchecked_mut(index) = T::default();
                }
                index += threads;
            }
        }
    }

    pub struct ItemKernelArgs {
        pub items: u32,
        pub item_id: u32,
    }

    #[allow(deprecated)]
    impl ItemKernelArgs {
        pub unsafe fn into_item_kernel(self) -> ItemKernel {
            let Self { items, item_id } = self;
            ItemKernel { items, item_id }
        }
    }
}

#[non_exhaustive]
pub struct Kernel {
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with global_threads()")]
    pub global_threads: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with global_id()")]
    pub global_id: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with groups()")]
    pub groups: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with group_id()")]
    pub group_id: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with subgroups()")]
    pub subgroups: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with subgroup_id()")]
    pub subgroup_id: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with subgroup_threads()")]
    pub subgroup_threads: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with subgroup_thread_id()")]
    pub subgroup_thread_id: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with threads()")]
    pub threads: u32,
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with thread_id()")]
    pub thread_id: u32,
}

#[allow(deprecated)]
impl Kernel {
    /// The number of global threads.
    ///
    /// `global_threads = groups * threads`
    pub fn global_threads(&self) -> usize {
        self.global_threads as usize
    }
    /// The global thread id.
    ///
    /// `global_id = group_id * threads + thread_id`
    pub fn global_id(&self) -> usize {
        self.global_id as usize
    }
    /// The number of thread groups.
    pub fn groups(&self) -> usize {
        self.groups as usize
    }
    /// The group id.
    pub fn group_id(&self) -> usize {
        self.group_id as usize
    }
    /// The number of subgroups per group.
    pub fn subgroups(&self) -> usize {
        self.subgroups as usize
    }
    /// The subgroup id.
    pub fn subgroup_id(&self) -> usize {
        self.subgroup_id as usize
    }
    /// The number of threads per subgroup.
    pub fn subgroup_threads(&self) -> usize {
        self.subgroup_threads as usize
    }
    /// The subgroup thread id.
    pub fn subgroup_thread_id(&self) -> usize {
        self.subgroup_thread_id as usize
    }
    /// The number of threads per group.
    pub fn threads(&self) -> usize {
        self.threads as usize
    }
    /// The thread id.
    pub fn thread_id(&self) -> usize {
        self.thread_id as usize
    }
}

#[non_exhaustive]
pub struct ItemKernel {
    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with items()")]
    pub items: u32,

    #[doc(hidden)]
    #[deprecated(since = "0.0.4", note = "replaced with item_id()")]
    pub item_id: u32,
}

#[allow(deprecated)]
impl ItemKernel {
    /// The number of items.
    ///
    /// This will be the minimum length of buffers with `#[item]`.
    pub fn items(&self) -> usize {
        self.items as usize
    }
    /// The id of the item.
    pub fn item_id(&self) -> usize {
        self.item_id as usize
    }
}
