#[cfg(target_arch = "spirv")]
use glam::{UVec2, UVec3, Vec3Swizzles};

#[doc(hidden)]
#[cfg(target_arch = "spirv")]
pub mod __private {
    use super::*;

    #[derive(Copy, Clone)]
    pub struct KernelArgs {
        pub groups: UVec3,
        pub group_id: UVec3,
        pub subgroups: u32,
        pub subgroup_id: u32,
        pub subgroup_threads: u32,
        pub subgroup_thread_id: u32,
        pub threads: UVec3,
        pub thread_id: UVec3,
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
                subgroups,
                subgroup_id,
                subgroup_thread_id,
                subgroup_threads,
                threads,
                thread_id,
                thread_index,
            } = self.kernel3();
            Kernel {
                global_threads: global_threads.x,
                global_id: global_id.x,
                global_index,
                groups: groups.x,
                group_id: group_id.x,
                group_index,
                subgroups,
                subgroup_id,
                subgroup_threads,
                subgroup_thread_id,
                threads: threads.x,
                thread_id: thread_id.x,
                thread_index,
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
                subgroups,
                subgroup_id,
                subgroup_thread_id,
                subgroup_threads,
                threads,
                thread_id,
                thread_index,
            } = self.kernel3();
            Kernel {
                global_threads: global_threads.xy(),
                global_id: global_id.xy(),
                global_index,
                groups: groups.xy(),
                group_id: group_id.xy(),
                group_index,
                subgroups,
                subgroup_id,
                subgroup_threads,
                subgroup_thread_id,
                threads: threads.xy(),
                thread_id: thread_id.xy(),
                thread_index,
            }
        }
        fn kernel3(self) -> Kernel<UVec3> {
            let Self {
                groups,
                group_id,
                subgroups,
                subgroup_id,
                subgroup_threads,
                subgroup_thread_id,
                threads,
                thread_id,
            } = self;
            fn index(id: UVec3, dim: UVec3) -> u32 {
                id.dot(UVec3::new(1, dim.x, dim.x * dim.y))
            }
            let global_threads = groups * threads;
            let global_id = group_id * threads + thread_id;
            let global_index = index(global_id, global_threads);
            let group_index = index(group_id, groups);
            let thread_index = index(thread_id, threads);
            Kernel {
                global_threads,
                global_id,
                global_index,
                groups,
                group_id,
                group_index,
                subgroups,
                subgroup_id,
                subgroup_threads,
                subgroup_thread_id,
                threads,
                thread_id,
                thread_index,
            }
        }
    }

    #[cfg(target_arch = "spirv")]
    impl From<KernelArgs> for Kernel<u32> {
        fn from(args: KernelArgs) -> Self {
            args.kernel1()
        }
    }

    #[cfg(target_arch = "spirv")]
    impl From<KernelArgs> for Kernel<UVec2> {
        fn from(args: KernelArgs) -> Self {
            args.kernel2()
        }
    }

    #[cfg(target_arch = "spirv")]
    impl From<KernelArgs> for Kernel<UVec3> {
        fn from(args: KernelArgs) -> Self {
            args.kernel3()
        }
    }

    pub struct ItemKernelArgs<D> {
        pub item_id: D,
        pub item_index: u32,
        pub items: D,
    }

    #[cfg(target_arch = "spirv")]
    impl<D> From<ItemKernelArgs<D>> for ItemKernel<D> {
        fn from(args: ItemKernelArgs<D>) -> Self {
            let ItemKernelArgs {
                item_id,
                item_index,
                items,
            } = args;
            Self {
                item_id,
                item_index,
                items,
            }
        }
    }
}

/// Kernels can be 1, 2, or 3 dimensional.
///
/// The dimension `D` is either [`u32`], [`UVec2`](glam::UVec2), or [`UVec3`](glam::UVec2).
/// This corresponds to x, y, and z dimensions, where z is the outer dimension and x is the fastest changing dimension.
pub struct Kernel<D> {
    global_threads: D,
    global_id: D,
    global_index: u32,
    groups: D,
    group_id: D,
    group_index: u32,
    subgroups: u32,
    subgroup_id: u32,
    subgroup_threads: u32,
    subgroup_thread_id: u32,
    threads: D,
    thread_id: D,
    thread_index: u32,
}

impl<D: Copy> Kernel<D> {
    /// The number of global threads.
    ///
    /// `global_threads = groups * threads`
    pub fn global_threads(&self) -> D {
        self.global_threads
    }
    /// The global thread id.
    ///
    /// See [`.global_index()`](Kernel::global_index).
    pub fn global_id(&self) -> D {
        self.global_id
    }
    /// The global thread index.
    ///
    /// `global_index = global_id.dot(1, global_threads.x, global_threads.x * global_threads.y)`
    pub fn global_index(&self) -> u32 {
        self.global_index
    }
    /// The number of thread groups.
    ///
    /// Kernels are dispatched with groups of threads.
    /// See [`.global_threads()`](Kernel::global_threads).
    pub fn groups(&self) -> D {
        self.groups
    }
    /// The group id.
    ///
    /// See [`.group_index()`](Kernel::group_index).
    pub fn group_id(&self) -> D {
        self.group_id
    }
    /// The group index.
    ///
    /// `group_index = group_id.dot(1, groups.x, groups.x * groups.y)`
    pub fn group_index(&self) -> u32 {
        self.group_index
    }
    /// The number of subgroups.
    pub fn subgroups(&self) -> u32 {
        self.subgroups
    }
    /// The subgroup id within the group.
    pub fn subgroup_id(&self) -> u32 {
        self.subgroup_id
    }
    /// The number of threads per subgroup
    pub fn subgroup_threads(&self) -> u32 {
        self.subgroup_threads
    }
    /// The id of the thread within the subgroup.
    pub fn subgroup_thread_id(&self) -> u32 {
        self.subgroup_thread_id
    }
    /// The number of threads.
    ///
    /// This is the same as provided via `#[kernel(threads(..))]`.
    pub fn threads(&self) -> D {
        self.threads
    }
    /// The thread id in the thread group.
    ///
    /// See [`.thread_index()`](Kernel::thread_index).
    pub fn thread_id(&self) -> D {
        self.thread_id
    }
    /// The thread index in the thread group.
    ///
    /// `thread_index = thread_id.dot(1, threads.x, threads.x * threads.y)`
    pub fn thread_index(&self) -> u32 {
        self.thread_index
    }
}

/// Item kernels are 1 dimensional, with `D` as [`u32`].
pub struct ItemKernel<D> {
    item_id: D,
    item_index: u32,
    items: D,
}

impl<D: Copy> ItemKernel<D> {
    /// The id of the item.
    /// See [`.item_index()`](ItemKernel::item_index).
    pub fn item_id(&self) -> D {
        self.item_id
    }
    /// The index of the item, from 0 .. [`.items()`](ItemKernel::items).
    pub fn item_index(&self) -> u32 {
        self.item_index
    }
    /// The number of items.
    ///
    /// This will be the minimum length of buffers with `#[item]`.
    pub fn items(&self) -> D {
        self.items
    }
}
