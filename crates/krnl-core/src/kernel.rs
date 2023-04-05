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
            let group_index = group_id.x + group_id.y * groups.x + group_id.z * groups.x * groups.y;
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

#[cfg(any(doc, target_arch = "spirv"))]
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

#[cfg(any(doc, target_arch = "spirv"))]
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
    pub fn subgroups(&self) -> u32 {
        self.subgroups
    }
    pub fn subgroup_id(&self) -> u32 {
        self.subgroup_id
    }
    pub fn subgroup_thread_id(&self) -> u32 {
        self.subgroup_thread_id
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
}

#[cfg(any(doc, target_arch = "spirv"))]
pub struct ItemKernel<D> {
    item_id: D,
    item_index: u32,
    items: D,
}

#[cfg(any(doc, target_arch = "spirv"))]
impl<D: Copy> ItemKernel<D> {
    pub fn item_id(&self) -> D {
        self.item_id
    }
    pub fn item_index(&self) -> u32 {
        self.item_index
    }
    pub fn items(&self) -> D {
        self.items
    }
}
