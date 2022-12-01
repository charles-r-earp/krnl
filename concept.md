```rust

#[module]
mod foo {
    #[kernel(foreach, threads(TS))]
    pub fn cast_u32_f32<#[spec] const TS: u32>(
        #[item] x: &u32,
        #[item] y: &mut f32,
    ) {
        *y = *x as _;
    }

    #[kernel(threads(256))]
    pub fn saxpy(#[builtin] global_index: usize, #[global] x: &[f32], #[push] alpha: f32, #[global] y: &mut UnsafeMut<[f32]>) {
        if global_index < x.len() && global_index < y.len() {
            unsafe {
                *y.unsafe_mut()[global_index] += alpha * x[global_index];
            }
        }
    }

    #[kernel(group_threads(1)]
    pub fn foo<#[spec] const N: u32>(
        /* attrs */
        #[global] x: &[f32],
        #[global] x: &mut UnsafeMut<[f32]>,
        #[group] x: &mut UnsafeMut<[u32; 10]>,
        #[subgroup] x: &mut [i32; 10],
        #[thread] x: &mut [u32; n],
        #[push] alpha: f32,
        #[builtin] global_threads: UVec2,
        /* global types */
        #[global] x: &[u32],
        #[global] y: &mut UnsafeMut<[i32]>,
        /* builtins */
        // Vector impl From<u32 or [u32; 2] or [u32; 3]> based on Dimensionality
        // set by threads(..)
        global_threads: Vector,
        global_index: usize,
        global_id: Vector,
        groups: Vector,
        group_index: index,
        group_id: Vecor,
        subgroups: Vector,
        subgroup_index: usize,
        subgroup_id: Vector,
        threads: Vector,
        thread_index: usize,
        thread_id: Vector,
        /* for_each */
        items: u32,
        item_id: Vector,
        item_index: usize,
        #[item] x: &u32,
        #[item] y: &mut f32,
    ) {
        w.load();
        w.store();
    }
}

```


```rust

#[module]
pub mod foo {
    // etc

    /* generated */
    pub mod module {
        pub fn proto(name: &str) -> Result<KernelProto> {
            todo!()
        }
    }
}

```

# krnlc

Compiler for krnl

Creates cache at $CARGO_MANIFEST_DIR .krnl/cache and tag .krnl/CASHDIRTAG

module macro
```rust
#[module]
pub mod kernels {

    /* generated */
    mod module {
        _ = {
            let bytes = module(module_path!());
            let success =
                bytes.len() == len
                && bytes[0] == other[0]
                && bytes[0] == other[1]
                ..;
            if !success {
                panic!("module modified!");
            }
        };

        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/.krnl/cache"));
        /* cache */
        const fn __module(path: &'static str) -> Option<&'static [u8]> {
            todo!()
        }
        pub(super) const fn __kernel(path: &'static str) -> Option<&'static [u8]> {
            todo!()
        }
    }
}

/* options */
#[module(
    dependencies(r#"
        foo = { version = "0.1", git = "https:/github.com/myuser/mycrate", }
    "#)
)]
#[krnl(crate=krnl)]
#[krnl(no_build)]
```

/* krnl dir */
```
- .krnl
    - CASHDIR.TAG
    - packages
        - <package>
            - .gitignore // ignore modules
            - cache
            - modules
                - <module>
                    - Cargo.toml
                    - config.toml
                    - src
                        - lib.rs
            - target // target dir for check / expand 

## encode data in entry_point
__krnl_ <hash>_ <threads> _ <access> _ <kernel_name>

- threads are c<constant> or s<spec id>
- access is r (readonly) or w (writeonly) or a (readwrite) for each buffer
