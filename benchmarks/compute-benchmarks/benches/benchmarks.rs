#[cfg(feature = "cuda")]
use compute_benchmarks::cuda_backend::CudaBackend;
use compute_benchmarks::krnl_backend::KrnlBackend;
#[cfg(feature = "ocl")]
use compute_benchmarks::ocl_backend::OclBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::{distributions::OpenClosed01, thread_rng, Rng};
use std::{
    cell::RefCell,
    env::var,
    rc::Rc,
    str::FromStr,
    time::{Duration, Instant},
};

fn index_from_env(name: &str) -> usize {
    if let Ok(value) = var(name) {
        usize::from_str(&value).unwrap()
    } else {
        0
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let n_max = if cfg!(debug_assertions) {
        1000
    } else {
        256_000_000
    };
    let lens = [
        ("1", 1),
        ("1M", 1_000_000.min(n_max)),
        ("64M", 64_000_000.min(n_max)),
        ("256M", 256_000_000.min(n_max)),
    ];
    let saxpy_lens = [("1", 1), ("64M", 64_000_000.min(n_max))];
    let x: Rc<Vec<f32>> = Rc::new(thread_rng().sample_iter(OpenClosed01).take(n_max).collect());
    let alpha = 0.5;
    let y: Vec<f32> = thread_rng().sample_iter(OpenClosed01).take(n_max).collect();
    let index = index_from_env("KRNL_DEVICE");
    let krnl = KrnlBackend::new(index).unwrap();
    for (s, n) in lens {
        c.bench_function(&format!("alloc_{s}_krnl"), |b| {
            let krnl = krnl.clone();
            b.iter_custom(move |i| {
                let mut duration = Duration::default();
                for _ in 0..i {
                    let start = Instant::now();
                    let _alloc = krnl.alloc(n).unwrap();
                    duration += start.elapsed();
                }
                duration
            });
        });
    }
    for (s, n) in lens {
        let krnl = krnl.clone();
        let x = x.clone();
        c.bench_function(&format!("upload_{s}_krnl"), move |b| {
            let krnl = krnl.clone();
            let x = x.clone();
            b.iter_custom(move |i| {
                let x = &x[..n];
                let mut duration = Duration::default();
                for _ in 0..i {
                    let start = Instant::now();
                    let _upload = krnl.upload(x).unwrap();
                    duration += start.elapsed();
                }
                duration
            });
        });
    }
    for (s, n) in lens {
        let krnl = krnl.clone();
        let x = x.clone();
        c.bench_function(&format!("download_{s}_krnl"), move |b| {
            let download = krnl.download(&x[..n]).unwrap();
            b.iter_custom(move |i| {
                let mut duration = Duration::default();
                for _ in 0..i {
                    let start = Instant::now();
                    download.run().unwrap();
                    duration += start.elapsed();
                }
                duration
            });
        });
    }
    for (s, n) in lens {
        let x = Rc::new(x[..n].to_vec());
        let y = Rc::new(y[..n].to_vec());
        let saxpy = Rc::new(RefCell::new(krnl.saxpy(&x, alpha, &y).unwrap()));
        c.bench_function(&format!("saxpy_{s}_krnl"), move |b| {
            let saxpy = saxpy.clone();
            b.iter_custom(move |i| {
                let mut duration = Duration::default();
                for _ in 0..i {
                    let start = Instant::now();
                    saxpy.borrow_mut().run().unwrap();
                    duration += start.elapsed();
                }
                duration
            });
        });
    }
    #[cfg(feature = "cuda")]
    {
        let index = index_from_env("CUDA_DEVICE");
        let cuda = CudaBackend::new(index).unwrap();
        for (s, n) in lens {
            c.bench_function(&format!("alloc_{s}_cuda"), |b| {
                let cuda = cuda.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = cuda.alloc(n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let cuda = cuda.clone();
            let x = x.clone();
            c.bench_function(&format!("upload_{s}_cuda"), move |b| {
                let cuda = cuda.clone();
                let x = x.clone();
                b.iter_custom(move |i| {
                    let x = &x[..n];
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _upload = cuda.upload(x).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let cuda = cuda.clone();
            let x = x.clone();
            c.bench_function(&format!("download_{s}_cuda"), move |b| {
                let download = cuda.download(&x[..n]).unwrap();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        download.run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let x = Rc::new(x[..n].to_vec());
            let y = Rc::new(y[..n].to_vec());
            let saxpy = Rc::new(RefCell::new(cuda.saxpy(&x, alpha, &y).unwrap()));
            c.bench_function(&format!("saxpy_{s}_cuda"), move |b| {
                let saxpy = saxpy.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        saxpy.borrow_mut().run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
    }
    #[cfg(feature = "ocl")]
    {
        let platform_index = index_from_env("OCL_PLATFORM");
        let device_index = index_from_env("OCL_DEVICE");
        let ocl = OclBackend::new(platform_index, device_index).unwrap();
        for (s, n) in lens {
            c.bench_function(&format!("alloc_{s}_ocl"), |b| {
                let ocl = ocl.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = ocl.alloc(n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let ocl = ocl.clone();
            let x = x.clone();
            c.bench_function(&format!("upload_{s}_ocl"), move |b| {
                let ocl = ocl.clone();
                let x = x.clone();
                b.iter_custom(move |i| {
                    let x = &x[..n];
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _upload = ocl.upload(x).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let ocl = ocl.clone();
            let x = x.clone();
            c.bench_function(&format!("download_{s}_ocl"), move |b| {
                let download = ocl.download(&x[..n]).unwrap();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        download.run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }

        for (s, n) in lens {
            let x = Rc::new(x[..n].to_vec());
            let y = Rc::new(y[..n].to_vec());
            let saxpy = Rc::new(RefCell::new(ocl.saxpy(&x, alpha, &y).unwrap()));
            c.bench_function(&format!("saxpy_{s}_ocl"), move |b| {
                let saxpy = saxpy.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        saxpy.borrow_mut().run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
