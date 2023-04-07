use clap::Parser;
use std::{
    env::var,
    fmt::Debug,
    path::PathBuf,
    process::{Command, ExitStatus},
    time::Instant,
};

#[derive(Parser, Debug)]
enum Cli {
    Krnlc {
        #[arg(long = "workspace")]
        workspace: bool,
        #[arg(long = "check")]
        check: bool,
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },
    WasmTest {
        #[arg(long = "firefox")]
        firefox: bool,
        #[arg(long = "safari")]
        safari: bool,
        #[arg(long = "chrome")]
        chrome: bool,
        #[arg(long = "node")]
        node: bool,
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },
    Lint {
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },
    Validate {
        #[arg(long = "device")]
        device: bool,
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli {
        Cli::Krnlc {
            workspace,
            check,
            verbose,
        } => {
            run_krnlc(workspace, check, verbose);
        }
        Cli::WasmTest {
            firefox,
            safari,
            chrome,
            node,
            verbose,
        } => {
            use WasmRunner::*;
            let runners = [
                (Firefox, firefox),
                (Safari, safari),
                (Chrome, chrome),
                (Node, node),
            ];
            if runners.iter().any(|(_, requested)| *requested) {
                for (runner, requested) in runners {
                    if requested {
                        run_wasm_tests(Some(runner), verbose);
                    }
                }
            } else {
                run_wasm_tests(None, verbose);
            }
        }
        Cli::Lint { verbose } => {
            run_lints(verbose);
        }
        Cli::Validate { device, verbose } => {
            run_validation(device, verbose);
        }
    }
}

trait ExitStatusExpect {
    fn expect2(self, msg: &str);
}

impl<E: Debug> ExitStatusExpect for Result<ExitStatus, E> {
    fn expect2(self, msg: &str) {
        if !self.expect(msg).success() {
            panic!("{msg}");
        }
    }
}

fn has_spirv_tools() -> bool {
    Command::new("spirv-val").arg("--version").output().is_ok()
}

fn run_krnlc(workspace: bool, check: bool, verbose: bool) {
    let spirv_tools = has_spirv_tools();
    if spirv_tools && verbose {
        eprintln!("found spirv-tools");
    }
    let manifest_dir = PathBuf::from(var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.parent().unwrap();
    let krnlc_dir = workspace_dir.join("crates/krnlc");
    let mut packages = vec!["krnl"];
    if workspace {
        packages.extend(["krnlc-tests", "compute-benchmarks"]);
    }
    let mut krnlc_args: Vec<_> = packages.iter().copied().flat_map(|p| ["-p", p]).collect();
    if check {
        krnlc_args.push("--check");
    }
    let mut command = Command::new("cargo");
    command.args([
        "run",
        "--release",
        "--target-dir",
        workspace_dir.join("target").to_str().unwrap(),
    ]);
    if verbose {
        command.arg("-v");
    }
    if has_spirv_tools() {
        command.args(["--no-default-features", "--features", "use-installed-tools"]);
    }
    command
        .args([
            "--",
            "--manifest-path",
            workspace_dir.join("Cargo.toml").to_str().unwrap(),
        ])
        .args(krnlc_args.as_slice())
        .current_dir(krnlc_dir.to_str().unwrap())
        .env_remove("RUSTUP_TOOLCHAIN");
    command.status().expect2("krnlc failed!");
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WasmRunner {
    Firefox,
    Safari,
    Chrome,
    Node,
}

impl WasmRunner {
    fn args(&self) -> &'static [&'static str] {
        use WasmRunner::*;
        match self {
            Firefox => &["--headless", "--firefox"],
            Safari => &["--headless", "--safari"],
            Chrome => &["--headless", "--chrome"],
            Node => &["--node"],
        }
    }
    fn app(&self) -> &'static str {
        use WasmRunner::*;
        match self {
            Firefox => "firefox",
            Safari => "safari",
            Chrome => "chrome",
            Node => "nodejs",
        }
    }
    fn installed(&self) -> bool {
        Command::new(self.app()).arg("--version").output().is_ok()
    }
    fn get() -> Option<Self> {
        use WasmRunner::*;
        [Firefox, Safari, Chrome, Node]
            .into_iter()
            .find(|&runner| runner.installed())
    }
}

fn run_wasm_tests(runner: Option<WasmRunner>, verbose: bool) {
    let runner = if let Some(runner) = runner {
        if !runner.installed() {
            panic!("{} is not installed!", runner.app());
        }
        runner
    } else {
        WasmRunner::get().expect("no wasm runner found, tried firefox, safari, chrome, nodejs")
    };
    let has_wasm_pack = Command::new("wasm-pack").arg("--version").status().is_ok();
    if !has_wasm_pack {
        Command::new("cargo")
            .args(["install", "wasm-pack"])
            .status()
            .expect2("installing wasm-pack failed!");
    }
    let manifest_dir = PathBuf::from(var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.parent().unwrap();
    let krnl_dir = workspace_dir.join("crates/krnl");
    let mut command = Command::new("wasm-pack");
    command.arg("test");
    command.args(runner.args());
    command
        .args([
            "--",
            "--no-default-features",
            "--test",
            "wasm_integration_tests",
        ])
        .current_dir(krnl_dir.to_str().unwrap());
    if verbose {
        command.arg("-v");
    }
    if runner != WasmRunner::Node {
        command.env("RUSTFLAGS", "--cfg run_in_browser");
    }
    command.status().expect2("test failed!");
}

fn run_lints(verbose: bool) {
    let manifest_dir = PathBuf::from(var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.parent().unwrap();
    let krnlc_dir = workspace_dir.join("crates/krnlc");
    let cuda = has_cuda();
    if cuda && verbose {
        eprintln!("found cuda");
    }
    Command::new("cargo")
        .args(["fmt", "--all", "--check"])
        .status()
        .expect2("cargo fmt failed!");
    let mut command = Command::new("cargo");
    command.args([
        "clippy",
        "--workspace",
        "--all-features",
        "--all-targets",
        "--exclude",
        "compute-benchmarks",
    ]);
    if verbose {
        command.arg("-v");
    }
    command.args(["--", "-D", "warnings"]);
    command.status().expect2("clippy failed!");
    Command::new("rustup")
        .args(["toolchain", "install", "nightly-2023-01-21"])
        .status()
        .expect2("failed to install nightly-2023-01-21!");
    Command::new("rustup")
        .args([
            "component",
            "add",
            "clippy",
            "rust-src",
            "rustc-dev",
            "llvm-tools-preview",
            "--toolchain",
            "nightly-2023-01-21",
        ])
        .status()
        .expect2("failed to install clippy on nightly-2023-01-21!");
    let mut command = Command::new("cargo");
    command.args([
        "+nightly-2023-01-21",
        "clippy",
        "--no-default-features",
        "--features",
        "use-installed-tools",
    ]);
    command
        .args([
            "--target-dir",
            workspace_dir.join("target").to_str().unwrap(),
            "--",
            "-D",
            "warnings",
        ])
        .current_dir(krnlc_dir.to_str().unwrap())
        .env_remove("RUSTUP_TOOLCHAIN");
    command.status().expect2("clippy failed!");
    let mut command = Command::new("cargo");
    command.args([
        "clippy",
        "-p",
        "compute-benchmarks",
        "--features",
        "autograph ocl",
    ]);
    if cuda {
        command.args(["--features", "cuda"]);
    }
    if verbose {
        command.arg("-v");
    }
    command.args(["--", "-D", "warnings"]);
    command.status().expect2("clippy failed!");
}

fn run_validation(device: bool, verbose: bool) {
    let start = Instant::now();
    run_lints(verbose);
    let targets = [
        "x86_64-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "x86_64-apple-ios",
        "x86_64-pc-windows-msvc",
        "wasm32-unknown-unknown",
    ];
    Command::new("rustup")
        .args(["target", "add"])
        .args(targets)
        .status()
        .expect2("target add failed!");
    for target in targets {
        let mut command = Command::new("cargo");
        command.args(["check", "--workspace", "--target", target]);
        if verbose {
            command.arg("-v");
        }
        if target.starts_with("wasm") {
            command.args(["--no-default-features", "--exclude", "compute-benchmarks"]);
        }
        command
            .status()
            .expect2(&format!("checking {target} failed!"));
    }
    Command::new("rustup")
        .args(["toolchain", "install", "nightly"])
        .status()
        .expect2("Failed to install nightly!");
    run_krnlc(true, true, verbose);
    let mut command = Command::new("cargo");
    command.args(["test", "--workspace", "--exclude", "xtask"]);
    if !device {
        command.args(["--exclude", "compute-benchmarks", "--no-default-features"]);
    }
    if verbose {
        command.arg("-v");
    }
    command.args(["--", "--format=terse"]);
    command.status().expect2("test failed!");
    run_wasm_tests(None, verbose);
    let mut command = Command::new("cargo");
    command.args([
        "check",
        "-p",
        "compute-benchmarks",
        "--benches",
        "--features",
        "autograph ocl",
    ]);
    let cuda = device && has_cuda();
    if cuda && verbose {
        eprintln!("found cuda");
        command.args(["--features", "cuda"]);
    }
    if verbose {
        command.arg("-v");
    }
    command.status().expect2("check failed!");
    if device {
        let mut command = Command::new("cargo");
        command.args([
            "test",
            "-p",
            "compute-benchmarks",
            "--benches",
            "--features",
            "autograph ocl",
        ]);
        if cuda {
            command.args(["--features", "cuda"]);
        }
        if verbose {
            command.arg("-v");
        }
        command.status().expect2("check failed!");
    }
    Command::new("rustup")
        .args([
            "component",
            "add",
            "miri",
            "--target",
            "x86_64-unknown-linux-gnu",
            "--toolchain",
            "nightly",
        ])
        .status()
        .expect2("Failed to install miri!");
    let mut command = Command::new("cargo");
    command.args([
        "+nightly",
        "miri",
        "test",
        "--target",
        "x86_64-unknown-linux-gnu",
        "--no-default-features",
    ]);
    if verbose {
        command.arg("-v");
    }
    command.args(["--", "--format=terse"]);
    command.env_remove("RUST_TOOLCHAIN");
    command.env_remove("CARGO");
    command.status().expect2("miri test failed!");
    println!("finished in {:?}", start.elapsed());
}

fn has_cuda() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok()
}
