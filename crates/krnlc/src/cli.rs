use camino::Utf8PathBuf;
use clap::{Parser, Subcommand};
use clap_cargo::{Manifest, Workspace};

#[derive(Parser, Debug)]
#[command(
    name = "krnlc",
    version = crate::VERSION_AND_SHA,
    about = "Compiler for krnl."
)]
pub struct Cli {
    #[clap(subcommand)]
    command: Command,
}

impl Cli {
    pub fn run(self) {
        self.command.run();
    }
}

#[derive(Subcommand, Debug)]
enum Command {
    #[cfg(feature = "rust-in")]
    Build(Build),
    #[cfg(feature = "print")]
    Print(Print),
}

impl Command {
    fn run(self) {
        match self {
            #[cfg(feature = "rust-in")]
            Self::Build(build) => build.run(),
            #[cfg(feature = "print")]
            Self::Print(print) => print.run(),
        }
    }
}

#[cfg(feature = "rust-in")]
#[derive(Parser, Debug)]
pub struct Build {
    #[command(flatten)]
    pub workspace: Workspace,
    #[command(flatten)]
    pub manifest: Manifest,
}

#[cfg(feature = "rust-in")]
impl Build {
    pub fn run(&self) {
        crate::rust_in::build_workspace(&self.workspace, &self.manifest);
    }
}

#[cfg(feature = "print")]
#[derive(Parser, Debug)]
pub struct Print {
    #[command(flatten)]
    pub workspace: Workspace,
    #[command(flatten)]
    pub manifest: Manifest,
    /// Path to module binary (ie 'krnl.spv').
    #[arg(long = "input-path")]
    pub input_path: Option<Utf8PathBuf>,
    filter: Option<String>,
}

#[cfg(feature = "print")]
impl Print {
    pub fn run(&self) {
        let filter = self.filter.as_deref();
        if let Some(input_path) = self.input_path.as_ref() {
            let spirv = std::fs::read(input_path).unwrap();
            let output = crate::print::print_to_string(spirv, filter);
            println!("{output}");
        } else {
            crate::print::print_workspace(&self.workspace, &self.manifest, filter);
        }
    }
}
