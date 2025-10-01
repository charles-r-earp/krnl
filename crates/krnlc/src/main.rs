use clap::Parser;
use krnlc::cli::Cli;

fn main() {
    Cli::parse().run();
}
