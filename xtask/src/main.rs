use clap::Parser;

mod charts;

#[derive(Parser, Debug)]
enum Cli {
    /// Generate charts.
    Charts
}

fn main() {
    let cli = Cli::parse();
    match cli {
        Cli::Charts => charts::generate(),
    }
}