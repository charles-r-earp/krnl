fn main() {
    let short = true;
    vergen::EmitBuilder::builder()
        .git_sha(short)
        .emit()
        .unwrap();
}
