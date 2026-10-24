fn main() {
    if let Err(error) = packtok_semantic_scale::runner::cli(std::env::args().skip(1).collect()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
