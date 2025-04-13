use std::fs;

fn main() {
    // Create generated directory if it doesn't exist
    fs::create_dir_all("src/generated").expect("Failed to create generated directory");

    capnpc::CompilerCommand::new()
        .src_prefix(".")
        .file("./proto/book.capnp")
        .output_path("src/generated")
        .run()
        .expect("schema compiler command failed");
}
