fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_files = ["book.capnp"];

    for file in &proto_files {
        capnpc::CompilerCommand::new().file(file).run()?;
    }

    // Optional: Generate reference files with newer settings
    for file in &proto_files {
        capnpc::CompilerCommand::new()
            .file(file)
            .output_path("proto/generated")
            // Add these options to generate more idiomatic Rust code
            .default_parent_module(vec!["generated".into()])
            .run()?;
    }

    Ok(())
}
