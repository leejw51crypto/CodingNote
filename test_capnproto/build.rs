fn main() -> Result<(), Box<dyn std::error::Error>> {
    capnpc::CompilerCommand::new()
        .file("src/my.capnp")
        .run()
        .unwrap();

    Ok(())
}
