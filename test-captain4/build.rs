fn main() -> Result<(), Box<dyn std::error::Error>> {
    capnpc::CompilerCommand::new()
        .file("proto/fruit.capnp")
        .run()
        .unwrap();
    Ok(())
}
