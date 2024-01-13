fn main() -> Result<(), Box<dyn std::error::Error>> {

    capnpc::CompilerCommand::new()
        .file("proto/hello.capnp")
        .run()
        .unwrap();

    Ok(())
}
