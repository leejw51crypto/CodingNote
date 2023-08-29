fn main() -> Result<(), Box<dyn std::error::Error>> {
    capnpc::CompilerCommand::new()
        .file("proto/blockchain.capnp")
        .run()
        .unwrap();
    Ok(())
}
