extern crate capnpc;

fn main() {
    capnpc::CompilerCommand::new()
        .file("src/addressbook.capnp")
        .run()
        .expect("Compiling Cap'n Proto schema");
}
