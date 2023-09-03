mod hello_generated;

use anyhow::anyhow;
use anyhow::Result;
use flatbuffers::FlatBufferBuilder;
extern crate flatbuffers;

use flatbuffers::WIPOffset;
use hello_generated::*;

use hello_generated::HelloWorld;
use hello_generated::HelloWorldArgs;

fn main() {
    let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(1024);

    let name = builder.create_string("Hello, World!");

    let data = builder.create_vector(&[1u8, 2, 3, 4, 5]);

    let hello_world = HelloWorld::create(
        &mut builder,
        &HelloWorldArgs {
            name: Some(name),
            data: Some(data),
        },
    );

    builder.finish(hello_world, None);

    let buf = builder.finished_data();

    println!("serialized {}", hex::encode(buf));

    // Deserialization
    let hello_world = flatbuffers::root::<HelloWorld>(buf).unwrap();
    println!("deserialized {:?}", hello_world);

    assert_eq!(hello_world.name().unwrap(), "Hello, World!");
    assert_eq!(hello_world.data().unwrap().bytes(), &[1, 2, 3, 4, 5]);
}
