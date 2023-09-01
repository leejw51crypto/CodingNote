mod hello_generated;

use flatbuffers::FlatBufferBuilder;
use hello_generated::hello_world::Hello;
use hello_generated::hello_world::root_as_hello;
use hello_generated::hello_world::HelloArgs;
use anyhow::Result;
use anyhow::anyhow;
fn main()->Result<()>{
    let message = "Hello, World!";
    
    // Serialize the message
    let mut builder = FlatBufferBuilder::with_capacity(1024);
    let message_offset = builder.create_string(message);
    let hello = Hello::create(&mut builder, &HelloArgs { message: Some(message_offset) });
    builder.finish(hello, None);

    let buf = builder.finished_data(); // Serialized data

    // Deserialize the message
    let hello = root_as_hello(buf)?;
    let deserialized_message = hello.message().ok_or(anyhow!("Message not found"))?;
    
    
    
    println!("{}", deserialized_message);
    Ok(())
}
