mod hello_capnp {
    include!(concat!(env!("OUT_DIR"), "/proto/hello_capnp.rs"));
}

use anyhow::Result;

fn serialize_my_data() -> Result<Vec<u8>> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut my_data = message.init_root::<hello_capnp::hello_world::my_data::Builder>();
        my_data.set_name("John Doe");
        my_data.set_age(30);
        my_data.set_disk(&[0x01, 0x02, 0x03, 0x04]);

        let mut notes = my_data.init_mynotes(2);
        notes.set(0, "Note 1");
        notes.set(1, "Note 2");
    }

    let mut serialized_data = Vec::new();
    capnp::serialize::write_message(&mut serialized_data, &message)?;

    Ok(serialized_data)
}

fn deserialize_my_data(data: &[u8]) -> Result<()> {
    let reader =
        capnp::serialize::read_message(&mut &data[..], capnp::message::ReaderOptions::new())
            ?;
    let my_data = reader
        .get_root::<hello_capnp::hello_world::my_data::Reader>()
        ?;

    println!("Name: {}", my_data.get_name()?);
    println!("Age: {}", my_data.get_age());
    println!("Disk: {:?}", my_data.get_disk()?);

    for note in my_data.get_mynotes()?.iter() {
        println!("Note: {}", note?);
    }
    Ok(())
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let serialized_data = serialize_my_data()?;
    // show hex of serialized data and length
    println!("serialized_data length: {}", serialized_data.len());
    println!("serialized_data: {}", hex::encode(&serialized_data));
    
    deserialize_my_data(&serialized_data)?;
    Ok(())
}
