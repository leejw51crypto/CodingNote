#![allow(dead_code)]
use anyhow::Result;
use capnp::serialize;
use std::io::{Cursor, Read, Write};
use rand::RngCore;
pub mod my_capnp {
    include!(concat!(env!("OUT_DIR"), "/src/my_capnp.rs"));
}

use my_capnp::{multimedia_file};

fn create_and_serialize(filename: &str, content: &[u8]) -> Result<Vec<u8>> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut multimedia_file = message.init_root::<multimedia_file::Builder>();
        multimedia_file.set_filename(filename.into());
        multimedia_file.set_size(content.len() as i64);
        multimedia_file.set_timestamp(chrono::Utc::now().timestamp_millis());
        multimedia_file.set_content(content);
        multimedia_file.set_is_public(true);

        // Set metadata
        let mut meta = multimedia_file.reborrow().init_metadata();
        meta.set_author("John Doe".into());
        meta.set_description("Sample multimedia file".into());
        meta.set_creation_date(chrono::Utc::now().timestamp_millis());
        meta.set_location("New York".into());

        // Set tags
        let tags = ["tag1", "tag2", "tag3"];
        let mut tag_list = multimedia_file.reborrow().init_tags(tags.len() as u32);
        for (i, tag) in tags.iter().enumerate() {
            tag_list.set(i as u32, (*tag).into());
        }
    }

    let mut serialized_data = Vec::new();
    serialize::write_message(&mut serialized_data, &message)?;
    Ok(serialized_data)
}

fn deserialize(serialized_data: &[u8]) -> Result<()> {
    let mut cursor = Cursor::new(serialized_data);
    let message_reader = serialize::read_message(&mut cursor, capnp::message::ReaderOptions::new())?;
    let multimedia_file = message_reader.get_root::<multimedia_file::Reader>()?;

    println!("Filename: {}", multimedia_file.get_filename()?.to_string()?);
    println!("Size: {}", multimedia_file.get_size());
    println!("Timestamp: {}", multimedia_file.get_timestamp());
    println!("Is Public: {}", multimedia_file.get_is_public());

    // Metadata
    let meta = multimedia_file.get_metadata()?;
    println!("Author: {}", meta.get_author()?.to_string()?);
    println!("Description: {}", meta.get_description()?.to_string()?);
    println!("Creation Date: {}", meta.get_creation_date());
    println!("Location: {}", meta.get_location()?.to_string()?);

    // Tags
    let tags = multimedia_file.get_tags()?;
    for i in 0..tags.len() {
        println!("Tag {}: {}", i, tags.get(i)?.to_string()?);
    }

    Ok(())
}

fn main() -> Result<()> {
    let filename = "example.jpg";
    let mut content: Vec<u8> = vec![0; 10 * 1024 * 1024];
    let mut rng = rand::thread_rng();
    rng.fill_bytes(&mut content);

    let serialized_data = create_and_serialize(filename, &content)?;
    let mut file = std::fs::File::create("a.bin")?;
    file.write_all(&serialized_data)?;
    deserialize(&serialized_data)?;
    println!("------------------------");

    let mut file = std::fs::File::open("a.bin")?;
    let mut serialized_data2 = Vec::new();
    file.read_to_end(&mut serialized_data2)?;
    deserialize(&serialized_data2)?;

    Ok(())
}
