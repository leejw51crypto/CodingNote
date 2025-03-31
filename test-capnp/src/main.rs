#![allow(dead_code)]
use anyhow::Result;
use capnp::serialize_packed;
use std::io::Cursor;

mod book_capnp {
    include!(concat!(env!("OUT_DIR"), "/book_capnp.rs"));
}
use book_capnp::{author, book};

fn serialize_book() -> Result<Vec<u8>> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut book_builder = message.init_root::<book::Builder>();

        // Set book title
        book_builder.set_title("The Rust Programming Language");

        // Set author information
        let mut author_builder = book_builder.reborrow().init_author();
        author_builder.set_name("Steve Minecraft");
        author_builder.set_email("steve email address");
    }

    let mut serialized_data = Vec::new();
    serialize_packed::write_message(&mut serialized_data, &message)?;
    Ok(serialized_data)
}

fn deserialize_book(serialized_data: &[u8]) -> Result<()> {
    let reader = serialize_packed::read_message(
        &mut Cursor::new(serialized_data),
        capnp::message::ReaderOptions::new(),
    )?;
    let book = reader.get_root::<book::Reader>()?;

    println!("Deserialized Book:");
    println!("Title: {}", book.get_title()?.to_string()?);

    let author = book.get_author()?;
    println!("Author:");
    println!("  Name: {}", author.get_name()?.to_string()?);
    println!("  Email: {}", author.get_email()?.to_string()?);

    // Serialize author directly with pack mode
    let mut author_bytes = Vec::new();
    {
        let mut message = capnp::message::Builder::new_default();
        message.set_root(author)?;
        serialize_packed::write_message(&mut author_bytes, &message)?;
    }
    println!("--------------------------------");
    println!("Author bytes (hex): {}", hex::encode(&author_bytes));
    println!("--------------------------------");

    // Deserialize and verify author bytes using pack mode
    let author_reader = serialize_packed::read_message(
        &mut Cursor::new(&author_bytes),
        capnp::message::ReaderOptions::new(),
    )?;
    let deserialized_author = author_reader.get_root::<author::Reader>()?;

    println!("Verifying deserialized author data:");
    println!("Original name: {}", author.get_name()?.to_string()?);
    println!(
        "Deserialized name: {}",
        deserialized_author.get_name()?.to_string()?
    );
    println!("Original email: {}", author.get_email()?.to_string()?);
    println!(
        "Deserialized email: {}",
        deserialized_author.get_email()?.to_string()?
    );
    println!(
        "Values match: {}",
        author.get_name()?.to_string()? == deserialized_author.get_name()?.to_string()?
            && author.get_email()?.to_string()? == deserialized_author.get_email()?.to_string()?
    );

    Ok(())
}

fn main() -> Result<()> {
    // Serialize a book
    let serialized_data = serialize_book()?;
    println!("Serialized data length: {}", serialized_data.len());
    println!("Serialized data (hex): {}", hex::encode(&serialized_data));

    // Deserialize and print the book
    deserialize_book(&serialized_data)?;

    Ok(())
}
