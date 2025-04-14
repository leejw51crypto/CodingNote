/*
init_root
write_messge

read_message
get_root
set_root

*/
use capnp::message::{Builder, Reader, ReaderOptions};
use capnp::serialize;
#[path = "../src/generated/proto/book_capnp.rs"]
pub mod book_capnp;
use crate::book_capnp::book;
use sha2::{Digest, Sha256};

fn main() -> capnp::Result<()> {
    // Create a message with a Book
    let mut message = Builder::new_default();
    {
        let mut book = message.init_root::<book::Builder>();
        book.set_title("The Rust Programming Language".into());
        book.set_author("Steve Klabnik and Carol Nichols".into());
        book.set_pages(500);
        book.set_publish_year(2019);
        book.set_is_available(true);
    }

    // Serialize the message to bytes
    let mut serialized_data = Vec::new();
    serialize::write_message(&mut serialized_data, &message)?;
    println!("Serialized data (hex):");
    println!("{}", hex::encode(&serialized_data));
    let hash = Sha256::digest(&serialized_data);
    println!("SHA-256: {}\n", hex::encode(hash));

    // Create a reader from the serialized data
    let reader = serialize::read_message(&serialized_data[..], ReaderOptions::new())?;

    // Create a new message and use set_root to copy the book
    let mut new_message = Builder::new_default();
    {
        let book_reader = reader.get_root::<book::Reader>()?;
        println!("\nBefore modification:");
        println!("Title: {}", book_reader.get_title()?.to_str()?);
        println!("Author: {}", book_reader.get_author()?.to_str()?);
        println!("Pages: {}", book_reader.get_pages());
        println!("Publish Year: {}", book_reader.get_publish_year());
        println!("Is Available: {}", book_reader.get_is_available());

        new_message.set_root(book_reader)?;

        // Modify the author
        let mut book_builder = new_message.get_root::<book::Builder>()?;
        book_builder.set_author("Steve Klabnik, Carol Nichols, and the Rust Community".into());

        println!("\nAfter modification:");
        let modified_book = new_message.get_root_as_reader::<book::Reader>()?;
        println!("Title: {}", modified_book.get_title()?.to_str()?);
        println!("Author: {}", modified_book.get_author()?.to_str()?);
        println!("Pages: {}", modified_book.get_pages());
        println!("Publish Year: {}", modified_book.get_publish_year());
        println!("Is Available: {}", modified_book.get_is_available());
    }

    // Serialize the modified message
    let mut modified_data = Vec::new();
    serialize::write_message(&mut modified_data, &new_message)?;
    println!("\nModified serialized data (hex):");
    println!("{}", hex::encode(&modified_data));
    let modified_hash = Sha256::digest(&modified_data);
    println!("SHA-256: {}", hex::encode(modified_hash));

    Ok(())
}
