use capnp::message::{Builder, ReaderOptions};
use capnp::serialize::write_message;
use capnp::serialize_packed;
use fake::faker::company::en::CompanyName;
use fake::faker::internet::en::SafeEmail;
use fake::faker::name::en::Name;
use fake::{Fake, Faker};
use hex;
use std::io::Write;

pub mod book_capnp {
    include!(concat!(env!("OUT_DIR"), "/book_capnp.rs"));
}

fn main() {
    println!("=== Creating Library Data ===");
    // Create a library with 10 random books
    let mut message = Builder::new_default();
    let mut library = message.init_root::<book_capnp::library::Builder>();
    library.set_name("My Awesome Library");
    println!("Created library: My Awesome Library");

    let mut books = library.reborrow().init_books(10);
    println!("\nGenerating 10 random books...");
    for i in 0..10 {
        let mut book = books.reborrow().get(i);
        let title: String = CompanyName().fake();
        book.set_title(&title);

        {
            let mut author = book.get_author().unwrap();
            let name: String = Name().fake();
            let email: String = SafeEmail().fake();
            author.set_name(&name);
            author.set_email(&email);
        }
        println!("Book {}: {} by {}", i + 1, title, Name().fake::<String>());
    }

    println!("\n=== Serialization ===");
    // First serialize to a Vec
    let mut raw_data = Vec::new();
    write_message(&mut raw_data, &message).unwrap();
    println!("Raw serialized size: {} bytes", raw_data.len());

    // Now create packed data
    let mut packed_output = Vec::new();
    serialize_packed::write_message(&mut packed_output, &message).unwrap();

    println!("Packed serialized size: {} bytes", packed_output.len());
    println!(
        "Compression ratio: {:.2}%",
        (1.0 - packed_output.len() as f64 / raw_data.len() as f64) * 100.0
    );

    // Print hex dump of packed data
    println!("\nPacked data hex dump (first 100 bytes):");
    let hex_dump = hex::encode(&packed_output[..packed_output.len().min(100)]);
    for chunk in hex_dump.as_bytes().chunks(32) {
        println!("{}", String::from_utf8_lossy(chunk));
    }

    println!("\n=== Deserialization ===");
    // Deserialize from packed format
    let message_reader =
        serialize_packed::read_message(&mut &packed_output[..], ReaderOptions::new()).unwrap();

    let library_reader = message_reader
        .get_root::<book_capnp::library::Reader>()
        .unwrap();
    println!("Successfully deserialized library");
    println!("\nLibrary name: {:?}", library_reader.get_name().unwrap());
    println!("\nBooks in library:");

    let books = library_reader.get_books().unwrap();
    for book in books.iter() {
        let author = book.get_author().unwrap();
        println!("\nTitle: {:?}", book.get_title().unwrap());
        println!(
            "Author: {:?} ({:?})",
            author.get_name().unwrap(),
            author.get_email().unwrap()
        );
    }
}
