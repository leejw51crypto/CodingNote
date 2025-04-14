use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;

#[path = "../src/generated/proto/book_capnp.rs"]
pub mod book_capnp;
use book_capnp::{book, shop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First, create a Book message
    let mut book_message = Builder::new_default();
    {
        let mut book = book_message.init_root::<book::Builder>();
        {
            let title = "The Rust Programming Language";
            let mut title_builder = book.reborrow().init_title(title.len() as u32);
            title_builder.push_str(title);
        }
        {
            let author = "Steve Klabnik";
            let mut author_builder = book.reborrow().init_author(author.len() as u32);
            author_builder.push_str(author);
        }
        book.set_pages(500);
        book.set_publish_year(2019);
        book.set_is_available(true);
    }

    // Serialize the book to bytes
    let mut book_bytes = Vec::new();
    serialize::write_message(&mut book_bytes, &book_message)?;
    println!("Book serialized size: {} bytes", book_bytes.len());

    // Now create a Shop and add the book to it
    let mut shop_message = Builder::new_default();
    {
        let mut shop = shop_message.init_root::<shop::Builder>();
        {
            let name = "My Bookstore";
            let mut name_builder = shop.reborrow().init_name(name.len() as u32);
            name_builder.push_str(name);
        }

        // Initialize a list of books with 1 entry
        let mut books = shop.reborrow().init_books(1);

        // Read the book from bytes and copy it to the shop
        let book_message = serialize::read_message(&mut &book_bytes[..], ReaderOptions::new())?;
        let book_reader = book_message.get_root::<book::Reader>()?;

        // Copy the book data to the shop's book list
        let mut shop_book = books.get(0);
        {
            let title = book_reader.get_title()?.to_str()?;
            let mut title_builder = shop_book.reborrow().init_title(title.len() as u32);
            title_builder.push_str(title);
        }
        {
            let author = book_reader.get_author()?.to_str()?;
            let mut author_builder = shop_book.reborrow().init_author(author.len() as u32);
            author_builder.push_str(author);
        }
        shop_book.set_pages(book_reader.get_pages());
        shop_book.set_publish_year(book_reader.get_publish_year());
        shop_book.set_is_available(book_reader.get_is_available());
    }

    // Serialize the shop to bytes
    let mut shop_bytes = Vec::new();
    serialize::write_message(&mut shop_bytes, &shop_message)?;
    println!("Shop serialized size: {} bytes", shop_bytes.len());

    // Read back the shop from bytes
    let shop_message = serialize::read_message(&mut &shop_bytes[..], ReaderOptions::new())?;
    let shop_reader = shop_message.get_root::<shop::Reader>()?;

    // Verify the data
    println!("\nShop name: {}", shop_reader.get_name()?.to_str()?);
    let books = shop_reader.get_books()?;
    println!("Number of books: {}", books.len());

    // Get the first book
    let book = books.get(0);
    println!("\nFirst book details:");
    println!("Title: {}", book.get_title()?.to_str()?);
    println!("Author: {}", book.get_author()?.to_str()?);
    println!("Pages: {}", book.get_pages());
    println!("Publish Year: {}", book.get_publish_year());
    println!("Is Available: {}", book.get_is_available());

    Ok(())
}
