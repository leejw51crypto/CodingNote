use capnp::message::{Builder, ReaderOptions};
use capnp::serialize::{read_message, write_message};
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[path = "../src/generated/proto/book_capnp.rs"]
pub mod book_capnp;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some sample books
    let mut message = Builder::new_default();
    {
        let mut book_list = message.init_root::<book_capnp::book_list::Builder>();
        let mut books = book_list.init_books(2);

        // Book 1
        {
            let mut book1 = books.reborrow().get(0);
            book1
                .reborrow()
                .set_title("The Rust Programming Language".into());
            book1
                .reborrow()
                .set_author("Steve Klabnik and Carol Nichols".into());
            book1.set_pages(500);
            book1.set_publish_year(2019);
            book1.set_is_available(true);
            {
                let mut genres = book1.init_genres(2);
                genres.reborrow().set(0, "Programming".into());
                genres.set(1, "Computer Science".into());
            }
        }

        // Book 2
        {
            let mut book2 = books.get(1);
            book2.reborrow().set_title("Zero to One".into());
            book2.reborrow().set_author("Peter Thiel".into());
            book2.set_pages(224);
            book2.set_publish_year(2014);
            book2.set_is_available(true);
            {
                let mut genres = book2.init_genres(2);
                genres.reborrow().set(0, "Business".into());
                genres.set(1, "Entrepreneurship".into());
            }
        }
    }

    // Write to file
    println!("Writing books to file...");
    {
        let file = File::create("books.bin")?;
        let mut writer = BufWriter::new(file);
        write_message(&mut writer, &message)?;
    }

    // Read from file
    println!("\nReading books from file...");
    {
        let file = File::open("books.bin")?;
        let reader = BufReader::new(file);
        let message_reader = read_message(reader, ReaderOptions::new())?;
        let book_list = message_reader.get_root::<book_capnp::book_list::Reader>()?;

        for book in book_list.get_books()?.iter() {
            println!("\nBook:");
            println!("Title: {}", book.get_title()?.to_str()?);
            println!("Author: {}", book.get_author()?.to_str()?);
            println!("Pages: {}", book.get_pages());
            println!("Published: {}", book.get_publish_year());
            println!("Available: {}", book.get_is_available());
            print!("Genres: ");
            for genre in book.get_genres()?.iter() {
                print!("{} ", genre?.to_str()?);
            }
            println!();
        }
    }

    Ok(())
}
