use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use crossbeam_channel::{bounded, select, Receiver, Sender};
use rand::Rng;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

#[path = "../src/generated/proto/book_capnp.rs"]
pub mod book_capnp;
use book_capnp::book;

// Generate a random book
fn generate_random_book() -> Builder<capnp::message::HeapAllocator> {
    let mut rng = rand::thread_rng();
    let mut message = Builder::new_default();
    {
        let mut book_builder = message.init_root::<book::Builder>();

        // Random title
        let titles = [
            "Rust in Action",
            "Programming Rust",
            "Zero to Production",
            "Async Rust",
        ];
        let title = titles[rng.gen_range(0..titles.len())];
        let mut title_builder = book_builder.reborrow().init_title(title.len() as u32);
        title_builder.push_str(title);

        // Random author
        let authors = ["John Doe", "Jane Smith", "Bob Wilson", "Alice Brown"];
        let author = authors[rng.gen_range(0..authors.len())];
        let mut author_builder = book_builder.reborrow().init_author(author.len() as u32);
        author_builder.push_str(author);

        // Random pages
        book_builder.set_pages(rng.gen_range(100..1000));

        // Random year
        book_builder.set_publish_year(rng.gen_range(2000..2024) as u16);

        // Random availability
        book_builder.set_is_available(rng.gen_bool(0.7));

        // Random genres
        let genres = [
            "Programming",
            "Technology",
            "Computer Science",
            "Software Engineering",
        ];
        let num_genres = rng.gen_range(1..3);
        let mut genre_list = book_builder.init_genres(num_genres as u32);
        for i in 0..num_genres {
            let genre = genres[rng.gen_range(0..genres.len())];
            let mut genre_text = genre_list.reborrow();
            let mut text = genre_text.init(i as u32, genre.len() as u32);
            text.push_str(genre);
        }
    }
    message
}

// Producer function
fn producer(tx: Sender<Vec<u8>>) {
    loop {
        // Generate a random book
        let message = generate_random_book();

        // Serialize the book
        let mut serialized_data = Vec::new();
        serialize::write_message(&mut serialized_data, &message).unwrap();

        // Send the serialized data
        if tx.send(serialized_data).is_err() {
            break;
        }

        // Sleep for a bit
        thread::sleep(Duration::from_secs(1));
    }
}

// Consumer function
fn consumer(rx: Receiver<Vec<u8>>) {
    loop {
        select! {
            recv(rx) -> msg => {
                match msg {
                    Ok(data) => {
                        // Deserialize and display the book
                        if let Ok(message) = serialize::read_message_from_flat_slice(
                            &mut &data[..],
                            ReaderOptions::new(),
                        ) {
                            if let Ok(book) = message.get_root::<book::Reader>() {
                                println!("\nReceived book:");
                                if let Ok(title) = book.get_title() {
                                    println!("Title: {}", title.to_str().unwrap_or("Invalid UTF-8"));
                                }
                                if let Ok(author) = book.get_author() {
                                    println!("Author: {}", author.to_str().unwrap_or("Invalid UTF-8"));
                                }
                                println!("Pages: {}", book.get_pages());
                                println!("Year: {}", book.get_publish_year());
                                println!("Available: {}", book.get_is_available());
                                if let Ok(genres) = book.get_genres() {
                                    print!("Genres: ");
                                    for i in 0..genres.len() {
                                        if let Ok(genre) = genres.get(i) {
                                            if let Ok(genre_str) = genre.to_str() {
                                                print!("{} ", genre_str);
                                            }
                                        }
                                    }
                                    println!();
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        }
    }
}

fn main() {
    // Create a bounded channel
    let (tx, rx) = bounded(10);

    // Spawn producer thread
    let producer_handle = thread::spawn(move || {
        producer(tx);
    });

    // Spawn consumer thread
    let consumer_handle = thread::spawn(move || {
        consumer(rx);
    });

    // Wait for user input to quit
    println!("Press 'q' to quit...");
    loop {
        let mut input = String::new();
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut input).unwrap();

        if input.trim().eq_ignore_ascii_case("q") {
            break;
        }
    }

    // Producer and consumer will exit when the channel is dropped
    drop(producer_handle);
    drop(consumer_handle);
}
