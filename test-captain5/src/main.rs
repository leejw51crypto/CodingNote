use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use chrono::Local;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::io::{self, BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
#[cfg(unix)]
// Include the generated code
mod generated;
use generated::proto::book_capnp;

fn create_sample_book() -> Builder<capnp::message::HeapAllocator> {
    let mut message = Builder::new_default();
    let mut book = message.init_root::<book_capnp::book::Builder>();

    book.set_title("The Rust Programming Language".into());
    book.set_author("Steve Klabnik and Carol Nichols".into());
    book.set_pages(500);
    book.set_publish_year(2019);
    book.set_is_available(true);

    {
        let mut genres = book.init_genres(2);
        genres.set(0, "Programming".into());
        genres.set(1, "Computer Science".into());
    }

    message
}

fn create_sample_fruit() -> Builder<capnp::message::HeapAllocator> {
    let mut message = Builder::new_default();
    let mut fruit = message.init_root::<book_capnp::fruit::Builder>();

    fruit.set_name("Apple".into());
    fruit.set_color("Red".into());
    fruit.set_weight_grams(200);
    fruit.set_is_ripe(true);
    fruit.set_variety("Honeycrisp".into());

    message
}

fn serialize_message(message: &Builder<capnp::message::HeapAllocator>) -> Vec<u8> {
    let mut serialized_data = Vec::new();
    serialize::write_message(&mut serialized_data, message).expect("Failed to serialize message");
    serialized_data
}

fn deserialize_and_print(data: &[u8]) {
    let reader = serialize::read_message(&mut &data[..], ReaderOptions::new())
        .expect("Failed to read message");

    let book_reader = reader
        .get_root::<book_capnp::book::Reader>()
        .expect("Failed to get root");

    println!("Book details:");
    println!(
        "Title: {}",
        book_reader.get_title().unwrap().to_str().unwrap()
    );
    println!(
        "Author: {}",
        book_reader.get_author().unwrap().to_str().unwrap()
    );
    println!("Pages: {}", book_reader.get_pages());
    println!("Publish Year: {}", book_reader.get_publish_year());
    println!("Is Available: {}", book_reader.get_is_available());
    println!(
        "Genres: {:?}",
        book_reader
            .get_genres()
            .unwrap()
            .iter()
            .map(|g| g.unwrap().to_str().unwrap())
            .collect::<Vec<_>>()
    );
}

fn modify_book_title(data: &[u8], new_title: &str) -> Vec<u8> {
    let mut message = Builder::new_default();
    {
        let reader = serialize::read_message(&mut &data[..], ReaderOptions::new())
            .expect("Failed to read message");
        let old_book = reader
            .get_root::<book_capnp::book::Reader>()
            .expect("Failed to get root");

        let mut new_book = message.init_root::<book_capnp::book::Builder>();
        new_book.set_title(new_title.into());
        new_book.set_author(old_book.get_author().unwrap());
        new_book.set_pages(old_book.get_pages());
        new_book.set_publish_year(old_book.get_publish_year());
        new_book.set_is_available(old_book.get_is_available());

        let old_genres = old_book.get_genres().unwrap();
        let mut new_genres = new_book.init_genres(old_genres.len());
        for (i, genre) in old_genres.iter().enumerate() {
            new_genres.set(i as u32, genre.unwrap());
        }
    }

    serialize_message(&message)
}

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

fn stream_writer(stream: TcpStream, running: Arc<AtomicBool>) -> std::result::Result<(), BoxError> {
    let mut rng = rand::thread_rng();
    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
    let mut writer = BufWriter::new(stream);
    let mut counter = 0;
    println!("Starting to send messages (press 'q' and Enter to quit)...");

    while running.load(Ordering::Relaxed) {
        counter += 1;
        let current_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        let mut message = Builder::new_default();
        let root = message.init_root::<book_capnp::message::Builder>();

        if rng.random_bool(0.5) {
            let mut book = root.init_book();
            let title = format!("Book #{} at {}", counter, current_time);
            book.set_title(title.as_str().into());
            book.set_author("Continuous Writer".into());
            book.set_pages(100 + counter);
            book.set_publish_year(2024);
            book.set_is_available(true);
            {
                let mut genres = book.init_genres(2);
                genres.set(0, "Streaming".into());
                genres.set(1, "Test".into());
            }
            println!("Client: Sending Book #{} at {}", counter, current_time);
        } else {
            let mut fruit = root.init_fruit();
            let name = format!("Fruit #{}", counter);
            fruit.set_name(name.as_str().into());
            fruit.set_color(["Red", "Green", "Yellow"][(counter % 3) as usize].into());
            fruit.set_weight_grams(rng.random_range(100..=500));
            fruit.set_is_ripe(rng.random_bool(0.7));
            fruit.set_variety("Mixed".into());
            println!("Client: Sending Fruit #{} at {}", counter, current_time);
        }

        match serialize::write_message(&mut writer, &message) {
            Ok(_) => {
                if let Err(e) = writer.flush() {
                    eprintln!("Failed to flush writer: {}", e);
                    break;
                }
            }
            Err(e) => {
                eprintln!("Failed to send message: {}", e);
                break;
            }
        }

        thread::sleep(Duration::from_secs(1));
    }

    println!("Writer shutting down");
    Ok(())
}

fn stream_reader(stream: TcpStream, running: Arc<AtomicBool>) -> std::result::Result<(), BoxError> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut reader = BufReader::new(stream);
    println!("Starting to read messages...");

    while running.load(Ordering::Relaxed) {
        match serialize::read_message(&mut reader, ReaderOptions::new()) {
            Ok(message) => {
                let root = message.get_root::<book_capnp::message::Reader>()?;

                match root.which()? {
                    book_capnp::message::Which::Book(Ok(book)) => {
                        println!("\nReceived Book:");
                        println!("Title: {}", book.get_title()?.to_str()?);
                        println!("Author: {}", book.get_author()?.to_str()?);
                        println!("Pages: {}", book.get_pages());
                        println!("Publish Year: {}", book.get_publish_year());
                        println!("Is Available: {}", book.get_is_available());
                        println!(
                            "Genres: {:?}",
                            book.get_genres()?
                                .iter()
                                .map(|g| g.unwrap().to_str().unwrap())
                                .collect::<Vec<_>>()
                        );
                    }
                    book_capnp::message::Which::Book(Err(e)) => {
                        eprintln!("Error reading book: {}", e);
                    }
                    book_capnp::message::Which::Fruit(Ok(fruit)) => {
                        println!("\nReceived Fruit:");
                        println!("Name: {}", fruit.get_name()?.to_str()?);
                        println!("Color: {}", fruit.get_color()?.to_str()?);
                        println!("Weight: {}g", fruit.get_weight_grams());
                        println!("Is Ripe: {}", fruit.get_is_ripe());
                        println!("Variety: {}", fruit.get_variety()?.to_str()?);
                    }
                    book_capnp::message::Which::Fruit(Err(e)) => {
                        eprintln!("Error reading fruit: {}", e);
                    }
                }
            }
            Err(e) => {
                if !running.load(Ordering::Relaxed) {
                    println!("Reader shutting down normally");
                    break;
                }
                if running.load(Ordering::Relaxed) {
                    if e.kind == ::capnp::ErrorKind::Failed {
                        println!("Connection closed by writer");
                        break;
                    } else {
                        eprintln!("Read error: {}", e);
                    }
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    }

    println!("Reader shutting down");
    Ok(())
}

fn create_listener() -> std::io::Result<TcpListener> {
    use std::net::SocketAddr;

    let addr: SocketAddr = "127.0.0.1:9020".parse().unwrap();

    // Try to bind and check if port is available
    match TcpListener::bind(addr) {
        Ok(listener) => {
            // Set SO_REUSEADDR on Unix systems
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                let sock_fd = listener.as_raw_fd();
                unsafe {
                    let optval: libc::c_int = 1;
                    libc::setsockopt(
                        sock_fd,
                        libc::SOL_SOCKET,
                        libc::SO_REUSEADDR,
                        &optval as *const _ as *const libc::c_void,
                        std::mem::size_of_val(&optval) as libc::socklen_t,
                    );
                }
            }
            Ok(listener)
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::AddrInUse {
                eprintln!("Error: Port 9020 is already in use. Please make sure no other instance is running.");
                std::process::exit(1);
            }
            Err(e)
        }
    }
}

fn main() -> std::result::Result<(), BoxError> {
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Start server thread first
    let running_server = running.clone();
    let server_ready = Arc::new(AtomicBool::new(false));
    let server_ready_clone = server_ready.clone();

    let server_thread = thread::spawn(move || -> std::result::Result<(), BoxError> {
        let listener = create_listener()?;
        println!("Server listening on 127.0.0.1:9020");
        server_ready_clone.store(true, Ordering::Release);

        match listener.incoming().next() {
            Some(Ok(stream)) => {
                println!("Server: New connection established!");
                stream_reader(stream, running_server)?;
            }
            Some(Err(e)) => {
                eprintln!("Server: Failed to accept connection: {}", e);
            }
            None => {
                eprintln!("Server: Listener closed");
            }
        }
        Ok(())
    });

    // Wait for server to be ready
    while !server_ready.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(10));
    }

    // Client thread
    let running_client = running.clone();
    let client_thread = thread::spawn(move || -> std::result::Result<(), BoxError> {
        // Try to connect with retries
        let mut retries = 5;
        let mut stream = None;

        while retries > 0 {
            match TcpStream::connect("127.0.0.1:9020") {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => {
                    if retries > 1 {
                        eprintln!("Client: Connection attempt failed, retrying... ({})", e);
                        thread::sleep(Duration::from_millis(100));
                        retries -= 1;
                    } else {
                        return Err(Box::new(e));
                    }
                }
            }
        }

        let stream = stream.ok_or_else(|| io::Error::other("Failed to connect after retries"))?;
        println!("Client: Connected to server");
        stream_writer(stream, running_client)
    });

    // Input handling thread
    let input_thread = thread::spawn(move || {
        println!("Press 'q' and Enter to quit");
        let mut input = String::new();
        while input.trim() != "q" {
            input.clear();
            if io::stdin().read_line(&mut input).is_ok() && input.trim() == "q" {
                println!("Quit signal received, shutting down...");
                running_clone.store(false, Ordering::Relaxed);
                break;
            }
        }
    });

    // Wait for input thread to complete (user entered 'q')
    input_thread.join().expect("Input thread panicked");

    // Wait for client and server to shut down
    if let Err(e) = client_thread.join().expect("Client thread panicked") {
        eprintln!("Client error: {}", e);
    }
    if let Err(e) = server_thread.join().expect("Server thread panicked") {
        eprintln!("Server error: {}", e);
    }

    println!("Program terminated successfully");
    Ok(())
}
