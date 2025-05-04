use fake::faker::company::en::CompanyName;
use fake::faker::lorem::en::Paragraph;
use fake::faker::name::en::Name;
use fake::{Fake, Faker};
use flexbuffers::{Builder, BuilderOptions, Reader};
use rand::Rng;
use serde::{Deserialize, Serialize};

// Define a Book struct with Serde support
#[derive(Serialize, Deserialize, Debug)]
struct Book {
    title: String,
    author: String,
    publisher: String,
    year: u32,
    pages: u32,
    description: String,
    price: f64,
    in_stock: bool,
}

fn main() {
    // Create fake book data
    let mut rng = rand::thread_rng();

    let book = Book {
        title: format!(
            "The {}",
            Paragraph(1..2)
                .fake::<String>()
                .split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ")
        ),
        author: Name().fake(),
        publisher: CompanyName().fake(),
        year: rng.gen_range(1950..2023),
        pages: rng.gen_range(50..1200),
        description: Paragraph(1..3).fake(),
        price: (rng.gen_range(599..3999) as f64) / 100.0,
        in_stock: rng.gen_bool(0.7),
    };

    println!("==== Flexbuffers Human-Readable Feature Demo ====\n");
    println!("Generated Book:");
    println!("  Title: {}", book.title);
    println!("  Author: {}", book.author);
    println!("  Publisher: {}", book.publisher);
    println!("  Year: {}", book.year);
    println!("  Pages: {}", book.pages);
    println!("  Price: ${:.2}", book.price);
    println!("  In Stock: {}", book.in_stock);
    println!("  Description: {}\n", book.description);

    // Standard serialization with human-readable features enabled via Cargo.toml
    let human_readable = flexbuffers::to_vec(&book).unwrap();
    println!(
        "Human-readable serialized size: {} bytes",
        human_readable.len()
    );

    // Print first 64 bytes of the human-readable buffer as hex for visualization
    println!("Human-readable buffer preview (first 64 bytes as hex):");
    for (i, byte) in human_readable.iter().take(64).enumerate() {
        print!("{:02x} ", byte);
        if (i + 1) % 8 == 0 {
            println!();
        }
    }
    println!("...");

    // Try to print any ASCII characters in the buffer
    println!("Human-readable buffer preview (first 64 bytes as ASCII where possible):");
    for byte in human_readable.iter().take(64) {
        if *byte >= 32 && *byte <= 126 {
            print!("{}", *byte as char);
        } else {
            print!("·");
        }
    }
    println!("...\n");

    // Deserialize with human-readable
    let reader = Reader::get_root(human_readable.as_slice()).unwrap();
    println!("FlexBuffer Type: {:?}", reader.flexbuffer_type());

    // Access fields using the reader
    let map = reader.as_map();
    println!("Accessing title: {}", map.idx("title").as_str());
    println!("Accessing year: {}", map.idx("year").as_u32());
    println!("Accessing price: ${:.2}", map.idx("price").as_f64());
    println!("Accessing in_stock: {}", map.idx("in_stock").as_bool());

    // Deserialize back to struct
    let decoded: Book = flexbuffers::from_slice(human_readable.as_slice()).unwrap();
    println!("\nSuccessfully decoded back to Book struct!");

    println!("\n==== Human-Readable Feature Explanation ====");
    println!("The human-readable feature in flexbuffers affects how data is serialized.");
    println!("With human_readable enabled (via Cargo.toml features), you can see that:");
    println!("1. String keys like 'title', 'author', etc. are clearly visible in the buffer");
    println!("2. String values (book title, author name, etc.) are also clearly visible");
    println!("3. This makes debugging easier when inspecting raw buffer contents");
    println!("4. Numerical values (year, pages, price) are still encoded in binary format");
    println!();
    println!("The human-readable feature is useful during development for:");
    println!("- Easier debugging by inspecting raw buffer contents");
    println!("- Better interoperability with other systems");
    println!("- Simpler data format inspection and validation");
}
