/*
Key Cap'n Proto Operations:

1. get_root<Reader>()
   - Gets a read-only view of the root structure
   - Used for reading existing messages
   - Cannot modify data through this reader

2. init_root<Builder>()
   - Creates a new writable root structure
   - Used when building a new message from scratch
   - Returns a builder for setting fields

3. set_root()
   - Transfers a structure from a Reader into a new message
   - Optimizes by moving data when possible, only copying when necessary
   - Used when you want to create a modifiable version of a message

4. get_root_as_reader()
   - Gets a read-only view of a message being built
   - Used to verify contents during construction
*/

use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;

#[path = "../src/generated/proto/book_capnp.rs"]
pub mod book_capnp;
use book_capnp::shop;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create initial shop data
    let mut initial_message = Builder::new_default();
    {
        let mut shop_builder = initial_message.init_root::<shop::Builder>();

        // Set basic shop info
        {
            let name = "Bookstore & Fruit Market";
            let mut name_builder = shop_builder.reborrow().init_name(name.len() as u32);
            name_builder.push_str(name);
        }
        {
            let address = "123 Main Street";
            let mut addr_builder = shop_builder.reborrow().init_address(address.len() as u32);
            addr_builder.push_str(address);
        }
        shop_builder.set_is_open(true);
        {
            let hours = "9:00 AM - 6:00 PM";
            let mut hours_builder = shop_builder
                .reborrow()
                .init_opening_hours(hours.len() as u32);
            hours_builder.push_str(hours);
        }
        {
            let phone = "+1-555-123-4567";
            let mut phone_builder = shop_builder
                .reborrow()
                .init_phone_number(phone.len() as u32);
            phone_builder.push_str(phone);
        }
        {
            let email = "contact@bookfruit.com";
            let mut email_builder = shop_builder.reborrow().init_email(email.len() as u32);
            email_builder.push_str(email);
        }
        shop_builder.set_rating(4.5);
        shop_builder.set_last_updated(1234567890); // Example timestamp

        // Add some books
        let mut books = shop_builder.reborrow().init_books(2);
        {
            let mut book0 = books.reborrow().get(0);
            let title = "The Rust Programming Language";
            let mut title_builder = book0.reborrow().init_title(title.len() as u32);
            title_builder.push_str(title);

            let author = "Steve Klabnik";
            let mut author_builder = book0.reborrow().init_author(author.len() as u32);
            author_builder.push_str(author);

            book0.set_pages(500);
            book0.set_publish_year(2019);
            book0.set_is_available(true);
        }
        {
            let mut book1 = books.get(1);
            let title = "Programming WebAssembly";
            let mut title_builder = book1.reborrow().init_title(title.len() as u32);
            title_builder.push_str(title);

            let author = "Kevin Hoffman";
            let mut author_builder = book1.reborrow().init_author(author.len() as u32);
            author_builder.push_str(author);

            book1.set_pages(300);
            book1.set_publish_year(2019);
            book1.set_is_available(false);
        }

        // Add some fruits
        let mut fruits = shop_builder.init_fruits(2);
        {
            let mut fruit0 = fruits.reborrow().get(0);
            let name = "Apple";
            let mut name_builder = fruit0.reborrow().init_name(name.len() as u32);
            name_builder.push_str(name);

            let color = "Red";
            let mut color_builder = fruit0.reborrow().init_color(color.len() as u32);
            color_builder.push_str(color);

            fruit0.set_weight_grams(200);
            fruit0.set_is_ripe(true);

            let variety = "Honeycrisp";
            let mut variety_builder = fruit0.init_variety(variety.len() as u32);
            variety_builder.push_str(variety);
        }
        {
            let mut fruit1 = fruits.get(1);
            let name = "Banana";
            let mut name_builder = fruit1.reborrow().init_name(name.len() as u32);
            name_builder.push_str(name);

            let color = "Yellow";
            let mut color_builder = fruit1.reborrow().init_color(color.len() as u32);
            color_builder.push_str(color);

            fruit1.set_weight_grams(150);
            fruit1.set_is_ripe(true);

            let variety = "Cavendish";
            let mut variety_builder = fruit1.init_variety(variety.len() as u32);
            variety_builder.push_str(variety);
        }
    }

    // Serialize to bytes
    let mut serialized_data = Vec::new();
    serialize::write_message(&mut serialized_data, &initial_message)?;
    println!(
        "Original serialized data size: {} bytes",
        serialized_data.len()
    );

    // Deserialize and modify the data
    let message =
        serialize::read_message_from_flat_slice(&mut &serialized_data[..], ReaderOptions::new())?;
    let shop_reader = message.get_root::<shop::Reader>()?;

    // Create new message and copy all data using set_root
    let mut modified_message = Builder::new_default();
    modified_message.set_root(shop_reader)?;
    // Print original values before modification
    {
        let shop = modified_message.get_root_as_reader::<shop::Reader>()?;
        println!("\nBefore modification:");
        println!("Is Open: {}", shop.get_is_open());
        println!("Rating: {}", shop.get_rating());
        println!("Last Updated: {}", shop.get_last_updated());
    }

    // Modify specific fields while keeping other data intact
    {
        let mut new_shop = modified_message.get_root::<shop::Builder>()?;
        new_shop.set_rating(4.8); // Update rating to 4.8
    }

    // Print new values after modification
    {
        let shop = modified_message.get_root_as_reader::<shop::Reader>()?;
        println!("\nAfter modification:");
        println!("Is Open: {}", shop.get_is_open());
        println!("Rating: {}", shop.get_rating());
        println!("Last Updated: {}", shop.get_last_updated());
    }

    // Serialize modified data
    let mut modified_data = Vec::new();
    serialize::write_message(&mut modified_data, &modified_message)?;
    println!(
        "Modified serialized data size: {} bytes",
        modified_data.len()
    );

    // Read back and verify changes
    let reader_message = serialize::read_message(&mut &modified_data[..], ReaderOptions::new())?;
    let shop_reader = reader_message.get_root::<shop::Reader>()?;

    println!("\nVerifying modified shop:");
    println!("Name: {}", shop_reader.get_name()?.to_str()?);
    println!("Address: {}", shop_reader.get_address()?.to_str()?);
    println!("Is Open: {}", shop_reader.get_is_open());
    println!("Rating: {}", shop_reader.get_rating());
    println!("Number of books: {}", shop_reader.get_books()?.len());
    println!("Number of fruits: {}", shop_reader.get_fruits()?.len());

    Ok(())
}
