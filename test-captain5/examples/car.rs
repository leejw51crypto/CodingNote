extern crate capnp;

#[path = "../src/generated/proto/book_capnp.rs"]
pub mod book_capnp;
use book_capnp::car;
use capnp::message::{Builder, ReaderOptions};
use sha2::{Digest, Sha256};

fn main() -> capnp::Result<()> {
    // Create a new message and car
    let mut message = Builder::new_default();
    {
        let mut car = message.init_root::<car::Builder>();
        {
            let mut make = car.reborrow().init_make(6); // "Toyota"
            make.push_str("Toyota");
        }
        {
            let mut model = car.reborrow().init_model(5); // "Camry"
            model.push_str("Camry");
        }
        car.set_year(2024);
        {
            let mut color = car.reborrow().init_color(6); // "Silver"
            color.push_str("Silver");
        }
        car.set_mileage(0);
        {
            let mut vin = car.reborrow().init_vin(17); // VIN length
            vin.push_str("1HGCM82633A123456");
        }
        car.set_is_electric(false);
        car.set_price(25000);
    }

    // Serialize and show info
    let mut serialized = Vec::new();

    capnp::serialize::write_message(&mut serialized, &message)?;
    let hash = calculate_hash(&serialized);
    println!("Original Car:");
    println!("Serialized size: {} bytes", serialized.len());
    println!("Hash: {}\n", hash);

    // Read the original message
    let reader = capnp::serialize::read_message(&mut &serialized[..], ReaderOptions::new())?;

    // Create a new message and use set_root to copy the car
    let mut modified_message = Builder::new_default();
    {
        let car_reader = reader.get_root::<car::Reader>()?;
        println!("Original Car Info:");
        println!("Make: {}", car_reader.get_make()?.to_str()?);
        println!("Model: {}", car_reader.get_model()?.to_str()?);
        println!("Year: {}", car_reader.get_year());
        println!("Color: {}", car_reader.get_color()?.to_str()?);
        println!("Mileage: {}", car_reader.get_mileage());
        println!("VIN: {}", car_reader.get_vin()?.to_str()?);
        println!("Is Electric: {}", car_reader.get_is_electric());
        println!("Price: ${}\n", car_reader.get_price());

        // Copy the original car and modify only the color
        modified_message.set_root(car_reader)?;
        let mut car_builder = modified_message.get_root::<car::Builder>()?;
        {
            let mut color = car_builder.reborrow().init_color(3); // "Red"
            color.push_str("Red"); // Only change the color
        }
    }

    // Serialize modified car and show info
    let mut modified_serialized = Vec::new();
    capnp::serialize::write_message(&mut modified_serialized, &modified_message)?;
    let modified_hash = calculate_hash(&modified_serialized);
    println!("Modified Car (color only):");
    println!("Serialized size: {} bytes", modified_serialized.len());
    println!("Hash: {}\n", modified_hash);

    // Show modified car info
    let modified_car = modified_message.get_root_as_reader::<car::Reader>()?;
    println!("Modified Car Info:");
    println!("Make: {}", modified_car.get_make()?.to_str()?);
    println!("Model: {}", modified_car.get_model()?.to_str()?);
    println!("Year: {}", modified_car.get_year());
    println!("Color: {}", modified_car.get_color()?.to_str()?);
    println!("Mileage: {}", modified_car.get_mileage());
    println!("VIN: {}", modified_car.get_vin()?.to_str()?);
    println!("Is Electric: {}", modified_car.get_is_electric());
    println!("Price: ${}", modified_car.get_price());

    Ok(())
}

fn calculate_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("{:x}", result)
}
