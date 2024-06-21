use anyhow::Result;
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize_packed;
use chrono::Utc;
pub mod fruit_capnp {
    include!(concat!(env!("OUT_DIR"), "/proto/fruit_capnp.rs"));
}

fn encode_fruit() -> Result<Vec<u8>> {
    let mut message = Builder::new_default();

    {
        let mut fruit = message.init_root::<fruit_capnp::fruit::Builder>();
        fruit.set_name("Apple");
        fruit.set_weight(150);
        fruit.set_is_ripe(true);

        let mut colors = fruit.reborrow().init_colors(2);
        colors.set(0, "Red");
        colors.set(1, "Green");

        // Generate large data for stress testing
        let large_data_size = crate::definition::TEST_SIZE;
        let large_data = vec![0u8; large_data_size];
        fruit.set_large_data(&large_data);
    }

    let mut encoded_message = Vec::new();
    let start_time = Utc::now();
    serialize_packed::write_message(&mut encoded_message, &message)?;
    let end_time = Utc::now();
    let encoding_time = end_time - start_time;

    println!(
        "Encoding time: {} micro-seconds",
        encoding_time.num_microseconds().unwrap()
    );

    Ok(encoded_message)
}

fn decode_fruit(encoded_message: &[u8]) -> Result<()> {
    let start_time = Utc::now();

    let reader = serialize_packed::read_message(&mut &encoded_message[..], ReaderOptions::new())?;

    let fruit = reader.get_root::<fruit_capnp::fruit::Reader>()?;
    let end_time = Utc::now();

    let decoding_time = end_time - start_time;

    println!(
        "Decoding time: {} micro-seconds",
        decoding_time.num_microseconds().unwrap()
    );

    println!("Name: {}", fruit.get_name()?.to_string()?);
    println!("Weight: {}", fruit.get_weight());
    println!("Is Ripe: {}", fruit.get_is_ripe());

    let colors = fruit.get_colors()?;
    println!("Colors:");
    for color in colors {
        println!("- {}", color?.to_string()?);
    }

    println!("Large Data Size: {} bytes", fruit.get_large_data()?.len());

    Ok(())
}

pub fn main() -> Result<()> {
    println!("capnp");
    let encoded_message = encode_fruit()?;
    println!("{} bytes", encoded_message.len());
    if crate::definition::DISPLAY_HEX {
        println!("{}", hex::encode(&encoded_message));
    }

    decode_fruit(&encoded_message)?;

    Ok(())
}
