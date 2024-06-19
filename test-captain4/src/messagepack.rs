use anyhow::Result;
use chrono::Utc;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct Fruit {
    name: String,
    weight: u16,
    is_ripe: bool,
    colors: Vec<String>,
    large_data: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Fruit2 {
    name: String,
    weight: u16,
    is_ripe: bool,
}

fn encode_fruit() -> Result<Vec<u8>> {
    let large_data_size = 10 * 1024 * 1024; // 10 MB
    let mut large_data = vec![0u8; large_data_size];

    let fruit = Fruit {
        name: "Apple".to_string(),
        weight: 150,
        is_ripe: true,
        colors: vec!["Red".to_string(), "Green".to_string()],
        large_data,
    };

    let start_time = Utc::now();
    let buf = rmp_serde::encode::to_vec(&fruit)?;
    let end_time = Utc::now();
    let encoding_time = end_time - start_time;
    println!("Encoding time: {} ms", encoding_time.num_milliseconds());

    Ok(buf)
}

fn decode_fruit(encoded_message: &[u8]) -> Result<()> {
    let start_time = Utc::now();
    let fruit: Fruit = rmp_serde::from_slice(encoded_message)?;
    let end_time = Utc::now();
    let decoding_time = end_time - start_time;
    println!("Decoding time: {} ms", decoding_time.num_milliseconds());

    println!("Name: {}", fruit.name);
    println!("Weight: {}", fruit.weight);
    println!("Is Ripe: {}", fruit.is_ripe);

    println!("Colors:");
    for color in &fruit.colors {
        println!("- {}", color);
    }

    Ok(())
}

pub fn main() -> Result<()> {
    println!("messagepack");

    let encoded_message: Vec<u8> = encode_fruit()?;

    println!("{} bytes", encoded_message.len());
    //println!("hex {}", hex::encode(&encoded_message));

    decode_fruit(&encoded_message)?;

    let value: Fruit = rmp_serde::from_slice(&encoded_message)?;
    //println!("decoded2 {}", serde_json::to_string_pretty(&value)?,);
    Ok(())
}
