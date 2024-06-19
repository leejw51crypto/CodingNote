use anyhow::Result;
use chrono::Utc;
use prost::Message;

pub mod myproto {
    include!(concat!(env!("OUT_DIR"), "/myproto.rs"));
}

use myproto::Fruit;

fn encode_fruit() -> Result<Vec<u8>> {
    let large_data_size = crate::definition::TEST_SIZE; 
    let large_data = vec![0u8; large_data_size];

    let fruit = Fruit {
        name: "Apple".to_string(),
        weight: 150,
        is_ripe: true,
        colors: vec!["Red".to_string(), "Green".to_string()],
        large_data,
    };

    let start_time = Utc::now();
    let buf = fruit.encode_to_vec();
    let end_time = Utc::now();
    let encoding_time = end_time - start_time;
    println!("Encoding time: {} micro-seconds", encoding_time.num_microseconds().unwrap());

    Ok(buf)
}

fn decode_fruit(encoded_message: &[u8]) -> Result<()> {
    let start_time = Utc::now();
    let fruit = Fruit::decode(encoded_message)?;
    let end_time = Utc::now();
    let decoding_time = end_time - start_time;
    println!("Decoding time: {} micro-seconds", decoding_time.num_microseconds().unwrap());

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
    println!("protobuf");

    let encoded_message: Vec<u8> = encode_fruit()?;

    println!("{} bytes", encoded_message.len());

    decode_fruit(&encoded_message)?;

    Ok(())
}
