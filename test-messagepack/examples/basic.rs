use serde::{Serialize, Deserialize};
use serde_json;
use rmp_serde;

#[derive(Serialize, Deserialize, Debug)]
struct Fruit {
    name: String,
    price: f32,
    quantity: u32,
}

fn main() {
    let fruit = Fruit {
        name: "Apple".to_string(),
        price: 0.99,
        quantity: 10,
    };

    // JSON encoding
    let json_encoded = serde_json::to_string(&fruit).unwrap();
    println!("JSON encoded: {}", json_encoded);
    println!("JSON encoded length: {} bytes", json_encoded.len());

    // JSON decoding
    let json_decoded: Fruit = serde_json::from_str(&json_encoded).unwrap();
    println!("JSON decoded: {:?}", json_decoded);

    // MessagePack encoding
    let msgpack_encoded = rmp_serde::to_vec(&fruit).unwrap();
    println!("MessagePack encoded: {:?}", msgpack_encoded);
    println!("MessagePack encoded length: {} bytes", msgpack_encoded.len());

    // MessagePack decoding
    let msgpack_decoded: Fruit = rmp_serde::from_slice(&msgpack_encoded).unwrap();
    println!("MessagePack decoded: {:?}", msgpack_decoded);
}