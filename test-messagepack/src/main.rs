use anyhow::Result;
use fake::faker::address::en::StreetName;
use fake::faker::company::en::Buzzword;
use fake::{Dummy, Fake, Faker};
use rmp_serde;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Serialize, Deserialize, Debug, Dummy)]
struct ItemInfo {
    #[dummy(faker = "StreetName()")]
    description: String,
    #[dummy(faker = "1..5")]
    rating: u32,
}

#[derive(Serialize, Deserialize, Debug, Dummy)]
struct Item {
    #[dummy(faker = "Buzzword()")]
    name: String,
    #[dummy(faker = "1000.0..2000.0")]
    price: f32,
    #[dummy(faker = "10..20")]
    quantity: u32,
    #[dummy]
    info: ItemInfo,
}

fn use_msgpack_encoded(msgpack_encoded: &[u8]) -> Result<()> {
    // Deserialize MessagePack-encoded data into a serde_json::Value
    let mut reader = std::io::Cursor::new(msgpack_encoded);
    let value: serde_json::Value = rmp_serde::from_read(&mut reader)?;

    // Access the deserialized data as a serde_json::Value
    println!("Deserialized MessagePack data: {}", value);

    // You can access specific fields using the serde_json::Value API
    if let Some(name) = value.get("name") {
        println!("Name: {}", name);
    }

    if let Some(price) = value.get("price") {
        println!("Price: {}", price);
    }

    // ... (access other fields as needed)

    Ok(())
}

fn main() -> Result<()> {
    let items: Vec<Item> = (0..10)
        .map(|_| {
            let item: Item = Faker.fake();
            item
        })
        .collect();

    for item in &items {
        // JSON encoding
        let json_encoded = serde_json::to_string(&item)?;
        println!("JSON encoded: {}", json_encoded);
        println!("JSON encoded length: {} bytes", json_encoded.len());

        // JSON decoding
        let json_decoded: Item = serde_json::from_str(&json_encoded)?;
        println!("JSON decoded: {:?}", json_decoded);

        // MessagePack encoding
        let msgpack_encoded = rmp_serde::to_vec(&item)?;
        println!("MessagePack encoded: {}", hex::encode(&msgpack_encoded));
        println!(
            "MessagePack encoded length: {} bytes",
            msgpack_encoded.len()
        );

        // MessagePack decoding
        let msgpack_decoded: Item = rmp_serde::from_slice(&msgpack_encoded)?;
        let json2 = serde_json::to_string(&msgpack_decoded)?;
        println!("MessagePack decoded: {:?}", msgpack_decoded);
        println!("JSON encoded from MessagePack: {}", json2);
        use_msgpack_encoded(&msgpack_encoded)?;

        println!();
    }
    Ok(())
}
