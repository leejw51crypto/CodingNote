use anyhow::Result;
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;

pub mod fruit_capnp {
    include!(concat!(env!("OUT_DIR"), "/proto/fruit_capnp.rs"));
}

// init_root
// message
// get_root

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
    }

    let mut encoded_message = Vec::new();
    serialize::write_message(&mut encoded_message, &message)?;
    Ok(encoded_message)
}

fn decode_fruit(encoded_message: &[u8]) -> Result<()> {
    let reader = serialize::read_message(&mut &encoded_message[..], ReaderOptions::new())?;
    let fruit = reader.get_root::<fruit_capnp::fruit::Reader>()?;

    println!("Name: {}", fruit.get_name()?.to_string()?);
    println!("Weight: {}", fruit.get_weight());
    println!("Is Ripe: {}", fruit.get_is_ripe());

    let colors = fruit.get_colors()?;
    println!("Colors:");
    for color in colors {
        println!("- {}", color?.to_string()?);
    }

    Ok(())
}

fn main() -> Result<()> {
    let encoded_message = encode_fruit()?;
    println!("{} bytes", encoded_message.len());
    println!("hex {}", hex::encode(&encoded_message));
    decode_fruit(&encoded_message)?;
    Ok(())
}
