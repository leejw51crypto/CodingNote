use anyhow::Result;
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize_packed;
use chrono::Utc;

pub mod vegetable_capnp {
    include!(concat!(env!("OUT_DIR"), "/proto/vegetable_capnp.rs"));
}

fn encode_vegetable() -> Result<Vec<u8>> {
    let mut message = Builder::new_default();
    let mut vegetable = message.init_root::<vegetable_capnp::vegetable::Builder>();
    
    populate_vegetable(&mut vegetable)?;

    let mut encoded_message = Vec::new();
    let start_time = Utc::now();
    serialize_packed::write_message(&mut encoded_message, &message)?;
    let encoding_time = Utc::now() - start_time;

    println!("Encoding time: {} microseconds", encoding_time.num_microseconds().unwrap());
    println!("Encoded message size: {} bytes", encoded_message.len());

    Ok(encoded_message)
}

fn populate_vegetable(vegetable: &mut vegetable_capnp::vegetable::Builder) -> Result<()> {
    let mut details = vegetable.reborrow().init_details();
    details.set_name("Carrot");
    details.set_scientific_name("Daucus carota");
    details.set_color("Orange");
    details.set_shape("Conical");
    details.set_growth_time(90);
    details.set_preferred_soil_p_h(6.0);
    details.set_harvest_method("Pull from ground");

    let mut nutrition = details.reborrow().init_nutrition();
    nutrition.set_calories(41);
    nutrition.set_protein(0.9);
    nutrition.set_fiber(2.8);

    let mut vitamins = nutrition.reborrow().init_vitamins(1);
    let mut vitamin = vitamins.reborrow().get(0);
    vitamin.set_name("Vitamin A");
    vitamin.set_amount(835.0);
    vitamin.set_unit("IU");

    let mut minerals = nutrition.reborrow().init_minerals(1);
    minerals.set(0, "Potassium");

    vegetable.set_id(1);
    vegetable.set_in_season(true);
    
    let mut plant_date = vegetable.reborrow().init_plant_date();
    plant_date.set_year(2024);
    plant_date.set_month(3);
    plant_date.set_day(15);

    let mut harvest_date = vegetable.reborrow().init_expected_harvest_date();
    harvest_date.set_year(2024);
    harvest_date.set_month(6);
    harvest_date.set_day(13);

    vegetable.set_price(1.99);
    vegetable.set_quantity(100);
    vegetable.set_organic(true);
    vegetable.set_supplier("Local Farm");

    let mut tags = vegetable.reborrow().init_tags(2);
    tags.set(0, "Root");
    tags.set(1, "Orange");

    let mut last_watered = vegetable.reborrow().init_last_watered();
    last_watered.set_hour(8);
    last_watered.set_minute(30);
    last_watered.set_second(0);

    // Generate large data for stress testing
    let large_data_size = test_captain4::definition::TEST_SIZE; // Adjust as needed
    let large_data = vec![0u8; large_data_size];
    vegetable.set_image(&large_data);

    Ok(())
}

fn decode_vegetable(encoded_message: &[u8]) -> Result<()> {
    let start_time = Utc::now();
    let reader = serialize_packed::read_message(&mut &encoded_message[..], ReaderOptions::new())?;
    let vegetable = reader.get_root::<vegetable_capnp::vegetable::Reader>()?;
    let decoding_time = Utc::now() - start_time;

    println!("Decoding time: {} microseconds", decoding_time.num_microseconds().unwrap());

    print_vegetable_info(&vegetable)?;

    Ok(())
}

fn print_vegetable_info(vegetable: &vegetable_capnp::vegetable::Reader) -> Result<()> {
    let details = vegetable.get_details()?;
    println!("Basic Information:");
    println!("  Name: {}", details.get_name()?.to_string()?);
    println!("  Scientific Name: {}", details.get_scientific_name()?.to_string()?);
    println!("  Color: {}", details.get_color()?.to_string()?);
    println!("  Shape: {}", details.get_shape()?.to_string()?);
    println!("  Growth Time: {} days", details.get_growth_time());
    println!("  Preferred Soil pH: {}", details.get_preferred_soil_p_h());
    println!("  Harvest Method: {}", details.get_harvest_method()?.to_string()?);

    let nutrition = details.get_nutrition()?;
    println!("\nNutrition Information:");
    println!("  Calories: {}", nutrition.get_calories());
    println!("  Protein: {}g", nutrition.get_protein());
    println!("  Fiber: {}g", nutrition.get_fiber());

    println!("  Vitamins:");
    for vitamin in nutrition.get_vitamins()?.iter() {
        println!("    - {}: {} {}", vitamin.get_name()?.to_string()?, vitamin.get_amount(), vitamin.get_unit()?.to_string()?);
    }

    println!("  Minerals:");
    for mineral in nutrition.get_minerals()?.iter() {
        println!("    - {}", mineral?.to_string()?);
    }

    println!("\nAdditional Information:");
    println!("  ID: {}", vegetable.get_id());
    println!("  In Season: {}", vegetable.get_in_season());
    
    let plant_date = vegetable.get_plant_date()?;
    println!("  Plant Date: {}-{:02}-{:02}", plant_date.get_year(), plant_date.get_month(), plant_date.get_day());

    let harvest_date = vegetable.get_expected_harvest_date()?;
    println!("  Expected Harvest Date: {}-{:02}-{:02}", harvest_date.get_year(), harvest_date.get_month(), harvest_date.get_day());

    println!("  Price: ${:.2}", vegetable.get_price());
    println!("  Quantity: {}", vegetable.get_quantity());
    println!("  Organic: {}", vegetable.get_organic());
    println!("  Supplier: {}", vegetable.get_supplier()?.to_string()?);

    println!("  Tags:");
    for tag in vegetable.get_tags()?.iter() {
        println!("    - {}", tag?.to_string()?);
    }

    let last_watered = vegetable.get_last_watered()?;
    println!("  Last Watered: {:02}:{:02}:{:02}", last_watered.get_hour(), last_watered.get_minute(), last_watered.get_second());

    println!("  Image Size: {} bytes", vegetable.get_image()?.len());

    Ok(())
}

pub fn main() -> Result<()> {
    let encoded_message = encode_vegetable()?;
    decode_vegetable(&encoded_message)?;
    Ok(())
}