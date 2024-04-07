use std::time::SystemTime;
use std::time::UNIX_EPOCH;
fn main() -> anyhow::Result<()> {
    // time with nanoseconds
    let timestamp1 = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as i64;
    let timestamp2 = chrono::Utc::now().timestamp_nanos_opt().unwrap();
    println!("timestamp1: {}", timestamp1);
    println!("timestamp2: {}", timestamp2);

    let timestamp1 = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros() as i64;
    let timestamp2 = chrono::Utc::now().timestamp_micros();
    println!("timestamp1: {}", timestamp1);
    println!("timestamp2: {}", timestamp2);
    // the same but with milliseconds
    let timestamp1 = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as i64;
    let timestamp2 = chrono::Utc::now().timestamp_millis();
    println!("timestamp1: {}", timestamp1);
    println!("timestamp2: {}", timestamp2);

    // the same but with seconds
    let timestamp1 = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    let timestamp2 = chrono::Utc::now().timestamp();
    println!("timestamp1: {}", timestamp1);
    println!("timestamp2: {}", timestamp2);


    let currenttime= chrono::Utc::now();
    let currentlocaltime= chrono::Local::now();
    println!("currenttime: {}", currenttime);
    println!("currentlocaltime: {}", currentlocaltime);
    Ok(())
}
