use clap::Parser;
use fake::faker::name::raw::*;
use fake::locales::*;
use fake::Fake;
use reqwest;
use tokio;

/// Command line arguments
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Server address
    #[clap(short, long, default_value = "http://127.0.0.1:8080")]
    server: String,
}

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    let args = Args::parse();
    let client = reqwest::Client::new();

    // Generate a large text string

    let mut large_text = String::new();
    while large_text.split_whitespace().count() < 10000 {
        let part: Vec<String> = (Name(EN), 3..5).fake();
        large_text.push_str(&part.join(" "));
        large_text.push(' '); // Add space between parts
    }
    let large_text_bytes = large_text.into_bytes();

    let response = client
        .post(&format!("{}/store", args.server))
        .body(large_text_bytes)
        .send()
        .await?;

    if response.status().is_success() {
        let response_bytes = response.bytes().await?;
        let response_text = String::from_utf8_lossy(&response_bytes);
        println!("Response Text: {}", response_text);
    } else {
        eprintln!("Failed to call /store");
    }

    Ok(())
}
