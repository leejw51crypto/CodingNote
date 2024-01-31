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
    while large_text.split_whitespace().count() < 1000 {
        let part: Vec<String> = (Name(EN), 3..5).fake();
        large_text.push_str(&part.join(" "));
        large_text.push(' '); // Add space between parts
    }
    //println!("large text {}", large_text);

    let response = client
        .get(&format!("{}/store", args.server))
        .query(&[("text", &large_text)])
        .send()
        .await?;

    if response.status().is_success() {
        let response_text = response.text().await?;
        println!("Response Text: {}", response_text);
    } else {
        eprintln!("Failed to call /store");
    }

    Ok(())
}
