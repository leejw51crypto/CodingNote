use anyhow::{Context, Result};
use rayon::prelude::*;
use reqwest;
use tokio::runtime::Runtime;

fn main() -> Result<()> {
    // List of URLs to fetch
    let urls = vec![
        "https://api.github.com/repos/rust-lang/rust",
        "https://api.github.com/repos/rayon-rs/rayon",
        "https://api.github.com/repos/tokio-rs/tokio",
    ];

    // Create a new Tokio runtime
    let runtime = Runtime::new().context("Failed to create Tokio runtime")?;

    // Process data in parallel using Rayon and collect results
    let results: Vec<Result<(String, usize)>> = urls
        .par_iter()
        .map(|&url| {
            runtime.block_on(async { process_data(url).await })
        })
        .collect();

    // Print results
    for result in results {
        match result {
            Ok((url, size)) => println!("Processed data size for {}: {} bytes", url, size),
            Err(e) => eprintln!("Error processing data: {}", e),
        }
    }

    Ok(())
}

async fn process_data(url: &str) -> Result<(String, usize)> {
    // Fetch data
    println!("Fetching data from {}", url);
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .send()
        .await
        .context("Failed to send request")?;
    let body = resp.text().await.context("Failed to get response body")?;

    // Simulate some CPU-intensive work
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    println!("Processed data from {}", url);
    Ok((url.to_string(), body.len()))
}
