use anyhow::{Result, Context};
use rayon::prelude::*;
use reqwest;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // List of URLs to fetch
    let urls = vec![
        "https://api.github.com/repos/rust-lang/rust",
        "https://api.github.com/repos/rayon-rs/rayon",
        "https://api.github.com/repos/tokio-rs/tokio",
    ];

    // Process data in parallel using Rayon
    let processed_data: Result<Vec<usize>> = urls.par_iter()
        .map(|&url| {
            let runtime = tokio::runtime::Runtime::new()
                .context("Failed to create Tokio runtime")?;
            runtime.block_on(process_data(url))
        })
        .collect();

    // Print results
    match processed_data {
        Ok(sizes) => {
            for (index, size) in sizes.iter().enumerate() {
                println!("Processed data size for URL {}: {} bytes", index + 1, size);
            }
        },
        Err(e) => eprintln!("Error processing data: {:#}", e),
    }

    Ok(())
}

async fn process_data(url: &str) -> Result<usize> {
    // Fetch data
    println!("Fetching data from {}", url);
    let client = reqwest::Client::new();
    let resp = client.get(url).send().await
        .context("Failed to send request")?;
    let body = resp.text().await
        .context("Failed to get response body")?;

    // Simulate some CPU-intensive work
    std::thread::sleep(std::time::Duration::from_millis(1000));
    println!("Processed data from {}", url);
    Ok(body.len())
}
