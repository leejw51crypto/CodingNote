use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let result = fetch_data().await;
    println!("Data fetched: {result}");
}

async fn fetch_data() -> String {
    println!("Simulating data fetch...");

    // Simulate network delay
    sleep(Duration::from_secs(2)).await;

    // Return the result after the delay
    String::from("Hello, async world!")
}
