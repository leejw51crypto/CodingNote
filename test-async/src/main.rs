use tokio::time::{sleep, Duration};

async fn blocking_task() {
    println!("blocking task started");
    // Simulating a blocking operation without yielding
    loop {
        // Perform some work without yielding
    }
}

async fn other_task() {
    println!("Other task started");
    sleep(Duration::from_secs(1)).await;
    println!("Other task completed");
}

#[tokio::main]
async fn main() {
    tokio::spawn(blocking_task());
    tokio::spawn(other_task());

    sleep(Duration::from_secs(2)).await;
    println!("Main task completed");
}
