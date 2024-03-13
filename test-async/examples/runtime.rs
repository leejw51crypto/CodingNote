use tokio::runtime::Runtime;
use tokio::time::{sleep, Duration};

async fn blocking_task() {
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
    // Create a new runtime for blocking tasks
    let blocking_runtime = Runtime::new().unwrap();

    // Spawn the blocking task on the blocking runtime
    blocking_runtime.spawn(async move {
        blocking_task().await;
    });

    // Spawn the other task on the main runtime
    tokio::spawn(other_task());

    sleep(Duration::from_secs(2)).await;
    println!("Main task completed");
}
