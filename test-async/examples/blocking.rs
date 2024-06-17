use anyhow::Result;
use std::io::{self, Write};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tokio::task;

fn blocking_call(command_receiver: mpsc::Receiver<String>) -> Result<()> {
    println!("Blocking call started");
    loop {
        // Perform some work without yielding
        thread::sleep(Duration::from_secs(1));
        println!("Blocking call running...");

        // Check if a command is received
        if let Ok(command) = command_receiver.try_recv() {
            if command == "quit" {
                println!("Blocking call received quit command. Exiting...");
                break;
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let (command_sender, command_receiver) = mpsc::channel();

    // Spawn a new thread for the blocking call using `task::spawn_blocking`
    let blocking_thread = task::spawn_blocking(move || {
        blocking_call(command_receiver)
    });

    println!("Enter commands ('quit' to exit):");

    loop {
        // Read user input
        print!("> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let command = input.trim().to_lowercase();

        // Send the command to the blocking thread
        command_sender.send(command.clone())?;

        // Check if the user entered the "quit" command
        if command == "quit" {
            println!("Main task received quit command. Waiting for tasks to finish...");
            break;
        }
    }

    // Wait for the blocking thread to finish
    blocking_thread.await??;

    println!("All tasks finished. Exiting...");
    Ok(())
}