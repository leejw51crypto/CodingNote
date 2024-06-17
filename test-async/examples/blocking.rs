use std::io::{self, Write};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn blocking_call(command_receiver: mpsc::Receiver<String>) {
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
}

fn main() {
    let (command_sender, command_receiver) = mpsc::channel();

    // Spawn a new thread for the blocking call
    let blocking_thread = thread::spawn(move || {
        blocking_call(command_receiver);
    });

    println!("Enter commands ('quit' to exit):");

    loop {
        // Read user input
        print!("> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let command = input.trim().to_lowercase();

        // Send the command to the blocking thread
        command_sender.send(command.clone()).unwrap();

        // Check if the user entered the "quit" command
        if command == "quit" {
            println!("Main thread received quit command. Waiting for threads to finish...");
            break;
        }
    }

    // Wait for the blocking thread to finish
    blocking_thread.join().unwrap();

    println!("All threads finished. Exiting...");
}