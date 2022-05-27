use std::sync::{Arc, Mutex};
use std::thread;

// Define MyData struct
#[derive(Debug, Default)]
struct MyData {
    pub name: String,
}

// Define MyProgram struct
#[derive(Clone, Debug, Default)]
struct MyProgram {
    pub data: Arc<Mutex<MyData>>,
}

fn main() {
    // Create MyProgram instance
    let my_program = MyProgram {
        data: Arc::new(Mutex::new(MyData::default())),
    };

    // Clone my_program for each thread
    let my_program1 = my_program.clone();
    let my_program2 = my_program.clone();

    // Spawn the first thread
    let t1 = thread::spawn(move || {
        let my1 = my_program1.clone();

        // Acquire the lock and update the data
        let mut data = my1.data.lock().unwrap();
        data.name = "t1".to_string();
        println!("t1: {:?}", data);

        // Explicitly drop the lock to release it
        drop(data);

        // Acquire the lock again and update the data
        let mut data2 = my1.data.lock().unwrap();
        data2.name = "t1a".to_string();
        println!("t1a: {:?}", data2);
    });

    // Spawn the second thread
    let t2 = thread::spawn(move || {
        let my2 = my_program2.clone();

        // Acquire the lock and update the data
        let mut data = my2.data.lock().unwrap();
        data.name = "t2".to_string();
        println!("t2: {:?}", data);
    });

    // Wait for both threads to complete
    t1.join().unwrap();
    t2.join().unwrap();
}
