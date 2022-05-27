use std::thread;

#[cxx::bridge(namespace = "org::applestore")]
mod ffi {
    // Shared structs with fields visible to both languages.
    struct AppleIphone {
        tags: Vec<String>,
    }

    // Rust types and signatures exposed to C++.
    extern "Rust" {
        type Iphone;

        fn push(self: &mut Iphone, data: &[u8]);
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("demo/include/apple.h");

        type ApplestoreClient;

        fn new_applestore_client() -> UniquePtr<ApplestoreClient>;
        fn onConnected(&self, info: String);
        fn onDisconnected(&self, info: String);

    }
}

#[derive(Default, Debug)]
pub struct Iphone {
    chunks: Vec<Vec<u8>>,
}
impl Iphone {
    pub fn push(self: &mut Iphone, data: &[u8]) {
        // add data to the buffer
        self.chunks.push(data.to_vec());
    }
}

fn main() {
    let thread_join_handle = thread::spawn(move || {
        let client = ffi::new_applestore_client();

        for i in 0..3 {
            
            client.onConnected("hello".to_string());
            // sleep 1 second
            println!("loop {}", i);
            thread::sleep(std::time::Duration::from_secs(1));
            client.onDisconnected("world".to_string());
        }


        
        
    });
    // some work here
    let res = thread_join_handle.join();

    
}
