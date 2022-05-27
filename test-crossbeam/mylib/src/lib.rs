use chrono::Local;
use libc::c_void;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::ffi::CString;
use std::os::raw::c_char;
use tokio::runtime::Builder;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::timeout;
use tokio::time::Duration;
pub struct ChannelWrapper {
    receiver: tokio::sync::Mutex<UnboundedReceiver<String>>,
    sender: UnboundedSender<String>,
    runtime: Runtime,
}
#[no_mangle]
pub extern "C" fn create_channel() -> *mut c_void {
    let (tx, rx): (UnboundedSender<String>, UnboundedReceiver<String>) = unbounded_channel();

    let runtime = Builder::new_multi_thread()
        .worker_threads(4) // Set the number of worker threads
        .enable_all() // Enable all Tokio features
        .build()
        .unwrap(); // Build the runtime

    let wrapper = Box::new(ChannelWrapper {
        receiver: rx.into(),
        sender: tx,
        runtime,
    });

    let mytx = wrapper.sender.clone();

    // Spawn a sender task
    let sender_task = async move {
        loop {
            let mut rng = StdRng::from_entropy();
            let random_data: [u8; 8] = rng.gen();
            let random_hex = hex::encode(random_data);
            let now = Local::now();

            let message = format!("Message: [Date-Time: {now}] [Random Data: {random_hex}]",);
            //let now = Local::now();
            //let message = "OK".to_string();

            mytx.send(message).unwrap();
            println!("Sender sent message {now}");
            tokio::time::sleep(Duration::from_millis(1000)).await;
        }
    };

    // Use the spawn method on the runtime
    wrapper.runtime.spawn(sender_task);

    Box::into_raw(wrapper) as *mut c_void
}

#[no_mangle]
pub extern "C" fn receive_message(channel_ptr: *mut c_void, timeout_ms: u32) -> *const c_char {
    let channel = unsafe { &*(channel_ptr as *mut ChannelWrapper) };
    let timeout_duration = Duration::from_millis(timeout_ms.into());
    // println!("receiving {:?}", timeout_ms);

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut receiver = channel.receiver.lock().await;
        match timeout(timeout_duration, receiver.recv()).await {
            Ok(Some(msg)) => {
                let c_str_msg = CString::new(msg).unwrap();
                c_str_msg.into_raw()
            }
            _ => std::ptr::null(),
        }
    })
}

#[no_mangle]
pub extern "C" fn destroy_channel(channel_ptr: *mut c_void) {
    if !channel_ptr.is_null() {
        unsafe {
            let _output = Box::from_raw(channel_ptr as *mut ChannelWrapper);
        }
    }
}

#[cxx::bridge(namespace = "mysdk")]
mod ffi {
    #[derive(Debug, Default)]
    pub struct MyMessage {
        pub mymessage: String,
    }

    extern "Rust" {
        fn helloworld() -> i32;
    }
}

pub fn helloworld() -> i32 {
    println!("hello world ================");
    0
}
