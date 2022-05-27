// use anyhow
use anyhow::Result;
#[cxx::bridge(namespace = "org::blobstore")]
mod ffi {
    extern "Rust" {
        fn mytest() -> Result<()>;
    }

    unsafe extern "C++" {
        include!("test-bindings/src/securestorage.h");
        fn test();
        fn secureStorageWrite(userkey2: String, uservalue2: String) -> i32;
        fn secureStorageRead(userkey2: String) -> String;
    }
}

fn mytest() -> Result<()> {
    println!("Hello, world!");
    Ok(())
}
