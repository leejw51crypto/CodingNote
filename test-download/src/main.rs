use std::fs::File;
use std::io::copy;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-osx-arm64-1.15.1.tgz";
    let response = reqwest::blocking::get(url)?;

    let path = Path::new("onnxruntime-osx-arm64-1.15.1.tgz");
    let mut file = File::create(&path)?;

    copy(&mut response.bytes()?.as_ref(), &mut file)?;
    let a=10;
    // print a
    


    println!("Downloaded to: {:?}", path);
    Ok(())
}

