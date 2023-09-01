use std::process::Command;

fn main() {
    // Run the flatc compiler
    let status = Command::new("flatc")
        .arg("--rust")
        .arg("hello.fbs")
        .current_dir("src")
        .status()
        .expect("Failed to execute flatc");

    // Make sure the command was successful
    if !status.success() {
        panic!("Failed to compile hello.fbs");
    }
    println!("cargo:rerun-if-changed=hello.fbs");
}
