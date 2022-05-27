fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("src/securestorage.cpp")
        .flag_if_supported("-std=c++14")
        .compile("cxx-demo");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/securestorage.cpp");
}
