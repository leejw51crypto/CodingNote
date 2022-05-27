const BRIDGES: &[&str] = &["src/lib.rs"];

fn main() {
    cxx_build::bridges(BRIDGES)
        .flag_if_supported("-std=c++14")
        .compile("mysdk");

    for bridge in BRIDGES {
        println!("cargo:rerun-if-changed={bridge}");
    }
}
