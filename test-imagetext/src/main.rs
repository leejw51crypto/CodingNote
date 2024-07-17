// src/main.rs
use viuer::{print_from_file, Config};

fn main() {
    let conf = Config {
        // set offset
     //   x: 20,
      //  y: 4,
        // set dimensions
        width: Some(80),
        height: Some(25),
        absolute_offset: false,
        ..Default::default()
    };

    // starting from row 4 and column 20,
    // display `img.jpg` with dimensions 80x25 (in terminal cells)
    // note that the actual resolution in the terminal will be 80x50
    print_from_file("img.jpg", &conf).expect("Image printing failed.");
}