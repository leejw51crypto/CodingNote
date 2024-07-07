use text_io::read;

fn main() {
    println!("Enter your name:");
    let name: String = read!("{}\n");
    println!("Hello, start<{}>end!", name);
    println!("Hello2, start<{}>end!", name.trim());
}
