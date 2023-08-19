struct Dummy {
    field: String,
}

impl Dummy {
    fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
        }
    }
}

fn clone_and_display_pointer(original: &Dummy) {
    let cloned = original.clone();
    
    // Print the memory addresses of both original and cloned values
    println!("Original address: {:p}", original);
    println!("Cloned address: {:p}", &cloned);
}

fn main() {
    let dummy_instance = Dummy::new("Hello, world!");
    clone_and_display_pointer(&dummy_instance);
}
