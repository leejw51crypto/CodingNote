// Define the Printable trait with a single method, `print`.
trait Printable {
    fn print(&self);
}

// Define a Person struct with a single field, `name`.
struct Person {
    name: String,
}

// Implement the Printable trait for the Person struct.
impl Printable for Person {
    fn print(&self) {
        // Print the name of the person.
        println!("My name is {}", self.name);
    }
}

// Define a generic function, `print_anything`, that takes a reference to an
// object that implements the Printable trait and calls the `print` method on it.
fn print_anything<T: Printable>(thing: &T) {
    thing.print();
}

fn main() {
    // Create a new Person instance with the name "John Doe".
    let person = Person { name: String::from("John Doe") };
    
    // Call the `print_anything` function with a reference to the person instance.
    print_anything(&person);
}
