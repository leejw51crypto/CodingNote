macro_rules! square_fn {
    ($fn_name:ident) => {
        fn $fn_name(x: i32) -> i32 {
            x * x
        }
    };
}

// Use the macro to define a function.
square_fn!(square);

fn main() {
    assert_eq!(square(2), 4);
    println!("Passed assertion.");
}
