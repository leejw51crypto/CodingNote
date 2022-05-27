use std::convert::{TryFrom, TryInto};

#[derive(Debug)]
struct Number(i32);

impl TryFrom<i32> for Number {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if value >= 0 {
            Ok(Number(value))
        } else {
            Err("Invalid number".to_string())
        }
    }
}

/*
impl From<i32> for Number {
    fn from(value: i32) -> Self {
        Number(value)
    }
}
*/

impl From<Number> for i32 {
    fn from(number: Number) -> Self {
        number.0
    }
}

fn main() {
    let valid_number: Result<Number, String> = Number::try_from(42);
    let invalid_number: Result<Number, String> = Number::try_from(-1);

    println!("{:?}", valid_number);   // Output: Ok(Number(42))
    println!("{:?}", invalid_number); // Output: Err("Invalid number")

    let number: i32 = valid_number.unwrap().into();
    println!("{}", number); // Output: 42
}
