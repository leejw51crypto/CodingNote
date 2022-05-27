use std::cell::RefCell;

struct MyStruct {
    value: RefCell<i32>,
}

impl MyStruct {
    fn new(value: i32) -> MyStruct {
        MyStruct {
            value: RefCell::new(value),
        }
    }

    fn update_value(&self, new_value: i32) {
        let mut value = self.value.borrow_mut();
        *value = new_value;
    }

    fn print_value(&self) {
        let value = self.value.borrow();
        println!("Value: {}", *value);
    }
}

fn main() {
    let my_struct = MyStruct::new(42);
    my_struct.print_value();
    my_struct.update_value(43);
    my_struct.print_value();
}
