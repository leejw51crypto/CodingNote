use wasm_bindgen::prelude::*;

// The `wasm_bindgen` macro marks this function for automatic JavaScript binding.
#[wasm_bindgen]
pub fn greet(name: &str) {
    let greeting = format!("Hello, {}!", name);
    // Using `web_sys` to interact with the console
    web_sys::console::log_1(&greeting.into());
}

// You can also expose Rust structures to JavaScript.
#[wasm_bindgen]
pub struct Person {
    name: String,
    age: u32,
}

#[wasm_bindgen]
impl Person {
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str, age: u32) -> Person {
        Person {
            name: name.to_owned(),
            age,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[wasm_bindgen(method)]
    pub fn celebrate_birthday(&mut self) {
        self.age += 1;
    }
}
