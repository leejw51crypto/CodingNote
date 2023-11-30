use wasm_bindgen::prelude::wasm_bindgen;
#[wasm_bindgen]
extern {
    fn alert(s: &str);    
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, world!");
}

#[wasm_bindgen]
pub fn myadd(a: i32, b: i32) -> i32 {
    a+b
}
