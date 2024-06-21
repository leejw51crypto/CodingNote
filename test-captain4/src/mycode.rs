#[derive(Debug, Clone, Default)]
pub struct MyCode {
    pub name: String,
}

impl MyCode {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}
