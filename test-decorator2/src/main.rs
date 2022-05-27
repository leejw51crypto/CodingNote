trait Base {
    fn execute(&self) -> i32;
}

struct ConcreteBase {
    value: i32,
}

impl Base for ConcreteBase {
    fn execute(&self) -> i32 {
        self.value
    }
}

struct Decorator {
    base: Box<dyn Base>,
}

impl Decorator {
    fn new(base: Box<dyn Base>) -> Self {
        Self { base }
    }
}

impl Base for Decorator {
    fn execute(&self) -> i32 {
        self.base.execute() + 1
    }
}

fn main() {
    let base = Box::new(ConcreteBase { value: 1 });
    let mut decorator = Decorator::new(base);
    println!("{}", decorator.execute());
    decorator = Decorator::new(Box::new(decorator));
    println!("{}", decorator.execute());
    decorator = Decorator::new(Box::new(decorator));
    println!("{}", decorator.execute());
}
