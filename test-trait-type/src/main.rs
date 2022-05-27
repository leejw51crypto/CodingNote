trait Foo {
    type Bar;
    fn foo(&self) -> Self::Bar;
}

struct FooStruct {
    bar: i32,
}
impl Foo for FooStruct {
    type Bar = i32;
    fn foo(&self) -> Self::Bar {
        self.bar
    }
}
fn main() {
    println!("Hello, world!");
    let foo = FooStruct { bar: 1 };
    println!("{}", foo.foo());
}
