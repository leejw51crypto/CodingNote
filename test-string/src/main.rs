
pub fn print(a:&str) {
    println!("{}", a);
}
fn main() {
    let a:String="apple".into();
    print(&a);
}
