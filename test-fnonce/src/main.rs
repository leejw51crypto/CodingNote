fn eat<F>(func: F)
where // f can be called only once
    F: FnOnce() -> String,
{
    println!("ate: {}", func());
}

fn eat2<F>(func:F) where F: Fn()->String,
{
    println!("ate: {}", func());
}

fn eat3<F>(mut func: F) where F: FnMut()->String,
{
    println!("ate: {}", func());
}


fn main() {
    let x:String = "apple".into();
    let a = move || x;
    eat(a);
    

    let y:String= "pear".into();
    let b=  || y.clone();
    eat2(b);
    eat2(b);
    
    let mut z: String="strawberry".into();
    let c= move ||  { z.push_str(" icecream"); z.clone() };
    eat3(c);
}
