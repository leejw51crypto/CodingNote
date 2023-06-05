use hex;
use ethers::{prelude::*, abi::AbiEncode};
fn test() {
    let a = 1000;
    // print a with formatting
    let b=format!("{:x}", a);
    println!("{}",b);
    let c= hex::decode(b).unwrap();
}
fn test2() {
    let a= U256::from_dec_str("1000").unwrap();
    // make byte 32 array 
    let mut a2: [u8; 32] = [0; 32];
    a.to_big_endian(&mut a2);
    let mut b=format!("{:02x}", a);
    // b length is odd, attach 0 in front
    if b.len()%2!=0 {
        let mut c=String::from("0");
        c.push_str(&b);
        b=c;
    }
    println!("b={}",b);
    let bytes=  hex::decode(b).unwrap();
    let c= U256::from_big_endian(&bytes);
    println!("c={}",c);
}
fn pad_zero(s: String) -> String {
    if s.len() % 2 != 0 {
        format!("0{}", s)
    } else {
        s
    }
}

fn main() {
    let a= U256::from_dec_str("1000").unwrap();
    let mut b= format!("{:x}",a);
    b=pad_zero(b);
    let bytes=  hex::decode(&b).unwrap();
    let c= U256::from_big_endian(&bytes);
    println!("a={}",a);
    println!("b={}",b);
    println!("c={}",c);
    assert_eq!(a,c);
}
