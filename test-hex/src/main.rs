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
fn main() {
    let a= U256::from_dec_str("100").unwrap();
    let mut b= format!("{:x}",a);
    if b.len()%2!=0 {
        b="0".to_string()+&b;
    } 
    println!("b={}",b);
}
