use ecies::{decrypt, encrypt, utils::generate_keypair};
use std::io::Read;

pub async fn process() {
    const MSG: &str = "helloworld🌍";
    let (sk, pk) = generate_keypair();
    let sk = sk.serialize();
    let pk = pk.serialize();
    // print length of sk, pk
    println!("Length of sk: {}", sk.len());
    println!("Length of pk: {}", pk.len());
    let msg: &[u8] = MSG.as_bytes();
    let encrypted_bytes = encrypt(&pk, msg).unwrap();
    let decrypted_bytes = decrypt(&sk, &encrypted_bytes).unwrap();
    let decrypted_str = std::str::from_utf8(&decrypted_bytes).unwrap();
    println!("Decrypted: {}", decrypted_str);
    assert_eq!(MSG, decrypted_str);
}