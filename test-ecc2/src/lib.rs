use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use aes::Aes256;
use hkdf::Hkdf;
use k256::SecretKey;
use rand::rngs::OsRng;
use sha2::Sha256;

pub fn make_key() -> Aes256 {
    // Generate an elliptic curve private key
    let ec_private_key = SecretKey::random(&mut OsRng);

    // Derive a symmetric key from the elliptic curve private key using HKDF
    let mut symmetric_key = [0u8; 32];
    let hk = Hkdf::<Sha256>::new(None, ec_private_key.to_be_bytes().as_slice());
    hk.expand(b"symmetric-key", &mut symmetric_key)
        .expect("Failed to derive symmetric key");

    // Create an instance of the AES-256 cipher with the derived symmetric key
    let cipher = Aes256::new(&symmetric_key.into());
    return cipher;
}
pub fn compute() {
    let cipher = make_key();
    let cipher2 = make_key();
    // Plaintext to be encrypted (must be a multiple of 16 bytes)
    let plaintext = b"Hello, World!!!!";
    // print plaintext
    println!("Plaintext: {:?}", plaintext);
    println!(
        "Plaintext: {}",
        String::from_utf8(plaintext.to_vec()).unwrap()
    );

    // Perform encryption
    let mut ciphertext = plaintext.to_vec();
    cipher.encrypt_block(ciphertext.as_mut_slice().into());

    println!("Ciphertext: {:?}", ciphertext);

    // Perform decryption
    let mut decrypted_plaintext = ciphertext.clone();
    cipher.decrypt_block(decrypted_plaintext.as_mut_slice().into());

    println!("Decrypted Plaintext: {:?}", decrypted_plaintext);
    // convert decrypted_plaintext to string , print
    let decrypted_plaintext = String::from_utf8(decrypted_plaintext).unwrap();
    println!("Decrypted Plaintext: {:?}", decrypted_plaintext);
}
