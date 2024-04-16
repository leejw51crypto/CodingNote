use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use aes::Aes256;
use anyhow::{anyhow, Result};
use hkdf::Hkdf;
use k256::SecretKey;
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::Sha256;

pub fn make_key() -> Result<Aes256> {
    // Generate an elliptic curve private key
    let ec_private_key = SecretKey::random(&mut OsRng);

    // Derive a symmetric key from the elliptic curve private key using HKDF
    let mut symmetric_key = [0u8; 32];
    let hk = Hkdf::<Sha256>::new(None, ec_private_key.to_be_bytes().as_slice());
    hk.expand(b"symmetric-key", &mut symmetric_key)
        .map_err(|e| anyhow!("Failed to derive symmetric key: {}", e))?;

    // Create an instance of the AES-256 cipher with the derived symmetric key
    let cipher = Aes256::new(&symmetric_key.into());
    Ok(cipher)
}

pub fn make_key2() -> Result<Aes256> {
    // Generate a random 256-bit key
    let mut rng = OsRng;
    let mut key = [0u8; 32];
    rng.fill_bytes(&mut key);

    // Create an instance of the AES-256 cipher with the random key
    let cipher = Aes256::new(&key.into());
    Ok(cipher)
}

pub fn compute() -> Result<()> {
    let cipher = make_key2()?;

    // Plaintext to be encrypted (must be a multiple of 16 bytes)
    let plaintext = b"Hello, World!!!!";

    // Print plaintext
    println!("Plaintext: {:?}", plaintext);
    println!(
        "Plaintext: {}",
        String::from_utf8(plaintext.to_vec()).map_err(|e| anyhow!("Invalid UTF-8: {}", e))?
    );

    // Perform encryption
    let mut ciphertext = plaintext.to_vec();
    cipher.encrypt_block(ciphertext.as_mut_slice().into());

    println!("Ciphertext: {:?}", ciphertext);

    // Perform decryption
    let mut decrypted_plaintext = ciphertext.clone();
    cipher.decrypt_block(decrypted_plaintext.as_mut_slice().into());

    println!("Decrypted Plaintext: {:?}", decrypted_plaintext);

    // Convert decrypted_plaintext to string and print
    let decrypted_plaintext =
        String::from_utf8(decrypted_plaintext).map_err(|e| anyhow!("Invalid UTF-8: {}", e))?;
    println!("Decrypted Plaintext: {:?}", decrypted_plaintext);

    Ok(())
}
