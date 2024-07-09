use ecies::{decrypt, encrypt};
use ethers::{
    prelude::*,
    signers::{coins_bip39::English, MnemonicBuilder},
};
use k256::{
    ecdsa::SigningKey,
    elliptic_curve::generic_array::{typenum::U32, GenericArray},
};
use std::time::Instant;

pub async fn process() -> anyhow::Result<()> {
    const MSG: &str = "helloworld🌍 123";
    println!("🔐 Message: {}", MSG);
    // print length of message
    println!("📏 Message Length: {}", MSG.len());

    // Retrieve mnemonic from environment variable
    let mymnemonics = std::env::var("MY_MNEMONICS").expect("MY_MNEMONICS must be set");

    // Create a wallet from the mnemonic
    let wallet = MnemonicBuilder::<English>::default()
        .phrase(mymnemonics.as_str())
        .index(0u32)?
        .build()
        .expect("Failed to build wallet");

    // Generate Ethereum address from wallet
    let eth_address = wallet.address();
    println!("🔒 Ethereum Address: {:?}", eth_address);

    // Convert Ethereum private key to a compatible format for ECIES
    let secret_key_bytes: Vec<u8> = wallet.signer().to_bytes().to_vec();
    let secret_key_bytes_array: GenericArray<u8, U32> =
        GenericArray::clone_from_slice(&secret_key_bytes);
    let secret_key = SigningKey::from_bytes(&secret_key_bytes_array).expect("Failed to create key");

    // Derive public key for encryption
    let public_key = secret_key.verifying_key();
    let public_key_bytes = public_key.to_encoded_point(false).to_bytes();

    // Show length of secret_key_bytes and public_key_bytes
    println!("🔑 Secret Key Length: {}", secret_key_bytes.len());
    println!("🗝️ Public Key Length: {}", public_key_bytes.len());

    let sk = &secret_key_bytes;
    let pk = &public_key_bytes;

    let msg: &[u8] = MSG.as_bytes();

    // Measure encryption time
    let start_encrypt = Instant::now();
    let encrypted_bytes = encrypt(&pk, msg)?;
    let encrypt_duration = start_encrypt.elapsed();

    // Measure decryption time
    let start_decrypt = Instant::now();
    let decrypted_bytes = decrypt(&sk, &encrypted_bytes)?;
    let decrypt_duration = start_decrypt.elapsed();

    let decrypted_str = std::str::from_utf8(&decrypted_bytes)?;
    println!("🔓 Decrypted: {}", decrypted_str);
    assert_eq!(MSG, decrypted_str);

    // Print encryption and decryption times
    println!("⏱️ Encryption Time: {:?}", encrypt_duration);
    println!("⏱️ Decryption Time: {:?}", decrypt_duration);

    Ok(())
}
