use anyhow::Result;
use bip39::{Language, Mnemonic};
use chrono;
use colored::*;
use ethers::prelude::k256::ecdsa::SigningKey;
use ethers::signers::{LocalWallet, MnemonicBuilder, Signer};
use hex;
use rpassword::prompt_password;
use std::env;

fn get_mnemonics() -> Result<String> {
    let mut mnemonics = prompt_password("Enter your mnemonics: ")?;
    if mnemonics.is_empty() {
        mnemonics = env::var("MY_MNEMONICS")?;
    }
    assert!(!mnemonics.is_empty(), "Mnemonics cannot be empty");
    Ok(mnemonics)
}

fn create_wallet(mnemonics: &str, index: u32) -> Result<LocalWallet> {
    MnemonicBuilder::<ethers::signers::coins_bip39::English>::default()
        .phrase(mnemonics)
        .index(index)?
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create wallet: {}", e))
}

fn print_wallet_info(wallet: &LocalWallet, index: u32) -> Result<()> {
    let wallet_address: Vec<u8> = wallet.address().0.to_vec();
    let wallet_string = hex::encode(&wallet_address);
    println!("Wallet {} Address: {}", index, wallet_string.green());

    let secret_key_bytes: [u8; 32] = wallet.signer().to_bytes().into();
    let secret_key =
        SigningKey::from_bytes(&secret_key_bytes.into()).expect("Failed to create key");

    let public_key = secret_key.verifying_key();
    let public_key_bytes = public_key.to_encoded_point(false).to_bytes();

    println!(
        "Public Key {}: {}",
        index,
        hex::encode(&public_key_bytes).blue()
    );
    println!(
        "Public Key {} Length: {} bytes",
        index,
        public_key_bytes.len()
    );

    println!(
        "Private Key {} Length: {} bytes",
        index,
        secret_key_bytes.len()
    );

    Ok(())
}

fn print_current_time() {
    let now = chrono::offset::Local::now();
    let utc_now = chrono::offset::Utc::now();

    println!("Local Time: {}", now.to_string().yellow());
    println!("UTC Time: {}", utc_now.to_string().yellow());
}

fn main() -> Result<()> {
    let mnemonics = get_mnemonics()?;

    for i in 0..5 {
        let wallet = create_wallet(&mnemonics, i)?;
        print_wallet_info(&wallet, i)?;
        println!();
    }

    print_current_time();

    Ok(())
}
