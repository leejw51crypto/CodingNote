use anyhow::Result;
use bip39::{Language, Mnemonic};
use chrono;
use colored::*;
use ethers::prelude::k256::ecdsa::SigningKey;
use ethers::signers::{LocalWallet, MnemonicBuilder, Signer};
use hex;
use prettytable::{row, Table};
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

    let secret_key_bytes: [u8; 32] = wallet.signer().to_bytes().into();
    let secret_key =
        SigningKey::from_bytes(&secret_key_bytes.into()).expect("Failed to create key");

    let public_key = secret_key.verifying_key();
    let public_key_bytes = public_key.to_encoded_point(false).to_bytes();

    let truncated_pubkey = format!(
        "{}...{}",
        hex::encode(&public_key_bytes[..8]),
        hex::encode(&public_key_bytes[public_key_bytes.len() - 8..])
    );

    let mut table = Table::new();
    table.set_format(*prettytable::format::consts::FORMAT_BOX_CHARS);
    table.add_row(row!["Index", index]);
    table.add_row(row!["Wallet Address", wallet_string]);
    table.add_row(row!["Public Key", truncated_pubkey]);
    table.add_row(row![
        "Public Key Length",
        format!("{} bytes", public_key_bytes.len())
    ]);
    table.add_row(row![
        "Private Key Length",
        format!("{} bytes", secret_key_bytes.len())
    ]);

    table.printstd();
    Ok(())
}

fn print_current_time() {
    let now = chrono::offset::Local::now();
    let utc_now = chrono::offset::Utc::now();

    let mut table = Table::new();
    table.set_format(*prettytable::format::consts::FORMAT_BOX_CHARS);
    table.add_row(row!["Local Time", now.to_string()]);
    table.add_row(row!["UTC Time", utc_now.to_string()]);

    table.printstd();
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
