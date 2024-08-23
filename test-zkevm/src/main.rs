use anyhow::Result;
use chrono;
use ethers::prelude::k256::ecdsa::SigningKey;
use ethers::signers::{LocalWallet, MnemonicBuilder, Signer};
use ethers::providers::{Provider, Http};
use ethers::types::{Address, U256};
use hex;
use prettytable::{row, Table};
use rpassword::prompt_password;
use std::env;
use ethers::middleware::SignerMiddleware;
use ethers::providers::Middleware;
use std::sync::Arc;
use zksync_web3_rs::{zks_wallet::TransferRequest, ZKSWallet};

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

async fn send_amount(from_wallet: &LocalWallet, to_address: Address, amount: U256) -> Result<()> {
    let provider = Provider::<Http>::try_from("https://testnet.zkevm.cronos.org")?;
    let chain_id = 282u64; // Cronos zkEVM testnet

    // Clone the wallet to avoid moving out of a shared reference
    let signer = from_wallet.clone().with_chain_id(chain_id);
    let client = SignerMiddleware::new(provider.clone(), signer);
    let client = Arc::new(client);

    let zk_wallet = ZKSWallet::new(client.signer().clone(), None, Some(provider.clone()), None)?;

    println!("Sender's balance before paying: {:?}", 
        provider.get_balance(from_wallet.address(), None).await?);
    println!("Receiver's balance before getting paid: {:?}", 
        provider.get_balance(to_address, None).await?);

    // Create a ZKSWallet transfer request
    let payment_request = TransferRequest::new(amount)
        .to(to_address)
        .from(from_wallet.address());

    // Send the transaction using ZKSWallet
    let tx_hash = zk_wallet.transfer(&payment_request, None).await?;
    let tx = provider.get_transaction_receipt(tx_hash).await?.unwrap();

    println!("Transaction receipt: {:?}", tx);

    println!("Sender's balance after paying: {:?}", 
        provider.get_balance(from_wallet.address(), None).await?);
    println!("Receiver's balance after getting paid: {:?}", 
        provider.get_balance(to_address, None).await?);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mnemonics = get_mnemonics()?;

    let mut wallets = Vec::new();
    for i in 0..5 {
        let wallet = create_wallet(&mnemonics, i)?;
        print_wallet_info(&wallet, i)?;
        wallets.push(wallet);
        println!();
    }

    print_current_time();

    // Send amount from wallet 0 to wallet 1
    let amount = U256::from(100_000_000_000_000_000u64); // 0.1 ETH
    let from_wallet = &wallets[0];
    let to_address = wallets[1].address();

    println!("Sending {} wei from wallet 0 to wallet 1", amount);
    send_amount(from_wallet, to_address, amount).await?;

    Ok(())
}