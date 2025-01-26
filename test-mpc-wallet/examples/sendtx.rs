use ethers::{
    prelude::*,
    utils::parse_ether,
};
use eyre::Result;
use std::str::FromStr;
use std::sync::Arc;
use dotenv::dotenv;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables
    dotenv().ok();
    
    // Configure provider for Cronos testnet
    const RPC_ENDPOINT: &str = "https://evm-t3.cronos.org/";
    const CHAIN_ID: u64 = 338;

    // Connect to network
    let provider = Provider::<Http>::try_from(RPC_ENDPOINT)?;
    let provider = Arc::new(provider);

    // Get private key from environment
    let private_key = env::var("MY_FULL_PRIVATEKEY")
        .expect("MY_FULL_PRIVATEKEY must be set")
        .strip_prefix("0x")
        .unwrap_or(&env::var("MY_FULL_PRIVATEKEY").unwrap())
        .to_string();

    // Create wallet
    let wallet = LocalWallet::from_str(&private_key)?
        .with_chain_id(CHAIN_ID);
    
    // Create client from provider and wallet
    let client = SignerMiddleware::new(provider.clone(), wallet);
    
    // Get recipient address from environment
    let to_address = env::var("MY_TO_ADDRESS")
        .expect("MY_TO_ADDRESS must be set");
    let to_address = Address::from_str(&to_address)?;

    // Get initial balances
    let sender_address = client.address();
    let sender_balance = provider.get_balance(sender_address, None).await?;
    let receiver_balance = provider.get_balance(to_address, None).await?;

    println!("\n=== Initial Balances ===");
    println!("Sender balance: {} TCRO", format_units(sender_balance, "ether")?);
    println!("Receiver balance: {} TCRO", format_units(receiver_balance, "ether")?);

    // Prepare transaction
    let amount = parse_ether("0.1")?;
    let tx = TransactionRequest::new()
        .to(to_address)
        .value(amount)
        .from(sender_address);

    // Send transaction
    println!("\n=== Sending Transaction ===");
    let tx = client.send_transaction(tx, None).await?;
    println!("Transaction hash: {}", tx.tx_hash());

    // Wait for transaction to be mined
    let receipt = tx.await?;
    println!("\n=== Transaction Confirmed ===");
    println!("Block number: {}", receipt.block_number.unwrap());

    // Calculate gas costs
    let gas_used = receipt.gas_used.unwrap();
    let gas_price = receipt.effective_gas_price.unwrap();
    let gas_cost = gas_used * gas_price;
    
    println!("Gas used: {} (Cost: {} TCRO)", 
        gas_used,
        format_units(gas_cost, "ether")?
    );

    // Get final balances
    let sender_balance = provider.get_balance(sender_address, None).await?;
    let receiver_balance = provider.get_balance(to_address, None).await?;

    println!("\n=== Final Balances ===");
    println!("Sender balance: {} TCRO", format_units(sender_balance, "ether")?);
    println!("Receiver balance: {} TCRO", format_units(receiver_balance, "ether")?);
    println!("Gas cost: {} TCRO", format_units(gas_cost, "ether")?);

    Ok(())
} 