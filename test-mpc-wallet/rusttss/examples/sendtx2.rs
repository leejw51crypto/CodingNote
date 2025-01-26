use dotenv::dotenv;
use ethers::{
    prelude::*,
    utils::{format_units, parse_ether},
};
use eyre::Result;
use std::env;
use std::str::FromStr;
use std::sync::Arc;

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
    let wallet = LocalWallet::from_str(&private_key)?.with_chain_id(CHAIN_ID);

    // Create client from provider and wallet
    let client = SignerMiddleware::new(provider.clone(), wallet);

    // Get recipient address from environment
    let to_address = env::var("MY_TO_ADDRESS").expect("MY_TO_ADDRESS must be set");
    let to_address = Address::from_str(&to_address)?;

    // Get initial balances
    let sender_address = client.address();
    let sender_balance = provider.get_balance(sender_address, None).await?;
    let receiver_balance = provider.get_balance(to_address, None).await?;

    println!("\n=== Initial Balances ===");
    println!(
        "Sender balance: {} TCRO",
        format_units(sender_balance, "ether")?
    );
    println!(
        "Receiver balance: {} TCRO",
        format_units(receiver_balance, "ether")?
    );

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

    // Add delay to ensure transaction is available
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Get transaction details including signature
    if let Some(tx_data) = provider.get_transaction(tx.tx_hash()).await? {
        println!("\n=== Signature Components ===");
        println!("r: {} (length: 32 bytes)", tx_data.r);
        println!("s: {} (length: 32 bytes)", tx_data.s);
        let raw_v = tx_data.v.as_u64();
        println!("v (raw): {} (chain_id * 2 + 35 + recovery_id)", raw_v);

        // Calculate normalized v (27 or 28)
        let normalized_v = if (raw_v - 35) % 2 == 0 { 27 } else { 28 };
        println!(
            "v (normalized): {} (27 or 28 for legacy transactions)",
            normalized_v
        );

        // Show calculation
        println!("\nv calculation:");
        println!("chain_id = {}", CHAIN_ID);
        println!("chain_id * 2 = {}", CHAIN_ID * 2);
        println!("chain_id * 2 + 35 = {}", CHAIN_ID * 2 + 35);
        println!("recovery_id = {}", if normalized_v == 27 { 0 } else { 1 });
    } else {
        println!("Could not retrieve transaction signature components immediately");
    }

    // Wait for transaction to be mined
    if let Some(receipt) = tx.await? {
        println!("\n=== Transaction Confirmed ===");
        if let Some(block_number) = receipt.block_number {
            println!("Block number: {}", block_number);
        }

        // Calculate gas costs
        if let (Some(gas_used), Some(gas_price)) = (receipt.gas_used, receipt.effective_gas_price) {
            let gas_cost = gas_used * gas_price;
            println!(
                "Gas used: {} (Cost: {} TCRO)",
                gas_used,
                format_units(gas_cost, "ether")?
            );

            // Get final balances
            let sender_balance = provider.get_balance(sender_address, None).await?;
            let receiver_balance = provider.get_balance(to_address, None).await?;

            println!("\n=== Final Balances ===");
            println!(
                "Sender balance: {} TCRO",
                format_units(sender_balance, "ether")?
            );
            println!(
                "Receiver balance: {} TCRO",
                format_units(receiver_balance, "ether")?
            );
            println!("Gas cost: {} TCRO", format_units(gas_cost, "ether")?);
        }
    }

    Ok(())
}
