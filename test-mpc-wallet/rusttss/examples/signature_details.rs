use dotenv::dotenv;
use ethers::{prelude::*, types::transaction::eip2718::TypedTransaction, utils::parse_ether};
use eyre::Result;
use serde_json::json;
use std::env;
use std::str::FromStr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
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

    // Create wallet with chain ID (important for EIP-155)
    let wallet = LocalWallet::from_str(&private_key)?.with_chain_id(CHAIN_ID);

    // Get recipient address
    let to_address = env::var("MY_TO_ADDRESS").expect("MY_TO_ADDRESS must be set");
    let to_address = Address::from_str(&to_address)?;

    // Get current gas prices and nonce
    let gas_price = provider.get_gas_price().await?;
    let nonce = provider
        .get_transaction_count(wallet.address(), None)
        .await?;

    // 1. Prepare the transaction (using legacy format)
    let amount = parse_ether("0.1")?;
    let gas_limit = U256::from(21000); // Standard ETH transfer gas limit

    // Create transaction request
    let tx = TransactionRequest::new()
        .to(to_address)
        .value(amount)
        .from(wallet.address())
        .chain_id(CHAIN_ID)
        .gas_price(gas_price)
        .nonce(nonce)
        .gas(gas_limit);

    println!("\n=== Transaction Details (Before Signing) ===");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "to": format!("{:?}", to_address),
            "from": format!("{:?}", wallet.address()),
            "value": format!("{:?}", amount),
            "chain_id": CHAIN_ID,
            "gas_price": format!("0x{:x}", gas_price),
            "gas_limit": format!("0x{:x}", gas_limit),
            "nonce": format!("0x{:x}", nonce),
        }))?
    );

    // 2. Convert to typed transaction and sign
    let typed_tx = TypedTransaction::Legacy(tx);
    let signature = wallet.sign_transaction_sync(&typed_tx)?;

    println!("\n=== Signature Details ===");
    // r and s are 32 bytes each, v is 1 byte
    println!("r: 0x{:x} (length: 32 bytes)", signature.r);
    println!("s: 0x{:x} (length: 32 bytes)", signature.s);
    println!("v: {} (decimal) = 0x{:x} (hex)", signature.v, signature.v);
    println!("\n=== Hex Representation ===");
    let mut r_bytes = [0u8; 32];
    let mut s_bytes = [0u8; 32];
    signature.r.to_big_endian(&mut r_bytes);
    signature.s.to_big_endian(&mut s_bytes);
    println!("r: {}", hex::encode(r_bytes));
    println!("s: {}", hex::encode(s_bytes));
    // v value in hex (0x2c8 = 712 in decimal)
    // In EIP-155: v = chain_id * 2 + 35 + recovery_id
    // For Cronos testnet (chain_id = 338):
    //   v = 338 * 2 + 35 + recovery_id
    //   v = 676 + 35 + recovery_id
    //   v = 711 + recovery_id
    // So v = 712 (decimal) = 0x2c8 (hex) means recovery_id = 1
    println!("v: 0x{:x}", signature.v);
    println!("Total signature length: 65 bytes"); // 32 + 32 + 1

    println!("\n=== Raw Signature Components ===");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "r": format!("0x{:x}", signature.r),
            "s": format!("0x{:x}", signature.s),
            "v": signature.v,
            "v_explanation": {
                "decimal": signature.v,
                "hex": format!("0x{:x}", signature.v),
                "calculation": {
                    "chain_id": CHAIN_ID,
                    "chain_id * 2": CHAIN_ID * 2,
                    "base_v": CHAIN_ID * 2 + 35,
                    "recovery_id": (signature.v - (CHAIN_ID * 2 + 35)) as u8,
                    "final_v_decimal": signature.v,
                    "final_v_hex": format!("0x{:x}", signature.v),
                }
            }
        }))?
    );

    // 3. Explain EIP-155 v calculation
    let recovery_id = (signature.v - (CHAIN_ID * 2 + 35)) as u8;
    println!("\n=== EIP-155 v Calculation ===");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "chain_id": CHAIN_ID,
            "formula": "v = chain_id * 2 + 35 + recovery_id",
            "calculation": {
                "chain_id * 2": CHAIN_ID * 2,
                "chain_id * 2 + 35": CHAIN_ID * 2 + 35,
                "recovery_id": recovery_id,
                "final_v": signature.v,
                "final_v_hex": format!("0x{:x}", signature.v),
            },
            "note": "recovery_id is either 0 or 1, used to recover the public key"
        }))?
    );

    // 4. Create signed transaction bytes
    println!("\n=== Signature Bytes ===");
    let mut r_bytes = [0u8; 32];
    let mut s_bytes = [0u8; 32];
    signature.r.to_big_endian(&mut r_bytes);
    signature.s.to_big_endian(&mut s_bytes);
    let v_byte = signature.v as u8;

    println!("r bytes: 0x{}", hex::encode(r_bytes));
    println!("s bytes: 0x{}", hex::encode(s_bytes));
    println!("v byte: 0x{:x} (dec: {})", v_byte, v_byte);
    let combined_sig = [&r_bytes[..], &s_bytes[..], &[v_byte]].concat();
    println!(
        "Combined signature bytes ({}): 0x{}",
        combined_sig.len(),
        hex::encode(&combined_sig)
    );
    let signed_tx = typed_tx.rlp_signed(&signature);

    // Send the signed transaction
    println!("\n=== Sending Signed Transaction ===");
    let pending_tx = provider.send_raw_transaction(signed_tx).await?;
    println!("Transaction hash: {}", pending_tx.tx_hash());

    // 5. Wait for confirmation and get full transaction details
    let receipt = pending_tx.await?;
    if let Some(receipt) = receipt {
        println!("\n=== Transaction Receipt ===");
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "block_number": receipt.block_number,
                "gas_used": receipt.gas_used,
                "effective_gas_price": receipt.effective_gas_price,
                "status": receipt.status,
            }))?
        );
    }

    Ok(())
}
