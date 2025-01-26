use anyhow::Result;
use dotenv::dotenv;
use ethers::abi::AbiEncode;
use ethers::{
    prelude::*,
    types::{transaction::eip2718::TypedTransaction, Signature},
    utils::{format_ether, parse_ether},
};
use num_bigint::BigUint;
use rand::Rng;
use rusttss::{check_sufficient_balance, verify_encoded_tx, EthereumTSS, TSSKeyData as KeyData};
use std::env;
use std::str::FromStr;
use std::sync::Arc;

// Setup TSS wallet using an existing private key
fn setup_tss_wallet(private_key: BigUint, threshold: u32, num_parties: u32) -> Result<KeyData> {
    let eth_tss = EthereumTSS::new();
    println!("\n=== Setting up TSS ===");
    let key_data = eth_tss.setup_existing_key(private_key, threshold, num_parties)?;
    println!("TSS Address: {}", key_data.tss_address);
    println!("Threshold: {}", key_data.threshold);
    println!("Total Parties: {}", key_data.num_parties);
    Ok(key_data)
}

// Generate and combine TSS signatures
fn create_tss_signature(
    eth_tss: &EthereumTSS,
    key_data: &KeyData,
    message_hash: &[u8],
    chain_id: u64,
) -> Result<Signature> {
    let mut rng = rand::thread_rng();
    let mut common_seed = [0u8; 32];
    rng.fill(&mut common_seed);

    println!("\n=== Common Seed ===");
    println!("Seed: 0x{}", hex::encode(common_seed));

    // Generate partial signatures from the first parties meeting threshold
    println!("\n=== Generating Partial Signatures ===");
    let partial_signatures =
        generate_partial_signatures(eth_tss, key_data, message_hash, &common_seed)?;

    // Combine the partial signatures
    println!("\n=== Combining Signatures ===");
    let combined_sig = eth_tss.combine_signatures(
        &partial_signatures,
        &key_data.parties,
        message_hash,
        &key_data.group_public_key,
    )?;

    create_ethereum_signature(combined_sig, message_hash, chain_id, &key_data.tss_address)
}

fn generate_partial_signatures(
    eth_tss: &EthereumTSS,
    key_data: &KeyData,
    message_hash: &[u8],
    common_seed: &[u8],
) -> Result<Vec<rusttss::PartialSignature>> {
    let mut partial_signatures = Vec::new();
    for i in 0..key_data.threshold {
        let party = &key_data.parties[i as usize];
        let partial_sig =
            eth_tss.create_partial_signature(party, message_hash, Some(common_seed), None)?;

        println!("Party {} partial signature:", i + 1);
        println!("- r value: 0x{}", hex::encode(partial_sig.r.to_bytes_be()));
        println!("- s value: 0x{}", hex::encode(&partial_sig.s));

        partial_signatures.push(partial_sig);
    }
    Ok(partial_signatures)
}

fn create_ethereum_signature(
    combined_sig: rusttss::CombinedSignature,
    message_hash: &[u8],
    chain_id: u64,
    tss_address: &str,
) -> Result<Signature> {
    println!("Aggregated signature (from TSS):");
    println!("- r: 0x{}", hex::encode(combined_sig.r.to_bytes_be()));
    println!("- s: 0x{}", hex::encode(combined_sig.s.to_bytes_be()));

    let mut s = combined_sig.s.to_bytes_be();
    let normalized_s = normalize_signature_s(&s);
    s = normalized_s.to_bytes_be();

    find_valid_signature(combined_sig.r, &s, message_hash, chain_id, tss_address)
}

fn normalize_signature_s(s: &[u8]) -> BigUint {
    // SECP256K1 curve order
    let n = BigUint::parse_bytes(
        b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
        16,
    )
    .unwrap();
    let half_n = n.clone() >> 1;

    // Always normalize s to the lower half
    let s_bigint = BigUint::from_bytes_be(s);
    if s_bigint > half_n {
        n - s_bigint
    } else {
        s_bigint
    }
}

fn find_valid_signature(
    r: BigUint,
    s: &[u8],
    message_hash: &[u8],
    chain_id: u64,
    tss_address: &str,
) -> Result<Signature> {
    for base_v in [27u64, 28u64] {
        let recovery_id = base_v - 27;
        let chain_v = chain_id * 2 + 35 + recovery_id;

        let test_signature = Signature {
            r: U256::from_big_endian(&r.to_bytes_be()),
            s: U256::from_big_endian(s),
            v: chain_v,
        };

        if let Ok(recovered_address) = test_signature.recover(H256::from_slice(message_hash)) {
            if hex::encode(recovered_address.as_bytes()) == tss_address[2..] {
                println!("\n=== Found Valid Signature ===");
                println!("Base v: {}", base_v);
                println!("Recovery ID: {}", recovery_id);
                println!("Chain ID: {}", chain_id);
                println!("Final v: {} (0x{:x})", chain_v, chain_v);
                println!("Final s (normalized): 0x{}", hex::encode(s));
                println!("Final r: 0x{:x}", test_signature.r);
                return Ok(test_signature);
            }
        }
    }

    Err(anyhow::anyhow!("Failed to find valid signature"))
}

async fn setup_provider(rpc_endpoint: &str) -> Result<Arc<Provider<Http>>> {
    Ok(Arc::new(Provider::<Http>::try_from(rpc_endpoint)?))
}

async fn print_balances(
    provider: &Provider<Http>,
    sender_address: Address,
    receiver_address: Address,
    prefix: &str,
) -> Result<()> {
    let sender_balance = provider.get_balance(sender_address, None).await?;
    let receiver_balance = provider.get_balance(receiver_address, None).await?;

    println!("\n=== {} Balances ===", prefix);
    println!("Sender balance: {} TCRO", format_ether(sender_balance));
    println!("Receiver balance: {} TCRO", format_ether(receiver_balance));
    Ok(())
}

async fn create_transaction(
    provider: &Provider<Http>,
    sender_address: Address,
    to_address: Address,
    amount: U256,
) -> Result<TransactionRequest> {
    let gas_price = provider.get_gas_price().await?;
    let gas_limit = U256::from(21000);

    // Check balance
    check_sufficient_balance(provider, sender_address, amount, gas_price, gas_limit).await?;

    // Create transaction request
    let nonce = provider.get_transaction_count(sender_address, None).await?;
    Ok(TransactionRequest::new()
        .nonce(nonce)
        .to(to_address)
        .value(amount)
        .gas_price(gas_price)
        .gas(gas_limit)
        .chain_id(338)
        .from(sender_address))
}

async fn send_and_confirm_transaction(
    provider: &Provider<Http>,
    encoded_tx: Bytes,
) -> Result<Option<TransactionReceipt>> {
    let pending_tx = provider.send_raw_transaction(encoded_tx).await?;
    println!("\n=== Transaction Sent ===");
    println!("Transaction hash: 0x{}", pending_tx.tx_hash().encode_hex());

    let receipt = pending_tx.await?;
    if let Some(receipt) = receipt.as_ref() {
        print_transaction_receipt(receipt).await?;
    }
    Ok(receipt)
}

async fn print_transaction_receipt(receipt: &TransactionReceipt) -> Result<()> {
    println!("\n=== Transaction Confirmed ===");
    if let Some(block_number) = receipt.block_number {
        println!("Block number: {}", block_number);
    }

    if let (Some(gas_used), Some(effective_gas_price)) =
        (receipt.gas_used, receipt.effective_gas_price)
    {
        let gas_cost = gas_used * effective_gas_price;
        println!(
            "Gas used: {} (Cost: {} TCRO)",
            gas_used,
            format_ether(gas_cost)
        );
        println!("Gas cost: {} TCRO", format_ether(gas_cost));
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // Initialize TSS
    let eth_tss = EthereumTSS::new();

    // Get private key from environment and setup TSS wallet
    let private_key_hex =
        std::env::var("MY_FULL_PRIVATEKEY").expect("MY_FULL_PRIVATEKEY must be set");
    let private_key_hex = private_key_hex
        .strip_prefix("0x")
        .unwrap_or(&private_key_hex);
    let private_key =
        BigUint::parse_bytes(private_key_hex.as_bytes(), 16).expect("Invalid private key format");

    // Setup TSS wallet with threshold of 2 out of 3 parties
    let key_data = setup_tss_wallet(private_key.clone(), 2, 3)?;

    // Configure provider for Cronos testnet
    const RPC_ENDPOINT: &str = "https://evm-t3.cronos.org/";
    const CHAIN_ID: u64 = 338;

    let provider = setup_provider(RPC_ENDPOINT).await?;
    let to_address =
        Address::from_str(&env::var("MY_TO_ADDRESS").expect("MY_TO_ADDRESS must be set"))?;
    let sender_address = Address::from_str(&key_data.tss_address)?;

    // Print initial balances
    print_balances(&provider, sender_address, to_address, "Initial").await?;

    // Create transaction
    let amount = parse_ether("0.1")?;
    let tx = create_transaction(&provider, sender_address, to_address, amount).await?;
    println!("Transaction: {:?}", tx);

    // Get transaction hash and create TSS signature
    let tx_hash = tx.sighash();
    let message_hash = H256::from_slice(tx_hash.as_bytes());
    println!("\n=== Message Information ===");
    println!(
        "Transaction hash: 0x{}",
        hex::encode(message_hash.as_bytes())
    );

    // Create TSS signature
    let tss_signature =
        create_tss_signature(&eth_tss, &key_data, message_hash.as_bytes(), CHAIN_ID)?;

    // Create and encode the transaction
    let typed_tx = TypedTransaction::Legacy(tx);
    let encoded_tx = typed_tx.rlp_signed(&tss_signature);

    // Verify the encoded transaction
    let is_valid = verify_encoded_tx(&encoded_tx, &tss_signature, CHAIN_ID)?;
    if !is_valid {
        anyhow::bail!("Encoded transaction signature verification failed!");
    }
    println!("✅ Encoded transaction signature verification passed!");

    // Send transaction and wait for confirmation
    let receipt = send_and_confirm_transaction(&provider, encoded_tx).await?;
    if receipt.is_some() {
        print_balances(&provider, sender_address, to_address, "Final").await?;
    }

    Ok(())
}
