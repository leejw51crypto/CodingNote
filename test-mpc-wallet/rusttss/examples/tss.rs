use ethers::{
    signers::LocalWallet,
    types::{Signature, H256},
    utils::hex,
};
use num_bigint::BigUint;
use rand::Rng;
use rusttss::EthereumTSS;
use std::error::Error;
use std::str::FromStr;
use web3::signing::keccak256;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables
    dotenv::dotenv().ok();

    // Initialize TSS
    let eth_tss = EthereumTSS::new();

    // Get private key from environment
    let private_key_hex =
        std::env::var("MY_FULL_PRIVATEKEY").expect("MY_FULL_PRIVATEKEY must be set");
    let private_key_hex = private_key_hex
        .strip_prefix("0x")
        .unwrap_or(&private_key_hex);
    let private_key =
        BigUint::parse_bytes(private_key_hex.as_bytes(), 16).expect("Invalid private key format");

    // Setup TSS with threshold of 2 out of 3 parties
    println!("\n=== Setting up TSS ===");
    let key_data = eth_tss.setup_existing_key(private_key.clone(), 2, 3)?;
    println!("TSS Address: {}", key_data.tss_address);
    println!("Threshold: {}", key_data.threshold);
    println!("Total Parties: {}", key_data.num_parties);

    // Generate a common seed for all parties first
    let mut rng = rand::thread_rng();
    let mut common_seed = [0u8; 32];
    rng.fill(&mut common_seed);
    println!("\n=== Common Seed ===");
    println!("Seed: 0x{}", hex::encode(&common_seed));

    // Test message to sign
    let message = b"hello world";
    let message_hash = H256::from_slice(&keccak256(message));
    println!("\n=== Message Information ===");
    println!("Original message: {}", String::from_utf8_lossy(message));
    println!("Message hash: 0x{}", hex::encode(message_hash.as_bytes()));

    // Sign directly with the private key for comparison
    println!("\n=== Direct Signature ===");

    // Create wallet from private key
    let wallet = LocalWallet::from_str(&format!("0x{}", hex::encode(private_key.to_bytes_be())))?;

    // Sign the message hash
    let signature = wallet.sign_hash(message_hash)?;
    let sig_bytes = signature.to_vec();

    println!("Direct signature (from private key):");
    println!("- r: 0x{}", hex::encode(&sig_bytes[..32]));
    println!("- s: 0x{}", hex::encode(&sig_bytes[32..64]));
    println!("- v: {}", signature.v);

    // Verify the signature recovers to the correct address
    // show hex of message_hash
    println!("message_hash: 0x{}", hex::encode(message_hash.as_bytes()));
    let recovered_address = signature.recover(message_hash)?;
    println!(
        "\nRecovered address: 0x{}",
        hex::encode(recovered_address.as_bytes())
    );
    println!("TSS address: {}", key_data.tss_address);
    println!(
        "Match: {}",
        recovered_address.as_bytes() == hex::decode(&key_data.tss_address[2..]).unwrap()
    );

    // Generate partial signatures from the first two parties (meeting threshold)
    println!("\n=== Generating Partial Signatures ===");
    let mut partial_signatures = Vec::new();
    for i in 0..key_data.threshold {
        let party = &key_data.parties[i as usize];
        let partial_sig = eth_tss.create_partial_signature(
            party,
            message_hash.as_bytes(),
            Some(&common_seed),
            None,
        )?;

        println!("Party {} partial signature:", i + 1);
        println!("- r value: 0x{}", hex::encode(partial_sig.r.to_bytes_be()));
        println!("- s value: 0x{}", hex::encode(&partial_sig.s));

        partial_signatures.push(partial_sig);
    }

    // Combine the partial signatures
    println!("\n=== Combining Signatures ===");
    let combined_sig = eth_tss.combine_signatures(
        &partial_signatures,
        &key_data.parties,
        message_hash.as_bytes(),
        &key_data.group_public_key,
    )?;

    println!("Aggregated signature (from TSS):");
    println!("- r: 0x{}", hex::encode(combined_sig.r.to_bytes_be()));
    println!("- s: 0x{}", hex::encode(combined_sig.s.to_bytes_be()));

    // Create ethers Signature from combined signature
    let mut sig_bytes = Vec::with_capacity(65);
    sig_bytes.extend_from_slice(&combined_sig.r.to_bytes_be());
    sig_bytes.extend_from_slice(&combined_sig.s.to_bytes_be());

    // Try both v values (27 and 28) and find the one that matches
    let mut correct_signature = None;
    for v in [27u8, 28u8] {
        let mut test_sig_bytes = Vec::with_capacity(65);
        test_sig_bytes.extend_from_slice(&combined_sig.r.to_bytes_be());
        test_sig_bytes.extend_from_slice(&combined_sig.s.to_bytes_be());
        test_sig_bytes.push(v);

        if let Ok(test_signature) = Signature::try_from(&test_sig_bytes[..]) {
            if let Ok(test_recovered_address) = test_signature.recover(message_hash) {
                if hex::encode(test_recovered_address.as_bytes()) == key_data.tss_address[2..] {
                    correct_signature = Some((test_signature, test_sig_bytes));
                    break;
                }
            }
        }
    }

    let (tss_signature, sig_bytes) =
        correct_signature.expect("Failed to find matching signature with either v value");

    // show hex of message_hash
    println!("message_hash: 0x{}", hex::encode(message_hash.as_bytes()));
    let tss_recovered_address = tss_signature.recover(message_hash)?;

    println!("\nSignature Verification and Comparison:");
    println!("\n1. Direct Key Signature:");
    println!("Address: 0x{}", hex::encode(recovered_address.as_bytes()));
    println!("Signature bytes: 0x{}", hex::encode(&sig_bytes));
    println!("- r: 0x{}", hex::encode(&sig_bytes[..32]));
    println!("- s: 0x{}", hex::encode(&sig_bytes[32..64]));
    println!("- v: {}", signature.v);

    println!("\n2. TSS Aggregated Signature:");
    println!(
        "Address: 0x{}",
        hex::encode(tss_recovered_address.as_bytes())
    );
    println!("Signature bytes: 0x{}", hex::encode(&sig_bytes));
    println!("- r: 0x{}", hex::encode(&sig_bytes[..32]));
    println!("- s: 0x{}", hex::encode(&sig_bytes[32..64]));
    println!("- v: {}", tss_signature.v);

    println!("\n3. Verification Results:");
    println!(
        "Recovered address: 0x{}",
        hex::encode(recovered_address.as_bytes())
    );
    println!(
        "TSS recovered address: 0x{}",
        hex::encode(tss_recovered_address.as_bytes())
    );
    println!(
        "Addresses match: {}",
        hex::encode(recovered_address.as_bytes()) == hex::encode(tss_recovered_address.as_bytes())
    );
    println!(
        "Signatures match: {}",
        hex::encode(&sig_bytes) == hex::encode(&sig_bytes)
    );

    Ok(())
}
