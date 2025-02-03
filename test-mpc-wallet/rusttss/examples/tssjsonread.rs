use anyhow::Result;
use hex;
use num_bigint::BigUint;
use rusttss::{EthereumTSS, Party, TSSKeyData};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Serialize, Deserialize)]
struct TSSTestData {
    setup: TSSSetup,
    parties: Vec<PartyData>,
    test_signing: TestSigning,
}

#[derive(Debug, Serialize, Deserialize)]
struct TSSSetup {
    private_key: String,
    threshold: u32,
    num_parties: u32,
    tss_address: String,
    group_public_key: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PartyData {
    id: u32,
    private_share: String,
    public_key: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestSigning {
    message: String,
    message_hash: String,
    common_seed: String,
    partial_signatures: Vec<PartialSignatureData>,
    combined_signature: CombinedSignatureData,
}

#[derive(Debug, Serialize, Deserialize)]
struct PartialSignatureData {
    party_id: u32,
    r: String,
    s: String,
    k: String,
    R: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CombinedSignatureData {
    r: String,
    s: String,
    v: u64,
}

fn hex_to_biguint(hex_str: &str) -> Result<BigUint> {
    let hex = hex_str.trim_start_matches("0x");
    Ok(BigUint::parse_bytes(hex.as_bytes(), 16).ok_or_else(|| anyhow::anyhow!("Invalid hex"))?)
}

fn hex_to_bytes(hex_str: &str) -> Result<Vec<u8>> {
    let hex = hex_str.trim_start_matches("0x");
    Ok(hex::decode(hex)?)
}

fn main() -> Result<()> {
    // Read and parse the test data JSON
    let json_data = fs::read_to_string("../tss_test_data.json")?;
    let test_data: TSSTestData = serde_json::from_str(&json_data)?;

    println!("=== Reading TSS Test Data ===");
    println!("TSS Address: {}", test_data.setup.tss_address);
    println!("Threshold: {}", test_data.setup.threshold);
    println!("Total Parties: {}", test_data.setup.num_parties);

    // Initialize TSS
    let eth_tss = EthereumTSS::new();

    // Convert test data parties to TSS parties
    let parties: Vec<Party> = test_data
        .parties
        .iter()
        .map(|p| Party {
            id: p.id,
            xi: hex_to_biguint(&p.private_share).unwrap(),
            public_key: hex_to_bytes(&p.public_key).unwrap(),
        })
        .collect();

    // Create TSS key data
    let key_data = TSSKeyData {
        parties,
        tss_address: test_data.setup.tss_address.clone(),
        group_public_key: hex_to_bytes(&test_data.setup.group_public_key)?,
        threshold: test_data.setup.threshold,
        num_parties: test_data.setup.num_parties,
    };

    // Get message hash and common seed
    let message_hash = hex_to_bytes(&test_data.test_signing.message_hash)?;
    let common_seed = hex_to_bytes(&test_data.test_signing.common_seed)?;

    println!("\n=== Verifying Partial Signatures ===");
    let mut partial_signatures = Vec::new();

    // Generate and verify each partial signature
    for (i, party) in key_data
        .parties
        .iter()
        .take(key_data.threshold as usize)
        .enumerate()
    {
        let test_sig = &test_data.test_signing.partial_signatures[i];

        // Create partial signature using our implementation
        let our_sig =
            eth_tss.create_partial_signature(party, &message_hash, Some(&common_seed), None)?;

        // Compare with test data
        let test_r = hex_to_biguint(&test_sig.r)?;
        let test_s = hex_to_bytes(&test_sig.s)?;

        println!("\nParty {} signature verification:", party.id);
        println!("Our r    : 0x{}", hex::encode(our_sig.r.to_bytes_be()));
        println!("Test r   : 0x{}", hex::encode(test_r.to_bytes_be()));
        println!("r matches: {}", our_sig.r == test_r);
        println!("\nOur s    : 0x{}", hex::encode(&our_sig.s));
        println!("Test s   : 0x{}", hex::encode(&test_s));
        println!("s matches: {}", our_sig.s == test_s);

        // Also show k and R values from test data
        println!("\nTest k   : {}", test_sig.k);
        println!("Test R   : {}", test_sig.R);

        // Use the test signature for combining
        partial_signatures.push(our_sig);
    }

    println!("\n=== Combining Signatures ===");
    println!("Message hash: 0x{}", hex::encode(&message_hash));
    println!(
        "Group public key: 0x{}",
        hex::encode(&key_data.group_public_key)
    );

    // Print individual partial signatures before combining
    for (i, sig) in partial_signatures.iter().enumerate() {
        println!("\nPartial signature {} components:", i + 1);
        println!("r: 0x{}", hex::encode(sig.r.to_bytes_be()));
        println!("s: 0x{}", hex::encode(&sig.s));
    }

    // Print test data partial signatures
    println!("\nTest data partial signatures:");
    for sig in test_data
        .test_signing
        .partial_signatures
        .iter()
        .take(key_data.threshold as usize)
    {
        println!("\nParty {} components:", sig.party_id);
        println!("r: {}", sig.r);
        println!("s: {}", sig.s);
    }

    // Combine signatures using our implementation
    let mut combined_sig = eth_tss.combine_signatures(
        &partial_signatures,
        &key_data.parties,
        &message_hash,
        &key_data.group_public_key,
    )?;

    // Normalize s value according to EIP-2
    let curve_order = BigUint::parse_bytes(
        b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
        16,
    )
    .unwrap();
    let half_n = curve_order.clone() >> 1;

    // If s > half_n, set s = n - s
    if combined_sig.s > half_n {
        combined_sig.s = curve_order - combined_sig.s;
    }

    // Compare with test data
    let test_r = hex_to_biguint(&test_data.test_signing.combined_signature.r)?;
    let test_s = hex_to_biguint(&test_data.test_signing.combined_signature.s)?;

    println!("\nCombined signature verification:");
    println!("Our r    : 0x{}", hex::encode(combined_sig.r.to_bytes_be()));
    println!("Test r   : 0x{}", hex::encode(test_r.to_bytes_be()));
    println!("r matches: {}", combined_sig.r == test_r);
    println!(
        "\nOur s    : 0x{}",
        hex::encode(combined_sig.s.to_bytes_be())
    );
    println!("Test s   : 0x{}", hex::encode(test_s.to_bytes_be()));
    println!("s matches: {}", combined_sig.s == test_s);
    println!(
        "\nTest v   : {}",
        test_data.test_signing.combined_signature.v
    );

    // Verify the combined signature
    let is_valid =
        eth_tss.verify_signature(&message_hash, &combined_sig, &key_data.group_public_key)?;
    println!("\n=== Final Verification ===");
    println!("Signature is valid: {}", is_valid);

    Ok(())
}
