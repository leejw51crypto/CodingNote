use anyhow::Result;
use hex;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use web3::signing::keccak256;

use crate::{EthereumTSS, Party, TSSKeyData};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_signatures() -> Result<()> {
        // Initialize TSS
        let eth_tss = EthereumTSS::new();

        // Get private key from environment variable
        let private_key_hex = env::var("MY_FULL_PRIVATEKEY")
            .expect("Environment variable MY_FULL_PRIVATEKEY must be set");
        let private_key =
            BigUint::parse_bytes(private_key_hex.trim_start_matches("0x").as_bytes(), 16)
                .expect("Invalid private key format");

        // Setup TSS wallet with 3 parties and threshold of 2
        let key_data = eth_tss.setup_existing_key(private_key, 2, 3)?;

        // Create message hash for "hello world"
        let message = b"hello world";
        let message_hash = keccak256(message);

        // Generate common seed using same method as Python test
        let mut seed_input = message_hash.to_vec();
        seed_input.extend_from_slice(b"test_seed");
        let common_seed = Sha256::digest(&seed_input).to_vec();

        println!("\n=== Test Message Info ===");
        println!("Message: {}", String::from_utf8_lossy(message));
        println!("Message Hash: 0x{}", hex::encode(message_hash));
        println!("Common Seed: 0x{}", hex::encode(&common_seed));

        // Generate partial signatures with the derived common seed
        let mut partial_signatures = Vec::new();
        for i in 0..key_data.threshold {
            let party = &key_data.parties[i as usize];
            let partial_sig =
                eth_tss.create_partial_signature(party, &message_hash, Some(&common_seed), None)?;

            println!("\nParty {} partial signature:", i + 1);
            println!("r: 0x{}", hex::encode(partial_sig.r.to_bytes_be()));
            println!("s: 0x{}", hex::encode(&partial_sig.s));

            partial_signatures.push(partial_sig);
        }

        // Combine the partial signatures
        let combined_sig = eth_tss.combine_signatures(
            &partial_signatures,
            &key_data.parties,
            &message_hash,
            &key_data.group_public_key,
        )?;

        println!("\nCombined signature:");
        println!("r: 0x{}", hex::encode(combined_sig.r.to_bytes_be()));
        println!("s: 0x{}", hex::encode(combined_sig.s.to_bytes_be()));

        // Verify the signature
        let is_valid =
            eth_tss.verify_signature(&message_hash, &combined_sig, &key_data.group_public_key)?;
        assert!(is_valid, "Signature verification failed");

        Ok(())
    }

    #[test]
    fn test_json_data_verification() -> Result<()> {
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

            assert_eq!(our_sig.r, test_r, "R value mismatch for party {}", party.id);
            assert_eq!(our_sig.s, test_s, "S value mismatch for party {}", party.id);

            partial_signatures.push(our_sig);
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

        assert_eq!(
            combined_sig.r, test_r,
            "Combined signature R value mismatch"
        );
        assert_eq!(
            combined_sig.s, test_s,
            "Combined signature S value mismatch"
        );

        // Verify the combined signature
        let is_valid =
            eth_tss.verify_signature(&message_hash, &combined_sig, &key_data.group_public_key)?;
        assert!(is_valid, "Combined signature verification failed");

        Ok(())
    }
}
