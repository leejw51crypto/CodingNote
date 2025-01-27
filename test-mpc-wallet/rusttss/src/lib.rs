use anyhow::{anyhow, Result};
use ethers::{prelude::*, types::Signature};
use num_bigint::{BigUint, RandBigInt};
use num_traits::{One, Zero};
use secp256k1::{PublicKey, Secp256k1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use web3::signing::{keccak256, recover};

// Constants for secp256k1
lazy_static::lazy_static! {
    static ref CURVE_ORDER: BigUint = BigUint::parse_bytes(b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16).unwrap();
    static ref GENERATOR_X: BigUint = BigUint::parse_bytes(b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16).unwrap();
    static ref GENERATOR_Y: BigUint = BigUint::parse_bytes(b"483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16).unwrap();
}

#[derive(Debug, Clone)]
pub struct PartialSignature {
    pub r: BigUint,
    pub s: Vec<u8>,
    pub k: BigUint,
    pub r_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CombinedSignature {
    pub r: BigUint,
    pub s: BigUint,
}

// Custom serialization for BigUint
mod biguint_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(biguint: &BigUint, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = biguint.to_bytes_be();
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BigUint, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let hex_str: String = String::deserialize(deserializer)?;
        let bytes = hex::decode(&hex_str).map_err(|e| D::Error::custom(e.to_string()))?;
        Ok(BigUint::from_bytes_be(&bytes))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Party {
    pub id: u32,
    #[serde(with = "biguint_serde")]
    pub xi: BigUint, // Secret share of the private key
    pub public_key: Vec<u8>, // Public key share
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSSKeyData {
    pub parties: Vec<Party>,
    pub tss_address: String,
    pub group_public_key: Vec<u8>,
    pub threshold: u32,
    pub num_parties: u32,
}

// Public alias for external use
pub type KeyData = TSSKeyData;

pub struct ThresholdSignatureScheme {
    secp: Secp256k1<secp256k1::All>,
}

impl ThresholdSignatureScheme {
    pub fn new() -> Self {
        Self {
            secp: Secp256k1::new(),
        }
    }

    fn mod_inverse(&self, a: &BigUint) -> Option<BigUint> {
        if a.is_zero() {
            return None;
        }
        Some(a.modpow(&(&*CURVE_ORDER - 2u32), &CURVE_ORDER))
    }

    pub fn lagrange_coefficient(&self, parties: &[Party], party_id: u32, x: u32) -> BigUint {
        let mut num = BigUint::one();
        let mut den = BigUint::one();

        for party in parties {
            if party.id != party_id {
                // Calculate numerator: x - x_m
                let x_m = BigUint::from(party.id);
                let term = if x == 0 {
                    // For x = 0: -x_m mod n = n - x_m
                    &*CURVE_ORDER - &x_m
                } else {
                    // For x != 0: (x - x_m) mod n
                    let x_big = BigUint::from(x);
                    if x_big >= x_m {
                        x_big - &x_m
                    } else {
                        &*CURVE_ORDER - (&x_m - x_big)
                    }
                };
                num = (&num * &term) % &*CURVE_ORDER;

                // Calculate denominator: x_j - x_m
                let x_j = BigUint::from(party_id);
                let term = if x_j >= x_m {
                    &x_j - &x_m
                } else {
                    &*CURVE_ORDER - (&x_m - &x_j)
                };
                den = (&den * &term) % &*CURVE_ORDER;
            }
        }

        // Calculate modular multiplicative inverse of denominator
        let den_inv = den.modpow(&(&*CURVE_ORDER - BigUint::from(2u32)), &CURVE_ORDER);

        // Final result
        (&num * &den_inv) % &*CURVE_ORDER
    }

    fn biguint_to_32bytes(value: &BigUint) -> Vec<u8> {
        let mut bytes = value.to_bytes_be();
        match bytes.len().cmp(&32) {
            std::cmp::Ordering::Less => {
                let mut padded = vec![0u8; 32 - bytes.len()];
                padded.extend_from_slice(&bytes);
                bytes = padded;
            }
            std::cmp::Ordering::Greater => {
                // Take only the last 32 bytes if longer
                bytes = bytes[bytes.len() - 32..].to_vec();
            }
            std::cmp::Ordering::Equal => {}
        }
        bytes
    }

    fn bytes_to_biguint(bytes: &[u8]) -> BigUint {
        BigUint::from_bytes_be(bytes)
    }

    pub fn compute_r_point(&self, k: &BigUint) -> (BigUint, Vec<u8>) {
        let k_bytes = Self::biguint_to_32bytes(k);
        let secp = Secp256k1::new();
        let secret_key = secp256k1::SecretKey::from_slice(&k_bytes).unwrap();
        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        let r_bytes = public_key.serialize_uncompressed();
        let x_coordinate = &r_bytes[1..33];
        let r = Self::bytes_to_biguint(x_coordinate) % &*CURVE_ORDER;
        (r, r_bytes.to_vec())
    }

    pub fn create_partial_signature(
        &self,
        party: &Party,
        message_hash: &[u8],
        common_seed: Option<&[u8]>,
        r_point: Option<&BigUint>,
    ) -> Result<PartialSignature> {
        // Use provided k value from common seed
        let k_value = if let Some(seed) = common_seed {
            // Generate k value from common seed and message hash
            let mut hasher = Sha256::new();
            hasher.update(message_hash);
            hasher.update(seed);
            let hash = hasher.finalize();
            BigUint::from_bytes_be(hash.as_slice()) % &*CURVE_ORDER
        } else {
            return Err(anyhow!("Common seed (k value) is required"));
        };

        // Use provided R point or generate one
        let (r_value, r_bytes) = if let Some(r) = r_point {
            // Convert R point to uncompressed format
            let r_bytes = Self::biguint_to_32bytes(r);
            let mut uncompressed = vec![0x04]; // Uncompressed point prefix
            uncompressed.extend_from_slice(&r_bytes);
            // Add y coordinate (zeros for now, not needed for signature)
            uncompressed.extend_from_slice(&[0u8; 32]);
            (r.clone(), uncompressed)
        } else {
            self.compute_r_point(&k_value)
        };

        // Convert message hash to BigUint
        let z = BigUint::from_bytes_be(message_hash);

        // Calculate k_inv
        let k_inv = self
            .mod_inverse(&k_value)
            .ok_or(anyhow!("Failed to compute k inverse"))?;

        // Each party computes their share of s = k^(-1)(z + r*x)
        // where x is their secret share (party.xi)
        let r_x_priv = (&r_value * &party.xi) % &*CURVE_ORDER;

        println!("Party {} partial signature components:", party.id);
        println!("k_inv: 0x{}", hex::encode(k_inv.to_bytes_be()));
        println!("r_x_priv: 0x{}", hex::encode(r_x_priv.to_bytes_be()));

        let s_i = (k_inv.clone() * (z + r_x_priv)) % &*CURVE_ORDER;
        println!("s_i before mod: 0x{}", hex::encode(s_i.to_bytes_be()));

        // Convert s_i to bytes
        let s_bytes = Self::biguint_to_32bytes(&s_i);

        Ok(PartialSignature {
            r: r_value,
            s: s_bytes.to_vec(),
            k: k_value,
            r_bytes,
        })
    }

    pub fn verify_signature(
        &self,
        message_hash: &[u8],
        signature: &CombinedSignature,
        group_public_key: &[u8],
    ) -> Result<bool> {
        // Create signature bytes
        let mut sig_bytes = Vec::with_capacity(65);
        sig_bytes.extend_from_slice(&Self::biguint_to_32bytes(&signature.r));
        sig_bytes.extend_from_slice(&Self::biguint_to_32bytes(&signature.s));

        // Try both v values if needed
        for v in [27, 28] {
            let recovery_id = v - 27;
            if let Ok(recovered_key) = recover(message_hash, &sig_bytes[..64], recovery_id) {
                let recovered_address = format!("0x{}", hex::encode(recovered_key.as_bytes()));
                let expected_address = self.derive_ethereum_address(group_public_key);
                println!("Verify - v value: {}", v);
                println!("Verify - Recovery ID: {}", recovery_id);
                println!("Verify - Recovered address: {}", recovered_address);
                println!("Verify - Expected address: {}", expected_address);
                if recovered_address.to_lowercase() == expected_address.to_lowercase() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    pub fn combine_signatures(
        &self,
        partial_signatures: &[PartialSignature],
        parties: &[Party],
        _message_hash: &[u8],
        group_public_key: &[u8],
    ) -> Result<CombinedSignature> {
        // Verify all r values are the same
        let first_r = &partial_signatures[0].r;
        let first_r_bytes = &partial_signatures[0].r_bytes;
        for sig in &partial_signatures[1..] {
            if &sig.r != first_r || &sig.r_bytes != first_r_bytes {
                return Err(anyhow!("Inconsistent R values in partial signatures"));
            }
        }

        println!("\n=== Combining Signatures ===");
        println!("\nCommon Values:");
        println!("R value: 0x{}", hex::encode(first_r.to_bytes_be()));
        println!(
            "k value: 0x{}",
            hex::encode(partial_signatures[0].k.to_bytes_be())
        );

        // Combine the partial s values using Lagrange interpolation
        let mut s_combined = BigUint::zero();
        let active_parties = &parties[..partial_signatures.len()];

        println!("\nPartial Signatures:");
        for (i, sig) in partial_signatures.iter().enumerate() {
            let party = &active_parties[i];
            let lambda_i = self.lagrange_coefficient(active_parties, party.id, 0);
            let s_i = BigUint::from_bytes_be(&sig.s);

            // Print individual signature components
            println!("\nParty {} signature piece:", party.id);
            println!("s_{}: 0x{}", party.id, hex::encode(&sig.s));
            println!("λ_{}: 0x{}", party.id, hex::encode(lambda_i.to_bytes_be()));

            // Calculate weighted signature piece
            let weighted_s = (lambda_i.clone() * s_i.clone()) % &*CURVE_ORDER;
            println!(
                "λ_{} * s_{}: 0x{}",
                party.id,
                party.id,
                hex::encode(weighted_s.to_bytes_be())
            );

            // Add to combined signature
            s_combined = (s_combined + weighted_s) % &*CURVE_ORDER;

            // Show running total
            println!("Running total: 0x{}", hex::encode(s_combined.to_bytes_be()));
        }

        println!("\nFinal Combined Signature:");
        println!("r: 0x{}", hex::encode(first_r.to_bytes_be()));
        println!("s: 0x{}", hex::encode(s_combined.to_bytes_be()));
        println!("\nGroup public key: 0x{}", hex::encode(group_public_key));

        Ok(CombinedSignature {
            r: first_r.clone(),
            s: s_combined,
        })
    }

    pub fn derive_ethereum_address(&self, public_key: &[u8]) -> String {
        // Make sure we're using the uncompressed public key format
        let key_to_hash = if public_key[0] == 0x04 {
            // Already uncompressed format, skip the prefix
            &public_key[1..]
        } else {
            // Compressed or other format, use as is
            public_key
        };
        let hash = keccak256(key_to_hash);
        let address = &hash[12..];
        format!("0x{}", hex::encode(address))
    }
}

impl Default for ThresholdSignatureScheme {
    fn default() -> Self {
        Self::new()
    }
}

pub struct EthereumTSS {
    tss: ThresholdSignatureScheme,
}

impl EthereumTSS {
    pub fn new() -> Self {
        Self {
            tss: ThresholdSignatureScheme::new(),
        }
    }

    fn biguint_to_bytes(value: &BigUint) -> Vec<u8> {
        let mut bytes = value.to_bytes_be();
        if bytes.len() < 32 {
            let mut padded = vec![0u8; 32 - bytes.len()];
            padded.extend_from_slice(&bytes);
            bytes = padded;
        }
        bytes
    }

    pub fn setup_existing_key(
        &self,
        private_key: BigUint,
        threshold: u32,
        num_parties: u32,
    ) -> Result<TSSKeyData> {
        if threshold > num_parties {
            return Err(anyhow!(
                "Threshold cannot be greater than number of parties"
            ));
        }

        // Debug: Show first and last byte of private key
        let pk_bytes = Self::biguint_to_bytes(&private_key);
        println!("\n=== Private Key Debug Info ===");
        println!(
            "Private key: 0x{:02x}..........{:02x} ({} bytes)",
            pk_bytes[0],
            pk_bytes[31],
            pk_bytes.len()
        );

        let mut rng = rand::thread_rng();
        let mut parties = Vec::new();
        let mut coefficients = vec![private_key.clone()];

        // Generate random coefficients for the polynomial
        for _ in 1..threshold {
            let coeff = rng.gen_biguint_below(&CURVE_ORDER);
            coefficients.push(coeff);
        }

        println!("\n=== Participant Shares Debug Info ===");
        // Generate shares for each party
        for i in 1..=num_parties {
            let mut share = coefficients[0].clone();
            let x = BigUint::from(i as u64);
            let mut x_power = x.clone();

            // Simplified loop without enumerate
            for coeff in coefficients.iter().skip(1) {
                share = (share + (coeff * &x_power)) % &*CURVE_ORDER;
                x_power = (x_power * &x) % &*CURVE_ORDER;
            }

            let share_bytes = Self::biguint_to_bytes(&share);
            println!(
                "Participant {}: 0x{:02x}..........{:02x} ({} bytes)",
                i,
                share_bytes[0],
                share_bytes[31],
                share_bytes.len()
            );

            let secret_key = secp256k1::SecretKey::from_slice(&share_bytes)?;
            let public_key = PublicKey::from_secret_key(&self.tss.secp, &secret_key);
            let public_key_bytes = public_key.serialize_uncompressed().to_vec();

            parties.push(Party {
                id: i,
                xi: share,
                public_key: public_key_bytes,
            });
        }

        // Compute group public key
        let group_secret_bytes = Self::biguint_to_bytes(&private_key);
        let group_secret = secp256k1::SecretKey::from_slice(&group_secret_bytes)?;
        let group_public_key = PublicKey::from_secret_key(&self.tss.secp, &group_secret);
        let group_public_key_bytes = group_public_key.serialize_uncompressed().to_vec();

        // Derive Ethereum address from the public key
        let hash = keccak256(&group_public_key_bytes[1..]); // Skip the '04' prefix
        let address = &hash[12..];
        let hex_address = hex::encode(address);

        // print the hex_address
        println!("\n=== Address Info ===");
        println!("Hex address: {}", hex_address);
        println!("TSS Address: 0x{}", hex_address);
        println!("Threshold: {}", threshold);
        println!("Total Parties: {}", num_parties);

        Ok(TSSKeyData {
            parties,
            tss_address: format!("0x{}", hex_address),
            group_public_key: group_public_key_bytes,
            threshold,
            num_parties,
        })
    }

    pub fn create_partial_signature(
        &self,
        party: &Party,
        message_hash: &[u8],
        common_seed: Option<&[u8]>,
        r_point: Option<&BigUint>,
    ) -> Result<PartialSignature> {
        self.tss
            .create_partial_signature(party, message_hash, common_seed, r_point)
    }

    pub fn combine_signatures(
        &self,
        partial_signatures: &[PartialSignature],
        parties: &[Party],
        message_hash: &[u8],
        group_public_key: &[u8],
    ) -> Result<CombinedSignature> {
        self.tss
            .combine_signatures(partial_signatures, parties, message_hash, group_public_key)
    }

    pub fn verify_signature(
        &self,
        message_hash: &[u8],
        signature: &CombinedSignature,
        group_public_key: &[u8],
    ) -> Result<bool> {
        self.tss
            .verify_signature(message_hash, signature, group_public_key)
    }

    pub fn calculate_v_values(&self, _base_v: u64, chain_id: u64) -> Vec<u64> {
        let mut possible_v_values = Vec::new();
        for base in [27u64, 28u64] {
            let v = chain_id * 2 + 35 + (base - 27);
            possible_v_values.push(v);
        }
        possible_v_values
    }

    pub fn recover_signer(
        &self,
        message_hash: &[u8; 32],
        r: &[u8],
        s: &[u8],
        chain_id: u64,
    ) -> Result<String> {
        let mut sig_bytes = Vec::with_capacity(65);

        // Pad r to 32 bytes
        let mut r_padded = vec![0u8; 32];
        if r.len() <= 32 {
            let r_start = 32 - r.len();
            r_padded[r_start..].copy_from_slice(r);
        } else {
            r_padded.copy_from_slice(&r[..32]);
        }
        sig_bytes.extend_from_slice(&r_padded);

        // Pad s to 32 bytes
        let mut s_padded = vec![0u8; 32];
        if s.len() <= 32 {
            let s_start = 32 - s.len();
            s_padded[s_start..].copy_from_slice(s);
        } else {
            s_padded.copy_from_slice(&s[..32]);
        }
        sig_bytes.extend_from_slice(&s_padded);

        // Try both base v values and chain v values
        for base_v in [27u64, 28u64] {
            // Try with base v
            sig_bytes.push((base_v - 27) as u8);
            if let Ok(recovered_key) = recover(message_hash, &sig_bytes[..64], (base_v - 27) as i32)
            {
                let recovered_address = format!("0x{}", hex::encode(recovered_key.as_bytes()));
                println!(
                    "Recovered with base_v {}: 0x{}",
                    base_v,
                    hex::encode(hex::decode(&recovered_address[2..]).unwrap())
                );
                return Ok(recovered_address);
            }
            sig_bytes.pop();

            // Try with chain v
            let chain_v = chain_id * 2 + 35 + (base_v - 27);
            sig_bytes.push(chain_v as u8);
            if let Ok(recovered_key) = recover(message_hash, &sig_bytes[..64], (base_v - 27) as i32)
            {
                let recovered_address = format!("0x{}", hex::encode(recovered_key.as_bytes()));
                println!(
                    "Recovered with chain_v {}: 0x{}",
                    chain_v,
                    hex::encode(hex::decode(&recovered_address[2..]).unwrap())
                );
                return Ok(recovered_address);
            }
            sig_bytes.pop();
        }

        Err(anyhow!("Could not recover signer with any v value"))
    }
}

impl Default for EthereumTSS {
    fn default() -> Self {
        Self::new()
    }
}

// Transaction-related functions
pub async fn check_sufficient_balance(
    provider: &Provider<Http>,
    sender_address: Address,
    amount: U256,
    gas_price: U256,
    gas_limit: U256,
) -> Result<()> {
    let balance = provider.get_balance(sender_address, None).await?;
    let total_cost = amount + (gas_price * gas_limit);

    if balance < total_cost {
        return Err(anyhow!(
            "Insufficient balance: have {} wei, need {} wei (tx: {} wei, gas: {} wei)",
            balance,
            total_cost,
            amount,
            gas_price * gas_limit
        ));
    }
    Ok(())
}

pub fn verify_encoded_tx(
    encoded_tx: &Bytes,
    expected_sig: &Signature,
    chain_id: u64,
) -> Result<bool> {
    // Decode the RLP encoded transaction
    let decoded = rlp::Rlp::new(encoded_tx.as_ref());

    // For EIP-155 transaction, verify all components
    let nonce = U256::from_big_endian(&decoded.val_at::<Vec<u8>>(0)?);
    let gas_price = U256::from_big_endian(&decoded.val_at::<Vec<u8>>(1)?);
    let gas_limit = U256::from_big_endian(&decoded.val_at::<Vec<u8>>(2)?);
    let to = decoded.val_at::<Vec<u8>>(3)?;
    let value = U256::from_big_endian(&decoded.val_at::<Vec<u8>>(4)?);
    let data = decoded.val_at::<Vec<u8>>(5)?;
    let v_bytes = decoded.val_at::<Vec<u8>>(6)?;
    let r_bytes = decoded.val_at::<Vec<u8>>(7)?;
    let s_bytes = decoded.val_at::<Vec<u8>>(8)?;

    let v = U256::from_big_endian(&v_bytes);
    let r = U256::from_big_endian(&r_bytes);
    let s = U256::from_big_endian(&s_bytes);

    println!("\n=== Verifying EIP-155 Encoded Transaction ===");
    println!("Transaction Components:");
    println!("nonce: 0x{:x}", nonce);
    println!("gas_price: 0x{:x}", gas_price);
    println!("gas_limit: 0x{:x}", gas_limit);
    println!("to: 0x{}", hex::encode(&to));
    println!("value: 0x{:x}", value);
    println!("data: 0x{}", hex::encode(&data));

    println!("\nSignature Components:");
    println!("v (encoded): {} (0x{:x})", v, v);
    println!("v (expected): {} (0x{:x})", expected_sig.v, expected_sig.v);
    println!("r (encoded): 0x{:x}", r);
    println!("r (expected): 0x{:x}", expected_sig.r);
    println!("s (encoded): 0x{:x}", s);
    println!("s (expected): 0x{:x}", expected_sig.s);

    let tx = TransactionRequest::new()
        .nonce(nonce)
        .gas_price(gas_price)
        .gas(gas_limit)
        .to(Address::from_slice(&to))
        .value(value)
        .data(Bytes::from(data))
        .chain_id(chain_id);

    let tx_hash = tx.sighash();
    let sig = Signature {
        r,
        s,
        v: v.as_u64(),
    };
    let recovered_address = sig.recover(tx_hash)?;

    println!("\nVerification Results:");
    println!("Sender Address: {}", recovered_address);

    let chain_v = v.as_u64();
    let recovery_id = (chain_v - (chain_id * 2 + 35)) as u8;
    println!("\nEIP-155 Verification:");
    println!("Chain ID from v: {}", chain_id);
    println!("Recovery ID: {}", recovery_id);

    let v_matches = v == U256::from(expected_sig.v);
    let r_matches = r == expected_sig.r;
    let s_matches = s == expected_sig.s;
    let valid_recovery_id = recovery_id <= 1;

    println!("\nVerification Results:");
    println!("v matches: {}", v_matches);
    println!("r matches: {}", r_matches);
    println!("s matches: {}", s_matches);
    println!("valid recovery_id: {}", valid_recovery_id);

    Ok(v_matches && r_matches && s_matches && valid_recovery_id)
}
