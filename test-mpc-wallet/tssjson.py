import json
from tsswallet import setup_tss_key, EthereumTSS
import hashlib
from web3 import Web3
from typing import Dict, Any
import os
from dotenv import load_dotenv


def generate_tss_test_data() -> Dict[str, Any]:
    # Load environment variables
    load_dotenv()

    # Get private key from environment
    private_key = os.getenv("MY_FULL_PRIVATEKEY")
    if not private_key:
        raise ValueError("MY_FULL_PRIVATEKEY not set in environment")

    # Remove '0x' prefix if present
    if private_key.startswith("0x"):
        private_key = private_key[2:]

    print("\nGenerating TSS test data...")
    # Only show first 4 and last 4 characters of private key
    print(f"Using private key from environment: 0x{private_key[:4]}...{private_key[-4:]}")

    # Setup TSS with 2/3 threshold
    key_data = setup_tss_key(private_key, threshold=2, num_parties=3)

    # Initialize TSS for signing
    eth_tss = EthereumTSS()

    # Create test message
    message = "hello world"
    message_bytes = message.encode("utf-8")
    message_hash = Web3.keccak(message_bytes)

    # Generate common seed for deterministic k values
    common_seed = hashlib.sha256(message_hash + b"test_seed").digest()

    print("\nGenerating partial signatures...")
    # Generate partial signatures
    partial_signatures = []
    partial_sigs_data = []

    for i in range(key_data.threshold):
        partial_sig = eth_tss.create_partial_signature(
            key_data.parties[i], message_hash, common_seed
        )
        partial_signatures.append(partial_sig)

        # Store partial signature data
        r, s_bytes, k, R_bytes = partial_sig
        partial_sigs_data.append(
            {
                "party_id": key_data.parties[i].id,
                "r": hex(r),
                "s": "0x" + s_bytes.hex(),
                "k": hex(k),
                "R": "0x" + R_bytes.hex(),
            }
        )
        print(f"Generated partial signature for party {i+1}")

    print("\nCombining signatures...")
    # Combine signatures
    possible_signed_txns = eth_tss.combine_signatures(
        partial_signatures, key_data.parties, message_hash
    )

    # Get the valid signature (first one that matches TSS address)
    valid_sig = None
    for signed_txn in possible_signed_txns:
        valid_sig = {"r": hex(signed_txn.r), "s": hex(signed_txn.s), "v": signed_txn.v}
        break

    # Create test data dictionary
    test_data = {
        "setup": {
            "private_key": f"0x{private_key[:4]}...{private_key[-4:]}",
            "threshold": key_data.threshold,
            "num_parties": key_data.num_parties,
            "tss_address": key_data.tss_address,
            "group_public_key": "0x" + key_data.group_public_key.hex(),
        },
        "parties": [
            {
                "id": party.id,
                "private_share": f"{hex(party.xi)[:6]}...{hex(party.xi)[-4:]}",
                "public_key": "0x" + party.public_key.hex(),
            }
            for party in key_data.parties
        ],
        "test_signing": {
            "message": message,
            "message_hash": "0x" + message_hash.hex(),
            "common_seed": "0x" + common_seed.hex(),
            "partial_signatures": partial_sigs_data,
            "combined_signature": valid_sig,
        },
    }

    return test_data


def main():
    # Generate test data
    test_data = generate_tss_test_data()

    # Save to file with pretty printing
    output_file = "tss_test_data.json"
    with open(output_file, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"\nTSS test data has been written to {output_file}")
    print(f"TSS wallet address: {test_data['setup']['tss_address']}")


if __name__ == "__main__":
    main()
