import pytest
from c import ThresholdSignatureScheme, EthereumTSS, Party, TSSKeyData
from web3 import Web3
import hashlib
import secrets
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from eth_account.messages import encode_defunct
from eth_account import Account
from eth_utils import to_checksum_address
import os


def test_combine_partial_signatures():
    print("\n=== Starting Threshold Signature Test ===")

    # Initialize TSS components
    tss = ThresholdSignatureScheme()
    eth_tss = EthereumTSS()
    print("\nInitialized TSS components")

    # Initialize Web3
    w3 = Web3()

    # Test parameters
    threshold = 2
    num_parties = 3
    print(f"\nTest Parameters:")
    print(f"Threshold (t): {threshold}")
    print(f"Number of parties (n): {num_parties}")

    # Get private key from environment variable
    private_key_hex = os.environ.get("MY_FULL_PRIVATEKEY")
    if private_key_hex.startswith("0x"):
        private_key_hex = private_key_hex[2:]  # Remove '0x' prefix if present
    private_key = int(private_key_hex, 16)
    print(f"\nUsing private key from environment: {hex(private_key)}")

    # Generate coefficients for polynomial (degree t-1)
    coefficients = [private_key]
    for i in range(threshold - 1):
        coeff = secrets.randbelow(tss.curve_order - 1) + 1
        coefficients.append(coeff)
        print(f"Generated polynomial coefficient a{i+1}: {hex(coeff)}")

    # Generate shares for each party
    print("\nGenerating shares for each party:")
    parties = []
    for i in range(num_parties):
        x = i + 1
        share = coefficients[0]  # First coefficient is the secret
        for j in range(1, threshold):
            exp = pow(x, j, tss.curve_order)
            term = (coefficients[j] * exp) % tss.curve_order
            share = (share + term) % tss.curve_order

        # Generate public key for share
        private_key_obj = ec.derive_private_key(share, tss.curve, default_backend())
        public_key = private_key_obj.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        parties.append(Party(i + 1, share, public_key))
        print(f"Party {i+1}:")
        print(f"  Share: {hex(share)}")
        print(f"  Public key: {public_key.hex()}")

    # Generate group public key
    group_public_key = (
        ec.derive_private_key(private_key, tss.curve, default_backend())
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
    )
    print(f"\nGroup public key: {group_public_key.hex()}")

    # Create TSSKeyData
    # Derive TSS address using the TSS class method
    tss_address = tss.derive_ethereum_address(group_public_key)
    print(f"\nDerived TSS address from public key: {tss_address}")

    key_data = TSSKeyData(
        parties, tss_address, group_public_key, threshold, num_parties
    )
    print(f"TSS Wallet address: {tss_address}")

    # Create test message and hash
    message = "hello world"
    message_bytes = message.encode("utf-8")
    message_hash = w3.keccak(text=message)
    print(f"\nTest message: '{message}'")
    print(f"Message hash: {message_hash.hex()}")

    # Generate common seed
    common_seed = hashlib.sha256(message_hash + b"test_seed").digest()
    print(f"Common seed: {common_seed.hex()}")

    # Generate partial signatures from first two parties (threshold = 2)
    print("\n=== Partial Signatures ===")
    partial_sigs = []
    for i in range(threshold):
        r, s_bytes, k, R_bytes = tss.create_partial_signature(
            parties[i], message_hash, common_seed
        )
        partial_sigs.append((r, s_bytes, k, R_bytes))
        print(f"\nPartial signature from Party {i+1}:")
        print(f"  r: {hex(r)}")
        print(f"  s: {hex(int.from_bytes(s_bytes, 'big'))}")
        print(f"  k: {hex(k)}")

    print("\n=== Combined Signature Variants ===")
    # Combine signatures using TSS implementation
    combined_signatures = tss.combine_partial_signatures(
        partial_sigs, parties, message_hash
    )

    for idx, sig in enumerate(combined_signatures, 1):
        print(f"\nTrying signature variant {idx}:")
        print(f"  r: {hex(sig.r)}")
        print(f"  s: {hex(sig.s)}")
        print(f"  v: {hex(sig.v)}")

        try:
            # Create signature object using eth_account
            vrs = (sig.v, sig.r, sig.s)

            # Recover address using eth_account
            recovered_address = Account._recover_hash(message_hash, vrs=vrs)
            print(f"  Recovered address: {recovered_address}")
            print(f"  Expected address: {tss_address}")

            if recovered_address.lower() == tss_address.lower():
                print("  ✓ Addresses match")
                print("\n✅ Found matching TSS wallet signature! Test passed.")
                return
            else:
                print("  ✗ Addresses don't match")

        except Exception as e:
            print(f"❌ Failed to verify signature variant {idx}: {str(e)}")
            continue

    raise AssertionError(
        "No valid signature found that recovers to the TSS wallet address"
    )

    print("\n=== Test Complete ===")
