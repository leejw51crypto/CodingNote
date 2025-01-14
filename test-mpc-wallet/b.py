# this is experimental code
# education and research purpose only

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
import secrets
import hashlib
from web3 import Web3
from eth_account.datastructures import SignedTransaction
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from eth_account import Account
import json


@dataclass
class Party:
    """Represents a signing party in the TSS protocol"""

    id: int
    xi: int  # Secret share of the private key
    public_key: bytes  # Public key share


class ThresholdSignatureScheme:
    def __init__(self):
        self.curve = ec.SECP256K1()
        self.curve_order = (
            0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        )
        # Generate base point G
        self.g = ec.derive_private_key(1, self.curve, default_backend()).public_key()

    def _mod_inverse(self, a: int, m: int) -> int:
        """Calculate modular multiplicative inverse"""

        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m

    def _lagrange_coefficient(self, parties: List[Party], j: int, x: int = 0) -> int:
        """Calculate Lagrange coefficient for party j"""
        num, den = 1, 1
        for m in parties:
            if m.id != j:
                num = (num * (x - m.id)) % self.curve_order
                den = (den * (j - m.id)) % self.curve_order
        return (num * self._mod_inverse(den, self.curve_order)) % self.curve_order

    def split_existing_key(
        self, private_key: int, threshold: int, num_parties: int
    ) -> Tuple[List[Party], bytes]:
        """
        Split an existing private key into shares using Shamir's Secret Sharing
        Returns list of parties with their shares and group public key
        """
        if threshold < 2 or threshold > num_parties:
            raise ValueError("Invalid threshold value")

        # Generate random coefficients for polynomial
        coefficients = [private_key]  # a0 is the secret (private key)
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(self.curve_order - 1) + 1)

        parties: List[Party] = []

        # Generate shares using polynomial evaluation
        for i in range(num_parties):
            x = i + 1
            # Evaluate polynomial
            share = coefficients[0]  # start with private key
            for j in range(1, threshold):
                exp = pow(x, j, self.curve_order)
                term = (coefficients[j] * exp) % self.curve_order
                share = (share + term) % self.curve_order

            # Generate public key share
            private_key_obj = ec.derive_private_key(
                share, self.curve, default_backend()
            )
            public_key = private_key_obj.public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )

            parties.append(Party(i + 1, share, public_key))

        # Group public key is the original public key
        group_public_key = (
            ec.derive_private_key(private_key, self.curve, default_backend())
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )
        )

        return parties, group_public_key

    def create_partial_signature(
        self, party: Party, message_hash: bytes, shared_randomness: Dict[str, int]
    ) -> Tuple[int, bytes]:
        """
        Create partial signature using standard ECDSA with Lagrange interpolation
        shared_randomness: Dictionary containing k value
        """
        k = shared_randomness.get("k")
        if not k:
            raise ValueError("k value not provided in shared randomness")

        # Generate R = k * G
        k_key = ec.derive_private_key(k, self.curve, default_backend())
        R = k_key.public_key()
        r = R.public_numbers().x % self.curve_order

        # Compute partial signature
        z = int.from_bytes(message_hash, "big")
        k_inv = self._mod_inverse(k, self.curve_order)

        # Each party computes their full share of s = k^(-1)(z + r*x)
        s = (k_inv * (z + (r * party.xi))) % self.curve_order

        return r, s.to_bytes(32, "big")

    def combine_partial_signatures(
        self,
        partial_signatures: List[Tuple[int, bytes]],
        parties: List[Party],
        message_hash: bytes,
    ) -> SignedTransaction:
        """
        Combine partial signatures using Lagrange interpolation
        """
        r = partial_signatures[0][0]  # Common r value

        # Combine s values using Lagrange interpolation
        s_combined = 0
        for i, (_, si) in enumerate(partial_signatures):
            s_i = int.from_bytes(si, "big")
            lambda_i = self._lagrange_coefficient(
                parties[: len(partial_signatures)], parties[i].id
            )
            s_combined = (s_combined + (lambda_i * s_i)) % self.curve_order

        # Create Ethereum-compatible signature
        v = 27  # or 28, depending on which value creates a valid signature

        # Check if we need to flip s and v according to EIP-2
        if s_combined > self.curve_order // 2:
            s_combined = self.curve_order - s_combined
            v = 28

        r_bytes = r.to_bytes(32, "big")
        s_bytes = s_combined.to_bytes(32, "big")

        # Create signature bytes
        signature = v.to_bytes(1, "big") + r_bytes + s_bytes

        return SignedTransaction(
            rawTransaction=signature,
            hash=Web3.keccak(signature),
            r=r,
            s=s_combined,
            v=v,
        )

    def derive_ethereum_address(self, group_public_key: bytes) -> str:
        """Derive Ethereum address from group public key"""
        public_key_bytes = group_public_key[1:]  # Remove '04' prefix
        keccak = Web3.keccak(public_key_bytes)
        address = keccak[-20:].hex()
        return Web3.to_checksum_address(address)


class EthereumTSS:
    def __init__(self):
        self.tss = ThresholdSignatureScheme()
        self.shared_randomness = {}

    def setup_existing_key(
        self, private_key: int, threshold: int, num_parties: int
    ) -> Tuple[List[Party], str]:
        """Set up the TSS protocol with an existing private key"""
        # Split existing key into shares
        parties, group_public_key = self.tss.split_existing_key(
            private_key, threshold, num_parties
        )
        eth_address = self.tss.derive_ethereum_address(group_public_key)

        # Generate shared randomness (in real implementation, this would be done via MPC)
        self.shared_randomness = {"k": secrets.randbelow(self.tss.curve_order - 1) + 1}

        return parties, eth_address

    def create_partial_signature(
        self, party: Party, message_hash: bytes
    ) -> Tuple[int, bytes]:
        """Create partial signature for a party"""
        return self.tss.create_partial_signature(
            party, message_hash, self.shared_randomness
        )

    def combine_signatures(
        self,
        partial_signatures: List[Tuple[int, bytes]],
        parties: List[Party],
        message_hash: bytes,
    ) -> SignedTransaction:
        """Combine partial signatures"""
        return self.tss.combine_partial_signatures(
            partial_signatures, parties, message_hash
        )


def verify_signature(message_hash: bytes, r: int, s: int, v: int, address: str) -> bool:
    """Verify an Ethereum signature using Web3's ecrecover"""
    try:
        w3 = Web3()

        # Create the signature bytes in the correct format
        signature_bytes = bytes.fromhex(
            hex(r)[2:].zfill(64)  # r value as 32 bytes
            + hex(s)[2:].zfill(64)  # s value as 32 bytes
        ) + bytes(
            [v]
        )  # v value as 1 byte

        # Create an EthereumMessage object manually
        recovered_address = w3.eth.account.recover_hash(
            message_hash, signature=signature_bytes
        )

        print(f"Recovered address: {recovered_address}")
        print(f"Expected address: {address}")

        return recovered_address.lower() == address.lower()
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return False


def demonstrate_simple_tss():
    # Initialize
    eth_tss = EthereumTSS()

    # Get private key from environment or use a test key
    private_key = os.getenv(
        "MY_FULL_PRIVATEKEY",
        "",
    )
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    # Convert hex private key to int for TSS
    private_key_int = int(private_key, 16)

    # Set up parameters
    threshold = 2
    num_parties = 3

    # Split the existing private key into shares
    parties, eth_address = eth_tss.setup_existing_key(
        private_key_int, threshold, num_parties
    )

    print(f"\n=== Simple TSS Demonstration ===")
    print(f"Generated TSS Address: {eth_address}")
    print(f"Threshold: {threshold}")
    print(f"Total Parties: {num_parties}")
    print(f"Using k value: {hex(eth_tss.shared_randomness['k'])}")

    # Create message and its Ethereum prefixed hash (EIP-191)
    message = "hello world"
    message_bytes = message.encode()
    prefix = "\x19Ethereum Signed Message:\n" + str(len(message_bytes))
    prefixed_message = prefix.encode() + message_bytes
    prefixed_hash = Web3.keccak(prefixed_message)

    print(f"\nSigning message: '{message}'")
    print(f"Prefixed message hash: {prefixed_hash.hex()}")

    # Direct signing for comparison
    print("\n=== Direct Signature ===")
    account = Account.from_key(private_key)

    # Sign using k value
    k = eth_tss.shared_randomness["k"]
    k_key = ec.derive_private_key(k, eth_tss.tss.curve, default_backend())
    R = k_key.public_key()
    r = R.public_numbers().x % eth_tss.tss.curve_order

    z = int.from_bytes(prefixed_hash, "big")
    k_inv = eth_tss.tss._mod_inverse(k, eth_tss.tss.curve_order)
    s = (k_inv * (z + (r * private_key_int))) % eth_tss.tss.curve_order

    # Check if we need to flip s according to EIP-2
    if s > eth_tss.tss.curve_order // 2:
        s = eth_tss.tss.curve_order - s
        v = 28
    else:
        v = 27

    print(f"Signer address: {account.address}")
    print(f"r: {hex(r)}")
    print(f"s: {hex(s)}")
    print(f"v: {v}")

    # Verify direct signature
    direct_valid = verify_signature(prefixed_hash, r, s, v, account.address)
    print(f"Direct signature valid: {direct_valid}")

    print("\n=== TSS Signature ===")
    # Generate partial signatures
    partial_signatures = []
    for i in range(threshold):
        partial_sig = eth_tss.create_partial_signature(parties[i], prefixed_hash)
        print(f"\nParty {i+1} share: {hex(parties[i].xi)}")
        print(f"Party {i+1} partial s: {hex(int.from_bytes(partial_sig[1], 'big'))}")
        partial_signatures.append(partial_sig)

    # Combine signatures
    try:
        tss_signed = eth_tss.combine_signatures(
            partial_signatures, parties, prefixed_hash
        )
        print(f"\nCombined TSS signature:")
        print(f"r: {hex(tss_signed.r)}")
        print(f"s: {hex(tss_signed.s)}")
        print(f"v: {tss_signed.v}")

        # Verify TSS signature
        tss_valid = verify_signature(
            prefixed_hash, tss_signed.r, tss_signed.s, tss_signed.v, eth_address
        )
        print(f"TSS signature valid: {tss_valid}")

    except Exception as e:
        print(f"Error in TSS signing: {str(e)}")


if __name__ == "__main__":
    demonstrate_simple_tss()
