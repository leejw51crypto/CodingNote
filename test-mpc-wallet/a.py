"""
Educational implementation of Shamir's Secret Sharing and Threshold Signatures.
This code is for learning purposes only and should not be used in production
without proper security review and RFC 6979 compliant nonce generation.
"""

import hashlib
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
import secrets
import base64
import json
import os
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.exceptions import InvalidSignature


class SecretSharingError(Exception):
    """Base exception for secret sharing errors"""

    pass


class ThresholdSignatureError(Exception):
    """Base exception for threshold signature errors"""

    pass


# Shamir's Secret Sharing Implementation
class ShamirSecretSharing:
    """Implementation of Shamir's Secret Sharing scheme"""

    def __init__(self, prime: int):
        """
        Initialize with a prime number that defines the finite field.

        Args:
            prime (int): Prime number for finite field arithmetic
        """
        if not self._is_prime(prime):
            raise ValueError("Input must be a prime number")
        self.prime = prime

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if a number is prime using Miller-Rabin test"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False

        # Miller-Rabin primality test
        def check(a, d, n, s):
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                return True
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    return True
            return False

        s = 0
        d = n - 1
        while d % 2 == 0:
            s += 1
            d //= 2

        # Test with first few prime numbers as witnesses
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == a:
                return True
            if not check(a, d, n, s):
                return False
        return True

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
            raise SecretSharingError("Modular inverse does not exist")
        return (x % m + m) % m

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method"""
        result = coefficients[-1]
        for coef in reversed(coefficients[:-1]):
            result = (result * x + coef) % self.prime
        return result

    def split_secret(
        self, secret: int, threshold: int, num_shares: int
    ) -> List[Tuple[int, int]]:
        """
        Split a secret into shares using Shamir's Secret Sharing.

        Args:
            secret (int): Secret to split
            threshold (int): Minimum number of shares needed to reconstruct
            num_shares (int): Total number of shares to generate

        Returns:
            List of (index, share) tuples
        """
        if not 0 <= secret < self.prime:
            raise SecretSharingError("Secret must be in the range [0, prime)")
        if threshold > num_shares:
            raise SecretSharingError(
                "Threshold cannot be greater than number of shares"
            )
        if threshold < 2:
            raise SecretSharingError("Threshold must be at least 2")

        # Generate random coefficients for polynomial using secure random
        coefficients = [secret] + [
            secrets.randbelow(self.prime) for _ in range(threshold - 1)
        ]

        # Generate shares
        shares = []
        for i in range(1, num_shares + 1):
            shares.append((i, self._evaluate_polynomial(coefficients, i)))

        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, int]], threshold: int) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation.

        Args:
            shares (List[Tuple[int, int]]): List of (index, share) tuples
            threshold (int): Number of shares needed for reconstruction

        Returns:
            Reconstructed secret
        """
        if len(shares) < threshold:
            raise SecretSharingError("Not enough shares for reconstruction")

        secret = 0

        try:
            for i, (x_i, y_i) in enumerate(shares[:threshold]):
                numerator = denominator = 1

                for j, (x_j, _) in enumerate(shares[:threshold]):
                    if i != j:
                        numerator = (numerator * (-x_j)) % self.prime
                        denominator = (denominator * (x_i - x_j)) % self.prime

                factor = (
                    numerator * self._mod_inverse(denominator, self.prime)
                ) % self.prime
                secret = (secret + (y_i * factor)) % self.prime
        except Exception as e:
            raise SecretSharingError(f"Error during secret reconstruction: {str(e)}")

        return secret


@dataclass
class KeyShare:
    """Represents a share of a private key"""

    index: int
    value: int

    def serialize(self) -> str:
        """Convert share to base64 string"""
        data = {"index": self.index, "value": self.value}
        return base64.b64encode(json.dumps(data).encode()).decode()

    @classmethod
    def deserialize(cls, data: str) -> "KeyShare":
        """Create share from base64 string"""
        try:
            json_data = json.loads(base64.b64decode(data).decode())
            return cls(json_data["index"], json_data["value"])
        except Exception as e:
            raise ThresholdSignatureError(f"Invalid share data: {str(e)}")


class ThresholdSignatureScheme:
    """Implementation of a threshold signature scheme using ECDSA"""

    def __init__(self, curve=ec.SECP256K1()):
        """
        Initialize with an elliptic curve

        Args:
            curve: Elliptic curve to use (default: SECP256K1)
        """
        self.curve = curve
        # SECP256K1 curve order - use this for all operations
        self.curve_order = (
            0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        )
        # Initialize SSS with curve order instead of prime field
        self.sss = ShamirSecretSharing(self.curve_order)

    def generate_key_shares(
        self, threshold: int, num_shares: int
    ) -> Tuple[List[KeyShare], ec.EllipticCurvePublicKey]:
        """
        Generate private key shares and corresponding public key

        Args:
            threshold (int): Minimum shares needed for signing
            num_shares (int): Total number of shares to generate

        Returns:
            Tuple of (key shares, public key)
        """
        if threshold < 2:
            raise ThresholdSignatureError("Threshold must be at least 2")
        if num_shares < threshold:
            raise ThresholdSignatureError("Number of shares must be at least threshold")

        # Generate private key in range [1, curve_order-1]
        private_key = 1 + secrets.randbelow(self.curve_order - 1)

        # Split private key into shares using SSS
        try:
            shares = self.sss.split_secret(private_key, threshold, num_shares)
            key_shares = [KeyShare(x, y) for x, y in shares]

            # Generate public key
            private_key_obj = ec.derive_private_key(private_key, self.curve)
            public_key = private_key_obj.public_key()

            return key_shares, public_key
        except Exception as e:
            raise ThresholdSignatureError(f"Error generating key shares: {str(e)}")

    def _deterministic_k(self, message: bytes, private_key: int) -> int:
        """Generate deterministic nonce k according to RFC 6979"""
        # Simple deterministic nonce for demonstration
        # In production, use proper RFC 6979 implementation
        h = hashlib.sha256()
        h.update(str(private_key).encode())
        h.update(message)
        return 1 + (
            int.from_bytes(h.digest(), byteorder="big") % (self.curve_order - 1)
        )

    def sign_message(
        self,
        message: bytes,
        shares: List[KeyShare],
        threshold: int,
        use_deterministic_k: bool = False,
    ) -> Tuple[int, int]:
        """
        Sign a message using threshold number of shares

        Args:
            message (bytes): Message to sign
            shares (List[KeyShare]): List of key shares
            threshold (int): Number of shares needed for signing
            use_deterministic_k (bool): Whether to use deterministic nonce

        Returns:
            Tuple of (r, s) signature components
        """
        if len(shares) < threshold:
            raise ThresholdSignatureError("Not enough shares for signing")

        try:
            # Reconstruct private key from shares
            share_tuples = [(share.index, share.value) for share in shares[:threshold]]
            private_key = self.sss.reconstruct_secret(share_tuples, threshold)

            if use_deterministic_k:
                # Use our deterministic k implementation
                k = self._deterministic_k(message, private_key)
                # Generate point R = k*G
                k_obj = ec.derive_private_key(k, self.curve)
                R = k_obj.public_key().public_numbers()
                r = R.x % self.curve_order

                # Calculate s = k^(-1)(h + r*d) mod n
                h = (
                    int.from_bytes(hashlib.sha256(message).digest(), byteorder="big")
                    % self.curve_order
                )
                k_inv = pow(k, -1, self.curve_order)
                s = (k_inv * (h + r * private_key)) % self.curve_order
            else:
                # Use library's implementation
                private_key_obj = ec.derive_private_key(private_key, self.curve)
                signature = private_key_obj.sign(
                    message,
                    ec.ECDSA(hashes.SHA256()),
                )
                r, s = decode_dss_signature(signature)

            # Normalize s to the lower value (to ensure consistency)
            # In ECDSA, both s and -s are valid, so we always choose the lower value
            if s > self.curve_order // 2:
                s = self.curve_order - s

            return r, s

        except Exception as e:
            raise ThresholdSignatureError(f"Error during signing: {str(e)}")

    def verify_signature(
        self,
        message: bytes,
        signature: Tuple[int, int],
        public_key: ec.EllipticCurvePublicKey,
    ) -> bool:
        """
        Verify a signature

        Args:
            message (bytes): Original message
            signature (Tuple[int, int]): Signature as (r, s)
            public_key: Public key for verification

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            r, s = signature
            encoded_signature = encode_dss_signature(r, s)
            public_key.verify(encoded_signature, message, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            raise ThresholdSignatureError(f"Error during verification: {str(e)}")


def demonstrate_schemes():
    """Demonstrates both SSS and TSS with detailed explanation of each step"""
    print("\n=== 1. Shamir's Secret Sharing (SSS) Demonstration ===")
    # Setup SSS
    prime = 2**256 - 2**32 - 977  # SECP256K1 prime field
    sss = ShamirSecretSharing(prime)

    # Get secret from environment variable
    secret = int(os.environ.get("TEST_SECRET_NUMBER", "12345"))
    threshold = 2
    num_shares = 3

    print(f"Original Secret: {secret}")
    print(f"Threshold: {threshold} (minimum shares needed)")
    print(f"Total Shares: {num_shares}")

    # Split secret
    shares = sss.split_secret(secret, threshold, num_shares)
    print("\nGenerated Shares:")
    for i, (x, y) in enumerate(shares, 1):
        print(f"Share {i}: (x={x}, y={hex(y)})")

    # Reconstruct with different combinations
    print("\nReconstruction:")
    combinations = [(0, 1), (1, 2), (0, 2)]
    for i, j in combinations:
        subset = [shares[i], shares[j]]
        reconstructed = sss.reconstruct_secret(subset, threshold)
        print(
            f"Using shares {i+1} and {j+1}: {reconstructed} (correct: {reconstructed == secret})"
        )

    print("\n=== 2. Threshold Signature Scheme (TSS) Demonstration ===")
    message = b"Hello, World!"
    tss = ThresholdSignatureScheme()

    print("Step 1: Generate distributed key pair")
    threshold = 2
    num_shares = 3
    print(f"- Threshold: {threshold} (minimum signers needed)")
    print(f"- Parties: {num_shares}")

    # Generate key shares
    key_shares, public_key = tss.generate_key_shares(threshold, num_shares)

    print("\nStep 2: Testing different party combinations")
    print("Using deterministic signatures to show same private key reconstruction")

    # Try different party combinations
    combinations = [(0, 1), (1, 2), (0, 2)]
    signatures = []

    for i, j in combinations:
        signing_shares = [key_shares[i], key_shares[j]]
        print(f"\nParties {i+1} and {j+1} collaborate:")
        r, s = tss.sign_message(
            message, signing_shares, threshold, use_deterministic_k=True
        )
        signatures.append((r, s))
        print(f"  r: {hex(r)[:32]}...")
        print(f"  s: {hex(s)[:32]}...")

        print("Verification:", end=" ")
        is_valid = tss.verify_signature(message, (r, s), public_key)
        print(f"{is_valid}")

    # Show that all signatures are the same
    print("\nComparing signatures:")
    for i, ((r1, s1), (r2, s2)) in enumerate(zip(signatures, signatures[1:]), 1):
        print(f"Signature {i} == Signature {i+1}: {r1 == r2 and s1 == s2}")

    print("\nKey points:")
    print("1. All party combinations reconstruct the same private key")
    print("2. Using deterministic nonce (k) shows signatures are identical")
    print("3. This proves the private key reconstruction is consistent")


if __name__ == "__main__":
    demonstrate_schemes()
