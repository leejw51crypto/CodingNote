"""
Threshold Signature Scheme (TSS) using Lagrange Interpolation

Mathematical Background:
------------------------
1. Key Generation (t,n):
   - Private key d is split into n shares using polynomial f(x)
   - f(x) = d + a₁x + a₂x² + ... + aₜ₋₁xᵗ⁻¹  mod q
   - Each party i gets share dᵢ = f(i)
   - Any t parties can reconstruct d, t-1 parties cannot

2. Signing Process:
   Step 1: Each party i computes partial signature
   - k = H(message)  [common nonce]
   - σᵢ = k·dᵢ·H(m) mod q  [partial signature]
   where:
   - dᵢ is party i's share
   - H(m) is message hash
   - k is common nonce
   - q is curve order

3. Signature Reconstruction:
   Step 1: Calculate Lagrange coefficients
   - λᵢ = ∏ⱼ≠ᵢ (0-j)/(i-j) mod q
   
   Step 2: Combine partial signatures
   - σ = ∑ᵢ (σᵢ·λᵢ) mod q
   
   Final signature = σ = k·d·H(m) mod q
   where:
   - d is original private key
   - k is common nonce
   - H(m) is message hash
"""

from typing import Dict, List, Tuple
import hashlib
import random
import hmac
from coincurve import PrivateKey, PublicKey
import os
from dotenv import load_dotenv

# Secp256k1 curve order q
CURVE_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def mod_inverse(a: int, m: int) -> int:
    """Calculate modular multiplicative inverse
    a⁻¹ mod m where a·a⁻¹ ≡ 1 (mod m)
    """

    def extended_gcd(a: int, b: int):
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


def lagrange_coefficient(parties: list, j: int, x: int = 0) -> int:
    """Calculate Lagrange coefficient λⱼ for party j

    λⱼ = ∏ⱼ≠ᵢ (x-m)/(j-m) mod q

    For x=0:
    λⱼ = ∏ⱼ≠ᵢ (-m)/(j-m) mod q
         = ∏ⱼ≠ᵢ m/(m-j) mod q
    """
    num, den = 1, 1
    for m in parties:
        if m != j:
            num = (num * (x - m)) % CURVE_ORDER  # numerator: (x-m)
            den = (den * (j - m)) % CURVE_ORDER  # denominator: (j-m)
    return (num * mod_inverse(den, CURVE_ORDER)) % CURVE_ORDER


def generate_shares(secret: int, threshold: int, n_parties: int) -> Dict[int, int]:
    """Generate n shares of the secret using (threshold-1) degree polynomial

    f(x) = d + a₁x + a₂x² + ... + aₜ₋₁xᵗ⁻¹ mod q
    where:
    - d is the secret
    - aᵢ are random coefficients
    - t is threshold
    - q is curve order

    Each share dᵢ = f(i) for i = 1...n
    """
    # Generate random coefficients [a₁, a₂, ..., aₜ₋₁]
    coefficients = [secret] + [
        random.randrange(CURVE_ORDER) for _ in range(threshold - 1)
    ]

    # Generate shares dᵢ = f(i) for each party
    shares = {}
    for i in range(1, n_parties + 1):
        # Evaluate f(i) = d + a₁i + a₂i² + ... + aₜ₋₁iᵗ⁻¹
        value = 0
        for power, coeff in enumerate(coefficients):
            value = (value + coeff * pow(i, power, CURVE_ORDER)) % CURVE_ORDER
        shares[i] = value

    return shares


def generate_nonce(message: str) -> int:
    """Generate deterministic nonce k using message hash
    k = H(message) mod q
    All parties must use same k for signature to work
    """
    h = hashlib.sha256(message.encode()).digest()
    return int.from_bytes(h, "big") % CURVE_ORDER


def create_partial_signature(message: str, share: int) -> Tuple[int, int]:
    """Create partial signature using share

    Standard ECDSA partial signature:
    - Calculate R = k·G and r = R_x
    - σᵢ = (H(m) + r·dᵢ)/k mod q
    where:
    - k is common nonce
    - dᵢ is party's share
    - H(m) is message hash
    - q is curve order

    Returns (r, σᵢ)
    """
    # k = H(message) mod q
    k = generate_nonce(message)
    k_inv = mod_inverse(k, CURVE_ORDER)

    # Calculate R = k·G and get x coordinate as r
    R = PrivateKey.from_int(k).public_key
    r = int.from_bytes(R.format(compressed=True)[1:33], "big")

    # H(m) = SHA256(m) mod q
    msg_hash = int(hashlib.sha256(message.encode()).hexdigest(), 16) % CURVE_ORDER

    # σᵢ = (H(m) + r·dᵢ)/k mod q
    partial_sig = (k_inv * (msg_hash + r * share)) % CURVE_ORDER

    return r, partial_sig


def combine_signatures(
    partial_sigs: Dict[int, Tuple[int, int]], participating_parties: List[int]
) -> int:
    """Combine partial signatures using Lagrange interpolation

    σ = ∑ᵢ (σᵢ·λᵢ) mod q
    where:
    - σᵢ are partial signatures
    - λᵢ are Lagrange coefficients
    - q is curve order

    Final signature σ = k·d·H(m) mod q
    """
    final_sig = 0

    # All parties must use same k
    k = partial_sigs[participating_parties[0]][0]

    for party_id in participating_parties:
        # Verify party used same k
        if partial_sigs[party_id][0] != k:
            raise ValueError(f"Party {party_id} used different nonce k")

        # Get λᵢ for this party
        coeff = lagrange_coefficient(participating_parties, party_id)

        # Add σᵢ·λᵢ to sum
        partial = (partial_sigs[party_id][1] * coeff) % CURVE_ORDER
        final_sig = (final_sig + partial) % CURVE_ORDER

    return final_sig


def verify_signature(message: str, r: int, s: int, secret: int):
    """Verify standard ECDSA signature

    The relationship between R and public key P:
    1. R = k·G  (where k is nonce)
    2. P = d·G  (where d is private key)

    They are related through the signature equation:
    s = (H(m) + r·d)/k

    This means:
    - R is a random point (k·G) whose x-coordinate is r
    - P is the public key point (d·G)
    - The signature s binds them together through k and d
    - During verification, we can reconstruct R without knowing k or d:
      R = s⁻¹(H(m)·G + r·P)
    """
    # Get public key point P = d·G
    privkey = PrivateKey.from_int(secret)
    pubkey = privkey.public_key

    # Get the original R point from k (we normally wouldn't have k, this is just for demonstration)
    k = generate_nonce(message)
    R_original = PrivateKey.from_int(k).public_key

    # Calculate message hash
    msg_hash = int(hashlib.sha256(message.encode()).hexdigest(), 16) % CURVE_ORDER

    # Calculate s⁻¹
    s_inv = mod_inverse(s, CURVE_ORDER)

    # Calculate u1 = H(m)·s⁻¹ mod n
    u1 = (msg_hash * s_inv) % CURVE_ORDER

    # Calculate u2 = r·s⁻¹ mod n
    u2 = (r * s_inv) % CURVE_ORDER

    # Calculate u1·G
    point1 = PrivateKey.from_int(u1).public_key

    # Calculate u2·P
    point2 = pubkey.multiply(bytes.fromhex(format(u2, "064x")))

    # Calculate R' = u1·G + u2·P = s⁻¹(H(m)·G + r·P)
    R_calc = point1.combine([point2])

    # Get x-coordinate of R'
    r_calc = int.from_bytes(R_calc.format(compressed=True)[1:33], "big")

    print("\nSignature Verification:")
    print(f"Public Key P = d·G: {pubkey.format(compressed=False).hex()}")
    print(f"Original R = k·G: {R_original.format(compressed=False).hex()}")
    print(f"Reconstructed R: {R_calc.format(compressed=False).hex()}")
    print(f"Message hash H(m): {msg_hash}")
    print(f"Signature (r,s): ({r}, {s})")
    print("\nRelationship between R and public key P:")
    print(f"1. R = k·G is a random point generated using nonce k")
    print(f"2. P = d·G is the public key point derived from private key d")
    print(f"3. r is just the x-coordinate of point R")
    print(f"4. The signature equation s = (H(m) + r·d)/k binds R and P together")
    print(f"5. During verification, we can reconstruct R = s⁻¹(H(m)·G + r·P)")
    print(f"   This works because both R and P are points on the same curve")
    print(f"   and are related through the private key d and nonce k")
    print(f"\nVerification succeeds because:")
    print(f"- Original r (x-coord of k·G): {r}")
    print(f"- Calculated r' (x-coord of reconstructed R): {r_calc}")
    print(f"- They match: {r == r_calc}")


def main():
    """Demonstrate 2-out-of-3 Threshold Signature Scheme
    - Total parties (n) = 3
    - Threshold (t) = 2
    - Any 2 parties can create valid signature
    """
    # Load environment variables
    load_dotenv()

    # Setup parameters
    secret = int(os.getenv("TEST_SECRET_NUMBER"))  # Private key d from environment
    threshold = 2  # t: minimum parties needed
    n_parties = 3  # n: total number of parties

    # Generate shares dᵢ = f(i)
    shares = generate_shares(secret, threshold, n_parties)
    print("Original private key (d):", secret)
    print("\nShares distributed to parties (dᵢ):")
    for party_id, share in shares.items():
        print(f"Party {party_id} (d_{party_id}): {share}")

    # Simulate signing
    message = "Hello, TSS!"
    print(f"\nMessage to sign (m): {message}")

    # Each party creates partial signature σᵢ
    partial_sigs = {}
    for party_id, share in shares.items():
        r, sig = create_partial_signature(message, share)
        partial_sigs[party_id] = (r, sig)
        print(f"Partial signature from Party {party_id} (σ_{party_id}): {sig}")

    # Reconstruct signature with different party combinations
    for participating_parties in [[1, 2], [2, 3], [1, 3]]:
        print(f"\nReconstructing signature with parties {participating_parties}")
        final_sig = combine_signatures(partial_sigs, participating_parties)

        # Get r value from first party (all use same k so same r)
        r = partial_sigs[participating_parties[0]][0]
        print(f"r: {r}")
        print(f"s: {final_sig}")

        # Verify the signature
        verify_signature(message, r, final_sig, secret)


if __name__ == "__main__":
    main()
