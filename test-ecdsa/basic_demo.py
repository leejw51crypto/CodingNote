"""
MIT License

Copyright (c) 2024 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

⚠️ SECURITY WARNING - FOR EDUCATIONAL PURPOSES ONLY ⚠️
====================================================
This implementation is NOT secure and should NEVER be used in production.
Critical security risks include:
1. Non-constant time operations (vulnerable to timing attacks)
2. Lack of proper input validation
3. Basic implementation of cryptographic operations
4. No protection against side-channel attacks
5. Not audited for security vulnerabilities

For production use:
- Use established cryptographic libraries (e.g., eth_keys, cryptography, pycryptodome)
- Never store private keys in code
- Implement proper key management
- Use secure random number generation
- Follow cryptographic best practices

ECDSA (Elliptic Curve Digital Signature Algorithm) Implementation
==============================================================

This implementation uses secp256k1 curve (the same curve used by Bitcoin and Ethereum).

Key Components:
--------------
1. Private Key (d): A random 256-bit integer
2. Public Key (Q): Q = d × G where G is the generator point of secp256k1
3. Message Hash (z): Keccak256 hash of the message
4. Signature: (r, s, v) where:
   - r is x-coordinate of point R = k × G
   - s = k⁻¹(z + rd) mod n
   - v is recovery id (Ethereum specific)

Security Requirements:
--------------------
1. Private key must be kept absolutely secret
2. Each signature must use a unique random k (nonce)
3. The curve order n must be prime and large (secp256k1 uses 256-bit n)

Mathematical Relationships:
------------------------
s = k⁻¹(z + rd) mod n
ks = z + rd mod n
k = (z + rd)s⁻¹ mod n

Verification works because:
P = u₁G + u₂Q = (zs⁻¹)G + (rs⁻¹)(dG) = (z + rd)s⁻¹G = kG
"""

from eth_keys import keys
import os
import binascii
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# secp256k1 curve parameters (these are well-known constants)
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_G_X = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP256K1_G_Y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F


def load_private_key():
    """
    Load private key (d) from environment variable.
    In ECDSA, private key is a random integer d where 1 < d < n-1
    n is the curve order of secp256k1
    """
    private_key_hex = os.environ.get("MY_FULL_PRIVATEKEY")
    if not private_key_hex:
        raise ValueError(
            "Private key not found in environment variables. Please set MY_FULL_PRIVATEKEY in .env file"
        )

    private_key_hex = private_key_hex.replace("0x", "")
    return bytes.fromhex(private_key_hex)


def sign_message(message: str, private_key_bytes: bytes):
    """
    Sign a message using ECDSA.

    Steps:
    1. Create private key object (d)
    2. Derive public key Q = d × G
    3. Hash message to get z
    4. Generate signature (r, s) where:
       - r is x-coordinate of R = k × G
       - s = k⁻¹(z + rd) mod n
       - k is a random nonce generated internally by eth_keys

    Note: eth_keys handles the secure generation of k (nonce) internally
    """
    # Create private key object
    private_key = keys.PrivateKey(private_key_bytes)

    # Get public key (Q = d × G)
    public_key = private_key.public_key

    # Hash the message to get z
    # We use keccak256 (sha3) as that's what Ethereum uses
    message_hash = hashlib.sha3_256(message.encode()).digest()

    # Sign the message hash
    # eth_keys internally:
    # 1. Generates random k
    # 2. Computes R = k × G
    # 3. Computes s = k⁻¹(z + rd) mod n
    signature = private_key.sign_msg_hash(message_hash)

    return {
        "message": message,
        "message_hash": "0x" + binascii.hexlify(message_hash).decode(),
        "signature": {
            "r": hex(signature.r),  # x-coordinate of R
            "s": hex(signature.s),  # s = k⁻¹(z + rd) mod n
            "v": hex(signature.v),  # recovery id (Ethereum specific)
        },
        "public_key": "0x" + binascii.hexlify(public_key.to_bytes()).decode(),
        "public_key_address": public_key.to_address(),
    }


def verify_signature(message: str, signature_dict: dict, public_key_hex: str):
    """
    Verify an ECDSA signature.

    Steps:
    1. Recreate public key Q from hex
    2. Hash message to get z
    3. Verify that signature (r,s) satisfies:
       - 0 < r < n and 0 < s < n
       - P = u₁G + u₂Q has x-coordinate equal to r mod n
       where:
       - u₁ = zw mod n
       - u₂ = rw mod n
       - w = s⁻¹ mod n
    """
    try:
        # Recreate public key object
        public_key_bytes = bytes.fromhex(public_key_hex.replace("0x", ""))
        public_key = keys.PublicKey(public_key_bytes)

        # Hash the message to get z
        message_hash = hashlib.sha3_256(message.encode()).digest()

        # Recreate signature object
        r = int(signature_dict["r"], 16)
        s = int(signature_dict["s"], 16)
        v = int(signature_dict["v"], 16)
        signature = keys.Signature(vrs=(v, r, s))

        # Verify
        return signature.verify_msg_hash(message_hash, public_key)
    except Exception as e:
        print(f"Verification error: {e}")
        return False


def main():
    """
    Demonstrate ECDSA signing and verification with detailed mathematical components
    """
    try:
        # Load private key
        private_key_bytes = load_private_key()

        # Example message
        message = "Hello, ECDSA!"

        print("\n# ECDSA Signing and Verification Demo")

        print("\n## 0. Curve Parameters (secp256k1)")
        print("```")
        print("Generator Point G:")
        print(f"  x: {hex(SECP256K1_G_X)}")
        print(f"  y: {hex(SECP256K1_G_Y)}")
        print("\nCurve Order n:")
        print(f"  n: {hex(SECP256K1_N)}")
        print("\nField Prime p:")
        print(f"  p: {hex(SECP256K1_P)}")
        print("```")

        print("\n## 1. Signing Message")
        result = sign_message(message, private_key_bytes)

        print("\n### Input")
        print("```")
        print(f"Message: {result['message']}")
        print("```")

        print("\n### Key Components")
        print("```")
        print("\nGenerator Point (G):")
        print(f"G.x = {hex(SECP256K1_G_X)}")
        print(f"G.y = {hex(SECP256K1_G_Y)}")

        print("\nPublic Key Calculation (Q = d × G):")
        print(f"Q   = {result['public_key']}")
        print("\nNote: Q (public key) is a point on the curve that results from")
        print("      multiplying the generator point G by the private key d.")
        print(
            "      This is a one-way operation due to the Discrete Logarithm Problem."
        )
        print("```")

        print("\n### Message Hash")
        print("```")
        print(f"z = Keccak256(message) = {result['message_hash']}")
        print("```")

        print("\n### Generated Signature Components")
        print("```")
        print("Signature Values:")
        print(f"r: {result['signature']['r']}")
        print(f"s: {result['signature']['s']}")
        print(f"v: {result['signature']['v']}")
        print("\nMathematical Relationship:")
        print("s = k⁻¹(z + rd) mod n")
        print("where:")
        print("  r is x-coordinate of R = k × G")
        print("  k is a unique random nonce (kept secret)")
        print("```")

        print("\n### Public Key Info")
        print("```")
        print(f"Public Key: {result['public_key']}")
        print(f"Address: {result['public_key_address']}")
        print("```")

        print("\n## 2. Verifying Original Signature")
        is_valid = verify_signature(message, result["signature"], result["public_key"])
        print("```")
        print(f"✓ Signature is valid: {is_valid}")
        print("```")

        print("\n## 3. Verification with Modified Message")
        print("To demonstrate that changing the message invalidates the signature:")
        wrong_message = "Wrong message!"
        print("```")
        print(f"Modified message: {wrong_message}")
        is_valid = verify_signature(
            wrong_message, result["signature"], result["public_key"]
        )
        print(f"✗ Signature is valid: {is_valid}")
        print("```")

        print("\n## Security Notes")
        print(
            """
### Important Security Considerations:
```
1. Private Key Protection:
   • NEVER hardcode private keys in source code
   • Store private keys securely (e.g., in encrypted form)
   • Use environment variables or secure key management systems
   
2. Nonce (k) Requirements:
   • Must be cryptographically secure random
   • Must be unique for each signature
   • Must never be reused
   • If k is reused, private key can be compromised
   
3. Implementation Security:
   • Use well-tested cryptographic libraries
   • Keep dependencies up to date
   • Follow secure coding practices
   • Regular security audits
   
4. Environment Security:
   • Secure the environment variables
   • Use proper access controls
   • Regular system updates and patches
   • Monitor for suspicious activities
```
"""
        )

    except Exception as e:
        print(f"Error in demonstration: {e}")
        print("Please ensure you have set up the environment variables correctly.")
        print("Create a .env file with your private key: MY_FULL_PRIVATEKEY=0x...")


if __name__ == "__main__":
    main()
