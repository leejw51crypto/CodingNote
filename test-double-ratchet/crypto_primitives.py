"""
⚠️  EDUCATIONAL IMPLEMENTATION ONLY ⚠️

This code is for learning and demonstration purposes.
DO NOT USE IN PRODUCTION - it lacks security features required for real-world applications.

Cryptographic primitives for Signal protocol implementation.

This module provides the basic cryptographic operations needed for the Signal protocol:
- Elliptic Curve Diffie-Hellman (ECDH) key exchange
- HMAC-based Key Derivation Function (HKDF)
- AES-256-GCM encryption/decryption
- Key generation and management utilities

For production use, consider established libraries like libsignal-protocol.
"""

import hashlib
import hmac
import os
from typing import Any, Dict, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class CryptoError(Exception):
    """Custom exception for cryptographic operations."""

    pass


class ECDHKeyPair:
    """Represents an ECDH key pair for the Signal protocol."""

    def __init__(self, private_key: Optional[ec.EllipticCurvePrivateKey] = None):
        if private_key is None:
            self.private_key = ec.generate_private_key(
                ec.SECP256R1(), default_backend()
            )
        else:
            self.private_key = private_key
        self.public_key = self.private_key.public_key()

    def get_public_key_bytes(self) -> bytes:
        """Get the public key as bytes."""
        return self.public_key.public_numbers().x.to_bytes(
            32, "big"
        ) + self.public_key.public_numbers().y.to_bytes(32, "big")

    def get_private_key_bytes(self) -> bytes:
        """Get the private key as bytes."""
        return self.private_key.private_numbers().private_value.to_bytes(32, "big")

    @classmethod
    def from_public_key_bytes(cls, public_key_bytes: bytes) -> "ECDHKeyPair":
        """Create a key pair from public key bytes (for remote keys)."""
        if len(public_key_bytes) != 64:
            raise CryptoError("Invalid public key length")

        x = int.from_bytes(public_key_bytes[:32], "big")
        y = int.from_bytes(public_key_bytes[32:], "big")

        public_numbers = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1())
        public_key = public_numbers.public_key(default_backend())

        # Create a dummy instance with no private key
        instance = cls.__new__(cls)
        instance.private_key = None
        instance.public_key = public_key
        return instance

    def perform_ecdh(self, other_public_key: "ECDHKeyPair") -> bytes:
        """Perform ECDH key exchange."""
        if self.private_key is None:
            raise CryptoError("Cannot perform ECDH without private key")

        shared_key = self.private_key.exchange(ec.ECDH(), other_public_key.public_key)
        return shared_key


class HKDFHelper:
    """Helper class for HKDF operations."""

    @staticmethod
    def derive_key(
        input_key_material: bytes,
        length: int,
        salt: Optional[bytes] = None,
        info: bytes = b"",
    ) -> bytes:
        """Derive a key using HKDF."""
        if salt is None:
            salt = b"\x00" * 32  # Default salt

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            info=info,
            backend=default_backend(),
        )
        return hkdf.derive(input_key_material)

    @staticmethod
    def derive_keys(
        input_key_material: bytes, salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """Derive root key and chain key from input key material."""
        if salt is None:
            salt = b"\x00" * 32

        # Derive 64 bytes: 32 for root key, 32 for chain key
        derived = HKDFHelper.derive_key(
            input_key_material, 64, salt, b"Signal_RootKey_ChainKey"
        )
        return derived[:32], derived[32:]


class AESGCMCipher:
    """AES-GCM encryption/decryption operations."""

    @staticmethod
    def encrypt(
        plaintext: bytes, key: bytes, additional_data: bytes = b""
    ) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt plaintext using AES-256-GCM.

        Returns:
            Tuple of (ciphertext, nonce, auth_tag)
        """
        if len(key) != 32:
            raise CryptoError("Key must be 32 bytes for AES-256")

        nonce = os.urandom(12)  # 96-bit nonce for GCM
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(nonce), backend=default_backend()
        )
        encryptor = cipher.encryptor()

        if additional_data:
            encryptor.authenticate_additional_data(additional_data)

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        auth_tag = encryptor.tag

        return ciphertext, nonce, auth_tag

    @staticmethod
    def decrypt(
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        auth_tag: bytes,
        additional_data: bytes = b"",
    ) -> bytes:
        """
        Decrypt ciphertext using AES-256-GCM.

        Returns:
            Decrypted plaintext
        """
        if len(key) != 32:
            raise CryptoError("Key must be 32 bytes for AES-256")

        cipher = Cipher(
            algorithms.AES(key), modes.GCM(nonce, auth_tag), backend=default_backend()
        )
        decryptor = cipher.decryptor()

        if additional_data:
            decryptor.authenticate_additional_data(additional_data)

        try:
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext
        except Exception as e:
            raise CryptoError(f"Decryption failed: {e}")


class SignalCrypto:
    """
    High-level cryptographic operations for Signal protocol.

    DOUBLE RATCHET OVERVIEW:
    =======================
    This class implements the core cryptographic functions for the Double Ratchet algorithm,
    which provides the security guarantees of the Signal protocol.

    THE DOUBLE RATCHET HAS TWO RATCHETS:
    ===================================

    1. 🐌 DH RATCHET (kdf_rk function):
       - Slow rotation when DH keys change
       - Updates: Root Key + Chain Keys
       - Frequency: Periodically (e.g., every few messages)
       - Purpose: Long-term forward secrecy

    2. ⚡ SYMMETRIC RATCHET (kdf_ck function):
       - Fast rotation for every message
       - Updates: Chain Key + derives Message Key
       - Frequency: Every single message
       - Purpose: Immediate forward secrecy

    KEY FLOW EXAMPLE:
    ================
    X3DH Shared Secret → Root Key → DH Ratchet → Chain Key → Symmetric Ratchet → Message Keys
                                       ↓                        ↓
                                   (periodic)              (every message)

    SECURITY GUARANTEES:
    ===================
    ✅ Forward Secrecy: Old keys cannot decrypt new messages
    ✅ Future Secrecy: New keys cannot decrypt old messages
    ✅ Self-Healing: Compromised state can recover
    ✅ Out-of-order: Messages can arrive in any order
    """

    @staticmethod
    def generate_keypair() -> ECDHKeyPair:
        """Generate a new ECDH key pair."""
        return ECDHKeyPair()

    @staticmethod
    def compute_shared_secret(
        private_key: ECDHKeyPair, public_key: ECDHKeyPair
    ) -> bytes:
        """Compute shared secret between two key pairs."""
        return private_key.perform_ecdh(public_key)

    @staticmethod
    def kdf_rk(root_key: bytes, dh_output: bytes) -> Tuple[bytes, bytes]:
        """
        Key derivation function for root key and chain key - THE DH RATCHET

        DH RATCHET EXPLANATION:
        ======================
        This function implements the DH (Diffie-Hellman) ratchet, which is the "slow"
        ratchet that rotates periodically when new DH key pairs are exchanged.

        WHEN DH RATCHET STEPS OCCUR:
        ============================
        - When Alice/Bob receives a message with a NEW DH public key
        - This indicates the sender generated a new DH key pair
        - Both parties must "step" their DH ratchet forward

        WHAT HAPPENS IN A DH RATCHET STEP:
        =================================
        1. 🤝 Perform DH exchange with new keys
        2. 🔑 Derive new root key and chain key from DH output
        3. 🔄 Replace old root key with new root key
        4. 📤📥 Initialize new sending/receiving chain keys
        5. 🔢 Reset message counters to 0

        KEY HIERARCHY:
        =============
        Root Key (from X3DH)
        └── DH Ratchet Step → New Root Key + Chain Key
            └── Symmetric Ratchet → Message Keys (one per message)

        TWO-LEVEL FORWARD SECRECY:
        ==========================
        - DH Ratchet: Old DH keys can't decrypt new chains
        - Symmetric Ratchet: Old message keys can't decrypt new messages
        - Combined: Provides both long-term and immediate forward secrecy

        This function uses HKDF to derive:
        - New root key (32 bytes) - for next DH ratchet step
        - New chain key (32 bytes) - to start symmetric ratchet chain
        """
        return HKDFHelper.derive_keys(dh_output, salt=root_key)

    @staticmethod
    def kdf_ck(chain_key: bytes) -> Tuple[bytes, bytes]:
        """
        Key derivation function for chain key - THE HEART OF SYMMETRIC KEY RATCHET

        DOUBLE RATCHET EXPLANATION:
        ==========================
        The Double Ratchet algorithm gets its name from having TWO ratchets (one-way advancement):

        1. 🔄 DH RATCHET (Slow rotation):
           - Rotates Diffie-Hellman key pairs periodically
           - Provides forward secrecy through key rotation
           - Updates root keys and chain keys

        2. 🔄 SYMMETRIC KEY RATCHET (Fast rotation):
           - Rotates message keys for EVERY SINGLE MESSAGE
           - Built on top of the DH ratchet using HMAC chains
           - Provides immediate forward secrecy

        HOW MESSAGE KEY ROTATION WORKS:
        ==============================
        This function (kdf_ck) implements the SYMMETRIC KEY RATCHET:

        For each message you want to send:
        1. INPUT: Current chain key (e.g., CK₀)
        2. OUTPUT:
           - New chain key (CK₁) - for next message
           - Message key (MK₁) - for this message only
        3. REPLACE: old chain key with new chain key
        4. USE: message key to encrypt this message
        5. DELETE: message key after use (forward secrecy!)

        Visual chain progression:
        CK₀ → [KDF] → CK₁ + MK₁ (encrypt msg 1)
        CK₁ → [KDF] → CK₂ + MK₂ (encrypt msg 2)
        CK₂ → [KDF] → CK₃ + MK₃ (encrypt msg 3)

        THE "RATCHET" EFFECT:
        ====================
        - You can only move FORWARD: CK₀ → CK₁ → CK₂ → CK₃...
        - You CANNOT go backward: CK₃ ❌→ CK₂
        - Each message key is DELETED after use
        - Even if someone steals CK₃, they cannot decrypt messages 1 or 2!

        HMAC CONSTRUCTION:
        =================
        - Next chain key: HMAC(current_chain_key, 0x02)
        - Message key: HMAC(current_chain_key, 0x01)
        - Different constants ensure different outputs
        - HMAC provides one-way function (cannot reverse)

        FORWARD SECRECY GUARANTEE:
        =========================
        - Old message keys: DELETED → cannot decrypt new messages ✅
        - New chain keys: Cannot derive old message keys ✅
        - Perfect forward secrecy achieved! 🎉
        """
        # Use HMAC to derive next chain key and message key
        next_chain_key = hmac.new(chain_key, b"\x02", hashlib.sha256).digest()
        message_key = hmac.new(chain_key, b"\x01", hashlib.sha256).digest()
        return next_chain_key, message_key

    @staticmethod
    def derive_message_keys(message_key: bytes) -> Dict[str, bytes]:
        """Derive encryption key, MAC key, and IV from message key."""
        encryption_key = HKDFHelper.derive_key(
            message_key, 32, info=b"MessageKey_EncryptionKey"
        )
        mac_key = HKDFHelper.derive_key(message_key, 32, info=b"MessageKey_MACKey")
        iv = HKDFHelper.derive_key(message_key, 16, info=b"MessageKey_IV")

        return {"encryption_key": encryption_key, "mac_key": mac_key, "iv": iv}

    @staticmethod
    def encrypt_message(
        plaintext: bytes, message_key: bytes, associated_data: bytes = b""
    ) -> bytes:
        """Encrypt a message using derived keys."""
        keys = SignalCrypto.derive_message_keys(message_key)
        ciphertext, nonce, auth_tag = AESGCMCipher.encrypt(
            plaintext, keys["encryption_key"], associated_data
        )

        # Combine nonce + auth_tag + ciphertext
        return nonce + auth_tag + ciphertext

    @staticmethod
    def decrypt_message(
        encrypted_data: bytes, message_key: bytes, associated_data: bytes = b""
    ) -> bytes:
        """Decrypt a message using derived keys."""
        if len(encrypted_data) < 28:  # 12 (nonce) + 16 (auth_tag) minimum
            raise CryptoError("Invalid encrypted message length")

        nonce = encrypted_data[:12]
        auth_tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]

        keys = SignalCrypto.derive_message_keys(message_key)
        return AESGCMCipher.decrypt(
            ciphertext, keys["encryption_key"], nonce, auth_tag, associated_data
        )


def generate_random_bytes(length: int) -> bytes:
    """Generate cryptographically secure random bytes."""
    return os.urandom(length)


def secure_compare(a: bytes, b: bytes) -> bool:
    """Constant-time comparison of byte strings."""
    return hmac.compare_digest(a, b)


# Test functions for verification
if __name__ == "__main__":
    # Test ECDH key exchange
    print("Testing ECDH key exchange...")
    alice_keypair = SignalCrypto.generate_keypair()
    bob_keypair = SignalCrypto.generate_keypair()

    alice_shared = SignalCrypto.compute_shared_secret(alice_keypair, bob_keypair)
    bob_shared = SignalCrypto.compute_shared_secret(bob_keypair, alice_keypair)

    assert alice_shared == bob_shared
    print("✓ ECDH key exchange working")

    # Test AES-GCM encryption
    print("Testing AES-GCM encryption...")
    key = generate_random_bytes(32)
    plaintext = b"Hello, Signal Protocol!"

    encrypted = SignalCrypto.encrypt_message(plaintext, key)
    decrypted = SignalCrypto.decrypt_message(encrypted, key)

    assert decrypted == plaintext
    print("✓ AES-GCM encryption working")

    print("All cryptographic primitives working correctly!")
