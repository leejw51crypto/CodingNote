from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.fernet import Fernet
from base64 import urlsafe_b64encode

# ed 25519
# example: A: dapp, B: wallet
#
# public_key_A = private_key_A * G
# public_key_B = private_key_B * G
#
# shared_secret_A = private_key_A * public_key_B
# shared_secret_A = private_key_A * (private_key_B * G)
#
# shared_secret_B = private_key_B * public_key_A
# shared_secret_B = private_key_B * (private_key_A * G)

#
# private_key_A * (private_key_B * G) = private_key_B * (private_key_A * G)
# shared_secret_A = shared_secret_B

class KeyExchange:
    def __init__(self):
        self.dapp_private_key = X25519PrivateKey.generate()
        self.wallet_private_key = X25519PrivateKey.generate()
        self.dapp_public_key = self.dapp_private_key.public_key()
        self.wallet_public_key = self.wallet_private_key.public_key()

    def derive_symmetric_key(self, private_key, public_key):
        shared_secret = private_key.exchange(public_key)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'handshake data',
        )
        return hkdf.derive(shared_secret)

    def serialize_keys(self, key, encoding, format, encryption_algorithm=None):
        if encryption_algorithm:
            return key.private_bytes(encoding, format, encryption_algorithm).hex()
        else:
            return key.public_bytes(encoding, format).hex()

    def generate_fernet_cipher(self, key):
        fernet_key = urlsafe_b64encode(key)
        return Fernet(fernet_key)

    def encrypt_decrypt_text(self, cipher_suite, text):
        encrypted_text = cipher_suite.encrypt(text)
        decrypted_text = cipher_suite.decrypt(encrypted_text)
        return encrypted_text, decrypted_text.decode("utf-8")

def main():
    key_exchange = KeyExchange()

    # Derive symmetric keys
    symmetric_key = key_exchange.derive_symmetric_key(key_exchange.dapp_private_key, key_exchange.wallet_public_key)
    symmetric_key_alternative = key_exchange.derive_symmetric_key(key_exchange.wallet_private_key, key_exchange.dapp_public_key)

    # Assert both derived keys are equal
    assert symmetric_key == symmetric_key_alternative, "Derived symmetric keys do not match!"

    # Serialize keys
    dapp_private_key_hex = key_exchange.serialize_keys(
        key_exchange.dapp_private_key,
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    wallet_private_key_hex = key_exchange.serialize_keys(
        key_exchange.wallet_private_key,
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    dapp_public_key_hex = key_exchange.serialize_keys(
        key_exchange.dapp_public_key,
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    wallet_public_key_hex = key_exchange.serialize_keys(
        key_exchange.wallet_public_key,
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    # Generate Fernet cipher
    cipher_suite = key_exchange.generate_fernet_cipher(symmetric_key)

    # Encrypt and decrypt sample text
    sample_text = b"hello world"
    encrypted_text, decrypted_text = key_exchange.encrypt_decrypt_text(cipher_suite, sample_text)

    # Print results
    print("DApp Private Key (hex):", dapp_private_key_hex)
    print("Wallet Private Key (hex):", wallet_private_key_hex)
    print("-" * 60)
    print("DApp Public Key (hex):", dapp_public_key_hex)
    print("Wallet Public Key (hex):", wallet_public_key_hex)
    print("-" * 60)
    print("Symmetric Key (hex) dapp private X wallet pubkey:", symmetric_key.hex())
    print("Symmetric2 Key (hex) wallet private X dapp pubkey:", symmetric_key_alternative.hex())
    print("-" * 60)
    print("Encrypted text (bytes):", encrypted_text)
    print("Decrypted text (string):", decrypted_text)

if __name__ == "__main__":
    main()

