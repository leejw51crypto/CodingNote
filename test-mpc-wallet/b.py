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
from dotenv import load_dotenv
import rlp
from eth_utils import to_bytes

# Load environment variables
load_dotenv()


@dataclass
class Party:
    """Represents a signing party in the TSS protocol"""

    id: int
    xi: int  # Secret share of the private key
    public_key: bytes  # Public key share


@dataclass
class TSSKeyData:
    """Represents the TSS key setup data"""

    parties: List[Party]
    tss_address: str
    group_public_key: bytes
    threshold: int
    num_parties: int


class ThresholdSignatureScheme:
    def __init__(self):
        self.curve = ec.SECP256K1()
        self.curve_order = (
            0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        )
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

    def _hmac(self, key: bytes, msg: bytes) -> bytes:
        """HMAC using SHA256"""
        block_size = 64  # SHA256 block size
        if len(key) > block_size:
            key = hashlib.sha256(key).digest()
        key = key + b"\x00" * (block_size - len(key))

        o_key_pad = bytes(x ^ 0x5C for x in key)
        i_key_pad = bytes(x ^ 0x36 for x in key)

        return hashlib.sha256(
            o_key_pad + hashlib.sha256(i_key_pad + msg).digest()
        ).digest()

    def _bits2int(self, b: bytes, q: int) -> int:
        """Convert bits to integer modulo q"""
        b_int = int.from_bytes(b, byteorder="big")
        l = len(b) * 8
        qlen = q.bit_length()
        if l > qlen:
            b_int >>= l - qlen
        return b_int

    def _int2octets(self, x: int, q: int) -> bytes:
        """Convert integer x to octet string of length ceil(qlen/8)"""
        qlen = q.bit_length()
        rlen = (qlen + 7) // 8
        return x.to_bytes(rlen, byteorder="big")

    def _bits2octets(self, b: bytes, q: int) -> bytes:
        """Convert bit string to octet string of length ceil(qlen/8)"""
        z1 = self._bits2int(b, q)
        z2 = z1 % q
        return self._int2octets(z2, q)

    def _generate_k_rfc6979(self, message_hash: bytes, private_key: int) -> int:
        """Generate k value according to RFC 6979"""
        V = b"\x01" * 32
        K = b"\x00" * 32
        x = self._int2octets(private_key, self.curve_order)
        h1 = self._bits2octets(message_hash, self.curve_order)
        K = self._hmac(K, V + b"\x00" + x + h1)
        V = self._hmac(K, V)
        K = self._hmac(K, V + b"\x01" + x + h1)
        V = self._hmac(K, V)

        while True:
            T = b""
            while len(T) < 32:
                V = self._hmac(K, V)
                T = T + V

            k = self._bits2int(T, self.curve_order)
            if k >= 1 and k < self.curve_order:
                return k

            K = self._hmac(K, V + b"\x00")
            V = self._hmac(K, V)

    def _generate_base_k(self, message_hash: bytes, common_seed: bytes) -> int:
        """Generate base k value that will be same for all parties using RFC 6979"""
        seed_int = int.from_bytes(common_seed, byteorder="big") % self.curve_order
        return self._generate_k_rfc6979(message_hash, seed_int)

    def _compute_r_point(self, k: int) -> Tuple[int, bytes]:
        """Compute R = k*G and return (r, R_bytes)"""
        k_key = ec.derive_private_key(k, self.curve, default_backend())
        R = k_key.public_key()
        R_bytes = R.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        r = R.public_numbers().x % self.curve_order
        return r, R_bytes

    def create_partial_signature(
        self, party: Party, message_hash: bytes, common_seed: bytes = None
    ) -> Tuple[int, bytes, int, bytes]:
        """Create partial signature with per-party deterministic k"""
        if common_seed is None:
            common_seed = hashlib.sha256(message_hash).digest()

        k = self._generate_base_k(message_hash, common_seed)
        r, R_bytes = self._compute_r_point(k)

        z = int.from_bytes(message_hash, "big")
        k_inv = self._mod_inverse(k, self.curve_order)
        s = (k_inv * (z + (r * party.xi))) % self.curve_order

        return r, s.to_bytes(32, "big"), k, R_bytes

    def combine_partial_signatures(
        self,
        partial_signatures: List[Tuple[int, bytes, int, bytes]],
        parties: List[Party],
        message_hash: bytes,
    ) -> List[SignedTransaction]:
        r = partial_signatures[0][0]
        R_bytes = partial_signatures[0][3]
        for sig_r, _, _, sig_R in partial_signatures[1:]:
            if sig_r != r or sig_R != R_bytes:
                raise ValueError("Inconsistent R values in partial signatures")

        s_combined = 0
        for i, (_, si, _, _) in enumerate(partial_signatures):
            s_i = int.from_bytes(si, "big")
            lambda_i = self._lagrange_coefficient(
                parties[: len(partial_signatures)], parties[i].id
            )
            s_combined = (s_combined + (lambda_i * s_i)) % self.curve_order

        chain_id = 338  # Cronos testnet
        possible_v_values = []
        for base_v in [27, 28]:
            v = chain_id * 2 + 35 + (base_v - 27)
            possible_v_values.append(v)

        if s_combined > self.curve_order // 2:
            s_combined = self.curve_order - s_combined

        r_bytes = r.to_bytes(32, "big")
        s_bytes = s_combined.to_bytes(32, "big")

        signatures = []
        for v in possible_v_values:
            v_bytes = v.to_bytes(32, "big")
            sig = r_bytes + s_bytes + v_bytes

            signed_tx = SignedTransaction(
                rawTransaction=sig,
                hash=Web3.keccak(sig),
                r=r,
                s=s_combined,
                v=v,
            )
            signatures.append(signed_tx)

        return signatures

    def derive_ethereum_address(self, group_public_key: bytes) -> str:
        """Derive Ethereum address from group public key"""
        public_key_bytes = group_public_key[1:]  # Remove '04' prefix
        keccak = Web3.keccak(public_key_bytes)
        address = keccak[-20:].hex()
        return Web3.to_checksum_address(address)


class EthereumTSS:
    def __init__(self):
        self.tss = ThresholdSignatureScheme()

    def setup_existing_key(
        self, private_key: int, threshold: int, num_parties: int
    ) -> TSSKeyData:
        """Set up the TSS protocol with an existing private key"""
        if threshold < 2 or threshold > num_parties:
            raise ValueError("Invalid threshold value")

        coefficients = [private_key]
        for _ in range(threshold - 1):
            coeff = secrets.randbelow(self.tss.curve_order - 1) + 1
            coefficients.append(coeff)

        parties: List[Party] = []
        for i in range(num_parties):
            x = i + 1
            share = coefficients[0]
            for j in range(1, threshold):
                exp = pow(x, j, self.tss.curve_order)
                term = (coefficients[j] * exp) % self.tss.curve_order
                share = (share + term) % self.tss.curve_order

            private_key_obj = ec.derive_private_key(
                share, self.tss.curve, default_backend()
            )
            public_key = private_key_obj.public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )
            parties.append(Party(i + 1, share, public_key))

        group_public_key = (
            ec.derive_private_key(private_key, self.tss.curve, default_backend())
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )
        )
        eth_address = self.tss.derive_ethereum_address(group_public_key)

        return TSSKeyData(
            parties, eth_address, group_public_key, threshold, num_parties
        )

    def create_partial_signature(
        self, party: Party, message_hash: bytes, common_seed: bytes = None
    ) -> Tuple[int, bytes, int, bytes]:
        """Create partial signature for a party"""
        return self.tss.create_partial_signature(party, message_hash, common_seed)

    def combine_signatures(
        self,
        partial_signatures: List[Tuple[int, bytes, int, bytes]],
        parties: List[Party],
        message_hash: bytes,
    ) -> List[SignedTransaction]:
        """Combine partial signatures"""
        return self.tss.combine_partial_signatures(
            partial_signatures, parties, message_hash
        )


def verify_signature(message_hash: bytes, r: int, s: int, v: int, address: str) -> bool:
    """Verify an Ethereum signature using Web3's ecrecover"""
    try:
        w3 = Web3()

        # Create the signature bytes in the correct format
        vrs = (v, r, s)

        # Recover the address
        recovered_address = Account.recover_message(
            signable_message=message_hash, vrs=vrs
        )

        print(f"Recovered address: {recovered_address}")
        print(f"Expected address: {address}")

        return recovered_address.lower() == address.lower()
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return False


def demonstrate_simple_tss():
    # Get private key from environment
    private_key = os.getenv("MY_FULL_PRIVATEKEY")
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    # Convert hex private key to int for TSS
    private_key_int = int(private_key, 16)

    # Set up parameters
    threshold = 2
    num_parties = 3

    # Initialize TSS
    eth_tss = EthereumTSS()
    key_data = eth_tss.setup_existing_key(private_key_int, threshold, num_parties)

    print(f"\n=== Simple TSS Demonstration ===")
    print(f"Generated TSS Address: {key_data.tss_address}")
    print(f"Threshold: {threshold}")
    print(f"Total Parties: {num_parties}")

    # Create message and its hash
    message = "hello world"
    message_bytes = message.encode()

    # Create the Ethereum signed message
    w3 = Web3()
    message_hash = Web3.keccak(
        b"\x19Ethereum Signed Message:\n"
        + str(len(message_bytes)).encode()
        + message_bytes
    )

    print(f"\nSigning message: '{message}'")
    print(f"Message hash: {message_hash.hex()}")

    # Generate common seed for deterministic k
    common_seed = hashlib.sha256(message_hash).digest()

    # Generate partial signatures
    print("\n=== Generating Partial Signatures ===")
    partial_signatures = []
    for i in range(threshold):
        partial_sig = eth_tss.create_partial_signature(
            key_data.parties[i], message_hash, common_seed
        )
        r, s_bytes, k, R_bytes = partial_sig
        print(f"\nParty {i+1} share: {hex(key_data.parties[i].xi)}")
        print(f"Party {i+1} partial s: {hex(int.from_bytes(s_bytes, 'big'))}")
        partial_signatures.append(partial_sig)

    # Combine signatures
    try:
        print("\n=== Combining Signatures ===")
        possible_signed_txns = eth_tss.combine_signatures(
            partial_signatures, key_data.parties, message_hash
        )

        # Try each possible signature
        for idx, signed_txn in enumerate(possible_signed_txns):
            print(f"\nSignature combination {idx + 1}:")
            print(f"r: {hex(signed_txn.r)}")
            print(f"s: {hex(signed_txn.s)}")
            print(f"v: {signed_txn.v}")

            # Create the signature in the correct format
            v = signed_txn.v % 256  # Convert to a single byte
            r = signed_txn.r
            s = signed_txn.s

            # Create the signature bytes in standard format
            signature = r.to_bytes(32, "big") + s.to_bytes(32, "big") + bytes([v])

            try:
                # Create a new account for verification
                account = Account()

                # Recover the address using the hash directly
                recovered_address = account._recover_hash(message_hash, vrs=(v, r, s))

                print(f"Recovered address: {recovered_address}")
                print(f"Expected address: {key_data.tss_address}")
                print(
                    f"Signature valid: {recovered_address.lower() == key_data.tss_address.lower()}"
                )
            except Exception as e:
                print(f"Verification error: {str(e)}")

    except Exception as e:
        print(f"Error in TSS signing: {str(e)}")


if __name__ == "__main__":
    demonstrate_simple_tss()
