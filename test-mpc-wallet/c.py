import os
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Tuple, Dict
import secrets
import hashlib
from eth_account.datastructures import SignedTransaction
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
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
        # a.  Process m through the hash function H, yielding: h1 = H(m)
        #     (h1 is already passed as message_hash)

        # b.  Set: V = 0x01 0x01 0x01 ... 0x01 (32 bytes)
        V = b"\x01" * 32

        # c.  Set: K = 0x00 0x00 0x00 ... 0x00 (32 bytes)
        K = b"\x00" * 32

        # d.  Set: K = HMAC_K(V || 0x00 || int2octets(x) || bits2octets(h1))
        x = self._int2octets(private_key, self.curve_order)
        h1 = self._bits2octets(message_hash, self.curve_order)
        K = self._hmac(K, V + b"\x00" + x + h1)

        # e.  Set: V = HMAC_K(V)
        V = self._hmac(K, V)

        # f.  Set: K = HMAC_K(V || 0x01 || int2octets(x) || bits2octets(h1))
        K = self._hmac(K, V + b"\x01" + x + h1)

        # g.  Set: V = HMAC_K(V)
        V = self._hmac(K, V)

        # h.  Apply the following algorithm until a proper value is found for k:
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
        # Use common_seed as private key input to ensure all parties generate same k
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
            # Generate a common seed if not provided
            common_seed = hashlib.sha256(message_hash).digest()

        # Generate base k value - same for all parties
        k = self._generate_base_k(message_hash, common_seed)

        # Compute R = k*G - will be same for all parties
        r, R_bytes = self._compute_r_point(k)

        # Compute partial signature
        z = int.from_bytes(message_hash, "big")
        k_inv = self._mod_inverse(k, self.curve_order)

        # Each party computes their share of s = k^(-1)(z + r*x)
        s = (k_inv * (z + (r * party.xi))) % self.curve_order

        return r, s.to_bytes(32, "big"), k, R_bytes

    def combine_partial_signatures(
        self,
        partial_signatures: List[Tuple[int, bytes, int, bytes]],
        parties: List[Party],
        message_hash: bytes,
    ) -> List[SignedTransaction]:
        # Verify all r values are the same
        r = partial_signatures[0][0]
        R_bytes = partial_signatures[0][3]
        for sig_r, _, _, sig_R in partial_signatures[1:]:
            if sig_r != r or sig_R != R_bytes:
                raise ValueError("Inconsistent R values in partial signatures")

        # Combine s values using Lagrange interpolation
        s_combined = 0
        for i, (_, si, _, _) in enumerate(partial_signatures):
            s_i = int.from_bytes(si, "big")
            lambda_i = self._lagrange_coefficient(
                parties[: len(partial_signatures)], parties[i].id
            )
            s_combined = (s_combined + (lambda_i * s_i)) % self.curve_order

        # Calculate v with EIP-155 replay protection
        chain_id = 338  # Cronos testnet

        # Try both possible v values
        possible_v_values = []
        for base_v in [27, 28]:
            v = chain_id * 2 + 35 + (base_v - 27)
            possible_v_values.append(v)

        # Normalize s according to EIP-2
        if s_combined > self.curve_order // 2:
            s_combined = self.curve_order - s_combined

        # Create signature components
        r_bytes = r.to_bytes(32, "big")
        s_bytes = s_combined.to_bytes(32, "big")

        # Return all possible signatures
        signatures = []
        for v in possible_v_values:
            v_bytes = v.to_bytes(32, "big")
            sig = r_bytes + s_bytes + v_bytes

            # Create SignedTransaction object
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
        public_key_bytes = group_public_key[1:]
        keccak = Web3.keccak(public_key_bytes)
        address = keccak[-20:].hex()
        return Web3.to_checksum_address(address)


class EthereumTSS:
    def __init__(self):
        self.tss = ThresholdSignatureScheme()

    def setup_existing_key(
        self, private_key: int, threshold: int, num_parties: int
    ) -> TSSKeyData:
        """Set up the TSS protocol with an existing private key and return the key data"""
        if threshold < 2 or threshold > num_parties:
            raise ValueError("Invalid threshold value")

        # Generate coefficients for polynomial
        coefficients = [private_key]
        for _ in range(threshold - 1):
            coeff = secrets.randbelow(self.tss.curve_order - 1) + 1
            coefficients.append(coeff)

        parties: List[Party] = []
        # Generate shares using polynomial evaluation
        for i in range(num_parties):
            x = i + 1
            share = coefficients[0]
            for j in range(1, threshold):
                exp = pow(x, j, self.tss.curve_order)
                term = (coefficients[j] * exp) % self.tss.curve_order
                share = (share + term) % self.tss.curve_order

            # Generate public key for share
            private_key_obj = ec.derive_private_key(
                share, self.tss.curve, default_backend()
            )
            public_key = private_key_obj.public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )
            parties.append(Party(i + 1, share, public_key))

        # Generate group public key
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


def setup_tss_key(
    private_key_hex: str, threshold: int = 2, num_parties: int = 3
) -> TSSKeyData:
    """Set up TSS key splitting - this is the only place where the full private key is used"""
    # Convert hex private key to int for TSS
    if private_key_hex.startswith("0x"):
        private_key_hex = private_key_hex[2:]
    private_key_int = int(private_key_hex, 16)

    # Get public key for display purposes
    private_key_obj = ec.derive_private_key(
        private_key_int, ec.SECP256K1(), default_backend()
    )
    public_key_bytes = private_key_obj.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )

    # Display key information
    print(f"\n=== Key Information ===")
    masked_private_key = f"0x{private_key_hex[:2]}{'*' * (len(private_key_hex) - 4)}{private_key_hex[-2:]}"
    print(f"Private Key: {masked_private_key}")
    print(f"Public Key: 0x{public_key_bytes.hex()}")

    # Initialize TSS and split the key
    eth_tss = EthereumTSS()
    key_data = eth_tss.setup_existing_key(private_key_int, threshold, num_parties)

    print(f"\n=== TSS Setup Information ===")
    print(f"TSS Address: {key_data.tss_address}")
    print(f"Threshold: {key_data.threshold}")
    print(f"Total Parties: {key_data.num_parties}")

    # Display partial secret information
    for i, party in enumerate(key_data.parties):
        secret_bytes = party.xi.to_bytes(
            (party.xi.bit_length() + 7) // 8, byteorder="big"
        )
        hex_value = secret_bytes.hex()
        masked_hex = f"0x{hex_value[:2]}{'*' * (len(hex_value) - 4)}{hex_value[-2:]}"
        print(f"\nParty {i+1} Secret Share:")
        print(f"- Length: {len(secret_bytes)} bytes")
        print(f"- Value (hex): {masked_hex}")

    return key_data


def send_eth_with_tss_participants(
    key_data: TSSKeyData,
    to_address: str,
    amount_eth: float,
):
    """Send ETH using TSS participants - no access to full private key here"""
    # Configure web3 with Cronos testnet
    RPC_ENDPOINT = "https://evm-t3.cronos.org/"
    CHAIN_ID = 338

    # Connect to network
    w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
    if not w3.is_connected():
        raise Exception("Failed to connect to the network")

    # Show initial balances
    sender_balance_before = w3.eth.get_balance(key_data.tss_address)
    receiver_balance_before = w3.eth.get_balance(to_address)
    print("\n=== Initial Balances ===")
    print(f"Sender balance: {w3.from_wei(sender_balance_before, 'ether')} TCRO")
    print(f"Receiver balance: {w3.from_wei(receiver_balance_before, 'ether')} TCRO")

    # Prepare transaction
    nonce = w3.eth.get_transaction_count(key_data.tss_address)
    amount_wei = w3.to_wei(amount_eth, "ether")

    transaction = {
        "nonce": nonce,
        "gasPrice": w3.eth.gas_price,
        "gas": 21000,  # Standard gas limit for ETH transfer
        "to": to_address,
        "value": amount_wei,
        "chainId": CHAIN_ID,
        "data": b"",
    }

    # Estimate gas
    try:
        estimated_gas = w3.eth.estimate_gas(transaction)
        transaction["gas"] = estimated_gas
    except Exception as e:
        print(f"Warning: Could not estimate gas: {e}")

    # Get transaction hash for signing
    transaction_dict = [
        w3.to_bytes(hexstr=hex(transaction["nonce"])),
        w3.to_bytes(hexstr=hex(transaction["gasPrice"])),
        w3.to_bytes(hexstr=hex(transaction["gas"])),
        w3.to_bytes(hexstr=transaction["to"]),
        w3.to_bytes(hexstr=hex(transaction["value"])),
        transaction["data"],
        w3.to_bytes(hexstr=hex(transaction["chainId"])),
        b"",  # s
        b"",  # r
    ]

    # Create RLP encoding of the transaction
    rlp_encoded = rlp.encode(transaction_dict)
    message_hash = Web3.keccak(rlp_encoded)
    print(f"\n=== Transaction Hash ===")
    print(f"Message hash: 0x{message_hash.hex()}")

    # Generate a single common seed for all parties
    common_seed = hashlib.sha256(message_hash + str(nonce).encode()).digest()
    print("\n=== Common Seed ===")
    print(f"Common seed: 0x{common_seed.hex()}")

    # Initialize TSS for signing
    eth_tss = EthereumTSS()

    # Generate partial signatures
    print("\n=== Generating Partial Signatures ===")
    partial_signatures = []
    for i in range(key_data.threshold):
        partial_sig = eth_tss.create_partial_signature(
            key_data.parties[i], message_hash, common_seed
        )
        r, s_bytes, k, R_bytes = partial_sig
        r_hex = hex(r)[2:].zfill(64)
        s_hex = s_bytes.hex().zfill(64)
        print(f"\nParty {i+1} Partial Signature:")
        print(f"- r value (hex): 0x{r_hex}")
        print(
            f"- r length: {len(r.to_bytes((r.bit_length() + 7) // 8, byteorder='big'))} bytes"
        )
        print(f"- s value (hex): 0x{s_hex}")
        print(f"- s length: {len(s_bytes)} bytes")
        partial_signatures.append(partial_sig)

    # Combine signatures
    try:
        print("\n=== Combining Signatures ===")
        possible_signed_txns = eth_tss.combine_signatures(
            partial_signatures, key_data.parties, message_hash
        )

        signed_txn = None
        final_transaction = None

        # Try each possible signature and find the one that matches our TSS address
        for idx, possible_txn in enumerate(possible_signed_txns):
            r_hex = hex(possible_txn.r)[2:].zfill(64)
            s_hex = hex(possible_txn.s)[2:].zfill(64)
            print(f"\nTrying signature combination {idx + 1}:")
            print(f"- r value (hex): 0x{r_hex}")
            print(f"- s value (hex): 0x{s_hex}")
            print(f"- v value: {possible_txn.v}")
            print(
                f"- r length: {len(possible_txn.r.to_bytes((possible_txn.r.bit_length() + 7) // 8, byteorder='big'))} bytes"
            )
            print(
                f"- s length: {len(possible_txn.s.to_bytes((possible_txn.s.bit_length() + 7) // 8, byteorder='big'))} bytes"
            )

            # Create the final transaction
            test_transaction = rlp.encode(
                [
                    (
                        w3.to_bytes(hexstr=hex(transaction["nonce"]))
                        if transaction["nonce"] != 0
                        else b"\x00"
                    ),
                    w3.to_bytes(hexstr=hex(transaction["gasPrice"])),
                    w3.to_bytes(hexstr=hex(transaction["gas"])),
                    w3.to_bytes(hexstr=transaction["to"]),
                    w3.to_bytes(hexstr=hex(transaction["value"])),
                    transaction["data"],
                    (
                        w3.to_bytes(hexstr=hex(possible_txn.v))
                        if possible_txn.v != 0
                        else b"\x00"
                    ),
                    (
                        w3.to_bytes(hexstr=hex(possible_txn.r))
                        if possible_txn.r != 0
                        else b"\x00"
                    ),
                    (
                        w3.to_bytes(hexstr=hex(possible_txn.s))
                        if possible_txn.s != 0
                        else b"\x00"
                    ),
                ]
            )

            # Recover and verify sender address from signature
            recovered_address = w3.eth.account.recover_transaction(test_transaction)
            print(f"Recovered sender address: {recovered_address}")
            print(f"Expected TSS address: {key_data.tss_address}")

            if recovered_address.lower() == key_data.tss_address.lower():
                print("\n=== Valid Signature Found! ===")
                signed_txn = possible_txn
                final_transaction = test_transaction
                break

        if signed_txn is None:
            raise ValueError("No valid signature found that matches TSS address!")

        # Send transaction
        tx_hash = w3.eth.send_raw_transaction(final_transaction)
        print(f"\n=== Transaction Sent ===")
        print(f"Transaction hash: {tx_hash.hex()}")

        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"\n=== Transaction Confirmed ===")
        print(f"Block number: {tx_receipt['blockNumber']}")

        gas_used = tx_receipt["gasUsed"]
        gas_price = transaction["gasPrice"]
        gas_cost = gas_used * gas_price
        print(f"Gas used: {gas_used} (Cost: {w3.from_wei(gas_cost, 'ether')} TCRO)")

        # Show final balances
        sender_balance = w3.eth.get_balance(key_data.tss_address)
        receiver_balance = w3.eth.get_balance(to_address)
        print("\n=== Final Balances ===")
        print(f"TSS address balance: {w3.from_wei(sender_balance, 'ether')} TCRO")
        print(f"Receiver balance: {w3.from_wei(receiver_balance, 'ether')} TCRO")
        print(f"Gas cost: {w3.from_wei(gas_cost, 'ether')} TCRO")

    except Exception as e:
        print(f"Error in TSS transaction: {str(e)}")


if __name__ == "__main__":
    to_address = os.getenv("MY_TO_ADDRESS")
    if not to_address:
        raise ValueError("MY_TO_ADDRESS not set in environment")

    private_key = os.getenv("MY_FULL_PRIVATEKEY")
    if not private_key:
        raise ValueError("MY_FULL_PRIVATEKEY not set in environment")

    # First phase: Split the private key (this is the only place where full private key is used)
    key_data = setup_tss_key(private_key)

    # Second phase: Use the split keys to send transaction (no access to full private key)
    send_eth_with_tss_participants(key_data, to_address, 0.1)
