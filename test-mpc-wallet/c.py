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

    def split_existing_key(
        self, private_key: int, threshold: int, num_parties: int
    ) -> Tuple[List[Party], bytes]:
        if threshold < 2 or threshold > num_parties:
            raise ValueError("Invalid threshold value")

        coefficients = [private_key]
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(self.curve_order - 1) + 1)

        parties: List[Party] = []

        for i in range(num_parties):
            x = i + 1
            share = coefficients[0]
            for j in range(1, threshold):
                exp = pow(x, j, self.curve_order)
                term = (coefficients[j] * exp) % self.curve_order
                share = (share + term) % self.curve_order

            private_key_obj = ec.derive_private_key(
                share, self.curve, default_backend()
            )
            public_key = private_key_obj.public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )

            parties.append(Party(i + 1, share, public_key))

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
        k = shared_randomness.get("k")
        if not k:
            raise ValueError("k value not provided in shared randomness")

        k_key = ec.derive_private_key(k, self.curve, default_backend())
        R = k_key.public_key()
        r = R.public_numbers().x % self.curve_order

        z = int.from_bytes(message_hash, "big")
        k_inv = self._mod_inverse(k, self.curve_order)
        s = (k_inv * (z + (r * party.xi))) % self.curve_order

        return r, s.to_bytes(32, "big")

    def combine_partial_signatures(
        self,
        partial_signatures: List[Tuple[int, bytes]],
        parties: List[Party],
        message_hash: bytes,
    ) -> SignedTransaction:
        r = partial_signatures[0][0]

        s_combined = 0
        for i, (_, si) in enumerate(partial_signatures):
            s_i = int.from_bytes(si, "big")
            lambda_i = self._lagrange_coefficient(
                parties[: len(partial_signatures)], parties[i].id
            )
            s_combined = (s_combined + (lambda_i * s_i)) % self.curve_order

        # Calculate v with EIP-155 replay protection
        chain_id = 338  # Cronos testnet
        v = chain_id * 2 + 35 + (27 - 27)  # Base v is 27

        if s_combined > self.curve_order // 2:
            s_combined = self.curve_order - s_combined
            v = chain_id * 2 + 35 + (28 - 27)  # Base v is 28

        # Create signature bytes
        r_bytes = r.to_bytes(32, "big")
        s_bytes = s_combined.to_bytes(32, "big")
        v_bytes = v.to_bytes(32, "big")

        return SignedTransaction(
            rawTransaction=r_bytes + s_bytes + v_bytes,
            hash=Web3.keccak(r_bytes + s_bytes + v_bytes),
            r=r,
            s=s_combined,
            v=v,
        )

    def derive_ethereum_address(self, group_public_key: bytes) -> str:
        public_key_bytes = group_public_key[1:]
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
        parties, group_public_key = self.tss.split_existing_key(
            private_key, threshold, num_parties
        )
        eth_address = self.tss.derive_ethereum_address(group_public_key)
        self.shared_randomness = {"k": secrets.randbelow(self.tss.curve_order - 1) + 1}
        return parties, eth_address

    def create_partial_signature(
        self, party: Party, message_hash: bytes
    ) -> Tuple[int, bytes]:
        return self.tss.create_partial_signature(
            party, message_hash, self.shared_randomness
        )

    def combine_signatures(
        self,
        partial_signatures: List[Tuple[int, bytes]],
        parties: List[Party],
        message_hash: bytes,
    ) -> SignedTransaction:
        return self.tss.combine_partial_signatures(
            partial_signatures, parties, message_hash
        )


def send_eth_with_tss(
    to_address: str, amount_eth: float, threshold: int = 2, num_parties: int = 3
):
    # Configure web3 with Cronos testnet
    RPC_ENDPOINT = "https://evm-t3.cronos.org/"
    CHAIN_ID = 338

    # Connect to network
    w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
    if not w3.is_connected():
        raise Exception("Failed to connect to the network")

    # Get private key from environment
    private_key = os.getenv("MY_FULL_PRIVATEKEY")
    if private_key.startswith("0x"):
        private_key = private_key[2:]

    # Convert hex private key to int for TSS
    private_key_int = int(private_key, 16)

    # Initialize TSS
    eth_tss = EthereumTSS()
    parties, tss_address = eth_tss.setup_existing_key(
        private_key_int, threshold, num_parties
    )

    print(f"TSS Address: {tss_address}")
    print(f"Threshold: {threshold}")
    print(f"Total Parties: {num_parties}")

    # Show initial balances
    sender_balance_before = w3.eth.get_balance(tss_address)
    receiver_balance_before = w3.eth.get_balance(to_address)
    print("\nInitial balances:")
    print(f"Sender balance: {w3.from_wei(sender_balance_before, 'ether')} TCRO")
    print(f"Receiver balance: {w3.from_wei(receiver_balance_before, 'ether')} TCRO")

    # Prepare transaction
    nonce = w3.eth.get_transaction_count(tss_address)
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
        transaction["data"],  # empty bytes for simple transfer
        w3.to_bytes(hexstr=hex(transaction["chainId"])),
        b"",  # s
        b"",  # r
    ]

    # Create RLP encoding of the transaction
    rlp_encoded = rlp.encode(transaction_dict)
    message_hash = Web3.keccak(rlp_encoded)

    # Generate partial signatures
    partial_signatures = []
    for i in range(threshold):
        partial_sig = eth_tss.create_partial_signature(parties[i], message_hash)
        print(f"Party {i+1} partial signature generated")
        partial_signatures.append(partial_sig)

    # Combine signatures
    try:
        signed_txn = eth_tss.combine_signatures(
            partial_signatures, parties, message_hash
        )

        # Create the final transaction
        final_transaction = rlp.encode(
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
                w3.to_bytes(hexstr=hex(signed_txn.v)) if signed_txn.v != 0 else b"\x00",
                w3.to_bytes(hexstr=hex(signed_txn.r)) if signed_txn.r != 0 else b"\x00",
                w3.to_bytes(hexstr=hex(signed_txn.s)) if signed_txn.s != 0 else b"\x00",
            ]
        )

        # Recover and verify sender address from signature
        recovered_address = w3.eth.account.recover_transaction(final_transaction)
        print(f"\nRecovered sender address from signature: {recovered_address}")
        print(f"Expected TSS address: {tss_address}")
        if recovered_address.lower() != tss_address.lower():
            raise ValueError("Recovered address doesn't match TSS address!")

        # Send transaction
        tx_hash = w3.eth.send_raw_transaction(final_transaction)
        print(f"\nTransaction sent! Hash: {tx_hash.hex()}")

        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Transaction confirmed in block {tx_receipt['blockNumber']}")

        gas_used = tx_receipt["gasUsed"]
        gas_price = transaction["gasPrice"]
        gas_cost = gas_used * gas_price
        print(f"Gas used: {gas_used} (Cost: {w3.from_wei(gas_cost, 'ether')} TCRO)")

        # Show final balances
        sender_balance = w3.eth.get_balance(tss_address)
        receiver_balance = w3.eth.get_balance(to_address)
        print("\nFinal balances:")
        print(f"TSS address balance: {w3.from_wei(sender_balance, 'ether')} TCRO")
        print(f"Receiver balance: {w3.from_wei(receiver_balance, 'ether')} TCRO")
        print(f"Gas cost: {w3.from_wei(gas_cost, 'ether')} TCRO")

    except Exception as e:
        print(f"Error in TSS transaction: {str(e)}")


if __name__ == "__main__":
    to_address = os.getenv("MY_TO_ADDRESS")
    if not to_address:
        raise ValueError("MY_TO_ADDRESS not set in environment")

    # Send 0.1 TCRO using TSS
    send_eth_with_tss(to_address, 0.1)
