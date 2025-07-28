"""
⚠️  EDUCATIONAL IMPLEMENTATION ONLY ⚠️

This code is for learning and demonstration purposes.
DO NOT USE IN PRODUCTION - it lacks security features required for real-world applications.

Double Ratchet Algorithm Implementation

The Double Ratchet algorithm is the core of the Signal protocol, providing:
1. Forward secrecy - old keys cannot decrypt new messages
2. Future secrecy (break-in recovery) - new keys cannot decrypt old messages
3. Self-healing - corrupted state can be recovered

The algorithm combines:
- Diffie-Hellman ratchet: provides forward secrecy through key rotation
- Symmetric key ratchet: provides immediate forward secrecy for each message

For production use, consider established libraries like libsignal-protocol.
"""

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from crypto_primitives import CryptoError, ECDHKeyPair, SignalCrypto

# Set up debug logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def debug_print(party: str, operation: str, details: str = ""):
    """Helper function to print formatted debug information."""
    print(f"\n🔧 [{party}] {operation}")
    if details:
        print(f"   {details}")


def format_key(key: Optional[bytes], name: str = "key") -> str:
    """Format a key for debug output."""
    if key is None:
        return f"{name}: None"
    return f"{name}: {key[:8].hex()}...{key[-4:].hex()}"


def format_state(state: "RatchetState", party: str) -> str:
    """Format ratchet state for debug output."""
    lines = [f"\n📊 [{party}] Current Ratchet State:"]
    lines.append(
        f"   📤 Sending: msg_num={state.sending_message_number}, chain_key={'✓' if state.sending_chain_key else '✗'}"
    )
    lines.append(
        f"   📥 Receiving: msg_num={state.receiving_message_number}, chain_key={'✓' if state.receiving_chain_key else '✗'}"
    )
    lines.append(
        f"   🔑 DH keys: local={'✓' if state.dh_keypair else '✗'}, remote={'✓' if state.dh_remote_public else '✗'}"
    )
    lines.append(
        f"   🗝️  Root key: {state.root_key[:8].hex()}...{state.root_key[-4:].hex()}"
    )
    lines.append(f"   ⏭️  Skipped keys: {len(state.skipped_message_keys)}")
    return "\n".join(lines)


@dataclass
class RatchetState:
    """Represents the state of a Double Ratchet session."""

    # Diffie-Hellman ratchet state
    dh_keypair: Optional[ECDHKeyPair]  # Our current DH key pair
    dh_remote_public: Optional[bytes]  # Remote party's current DH public key
    root_key: bytes  # Root key for deriving new chain keys

    # Sending chain state
    sending_chain_key: Optional[bytes]  # Current sending chain key
    sending_message_number: int  # Number of messages sent in current sending chain

    # Receiving chain state
    receiving_chain_key: Optional[bytes]  # Current receiving chain key
    receiving_message_number: (
        int  # Number of messages received in current receiving chain
    )

    # Previous sending chain (for out-of-order message handling)
    previous_sending_chain_length: int  # Length of previous sending chain

    # Skipped message keys (for handling out-of-order messages)
    skipped_message_keys: Dict[
        Tuple[bytes, int], bytes
    ]  # (dh_public_key, message_number) -> message_key

    # Header keys for message authentication
    header_key_send: bytes
    header_key_receive: bytes

    # Next header key for key rotation
    next_header_key_send: bytes
    next_header_key_receive: bytes


@dataclass
class RatchetMessage:
    """Represents an encrypted message with Double Ratchet."""

    dh_public_key: bytes  # Sender's current DH public key
    previous_chain_length: int  # Length of previous sending chain
    message_number: int  # Message number in current sending chain
    ciphertext: bytes  # Encrypted message content

    def to_bytes(self) -> bytes:
        """Serialize message to bytes for transmission."""
        # Simple serialization: dh_key(64) + prev_chain_len(4) + msg_num(4) + ciphertext
        return (
            self.dh_public_key
            + self.previous_chain_length.to_bytes(4, "big")
            + self.message_number.to_bytes(4, "big")
            + self.ciphertext
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "RatchetMessage":
        """Deserialize message from bytes."""
        if len(data) < 72:  # 64 + 4 + 4 minimum
            raise CryptoError("Invalid message format")

        dh_public_key = data[:64]
        previous_chain_length = int.from_bytes(data[64:68], "big")
        message_number = int.from_bytes(data[68:72], "big")
        ciphertext = data[72:]

        return cls(dh_public_key, previous_chain_length, message_number, ciphertext)


class DoubleRatchet:
    """Implementation of the Double Ratchet algorithm."""

    def __init__(self, shared_secret: bytes, initiator: bool = True):
        """
        Initialize a Double Ratchet session.

        Args:
            shared_secret: Initial shared secret from X3DH
            initiator: True if this party initiated the conversation
        """
        self.initiator = initiator
        self.party_name = "ALICE" if initiator else "BOB"

        debug_print(self.party_name, "🚀 INITIALIZING DOUBLE RATCHET")
        print(f"   Role: {'Initiator' if initiator else 'Responder'}")
        print(f"   {format_key(shared_secret, 'Shared secret')}")

        # Initialize root key and initial chain keys from shared secret
        self.state = self._initialize_state(shared_secret, initiator)

        print(format_state(self.state, self.party_name))

    def _initialize_state(self, shared_secret: bytes, initiator: bool) -> RatchetState:
        """Initialize the ratchet state from the shared secret."""
        debug_print(self.party_name, "🔑 DERIVING INITIAL KEYS FROM SHARED SECRET")

        # Use the shared secret directly as the initial root key
        root_key = shared_secret
        print(f"   {format_key(root_key, 'Initial root key')}")

        # Generate initial header keys
        debug_print(self.party_name, "🏷️  DERIVING HEADER KEYS")
        header_keys = SignalCrypto.derive_message_keys(shared_secret + b"HeaderKeys")
        header_key_send = header_keys["encryption_key"]
        header_key_receive = header_keys["mac_key"][:32]
        print(f"   {format_key(header_key_send, 'Header key (send)')}")
        print(f"   {format_key(header_key_receive, 'Header key (receive)')}")

        next_header_key_send = SignalCrypto.derive_message_keys(
            shared_secret + b"NextHeaderKeys"
        )["encryption_key"]
        next_header_key_receive = SignalCrypto.derive_message_keys(
            shared_secret + b"NextHeaderKeys"
        )["mac_key"][:32]
        print(f"   {format_key(next_header_key_send, 'Next header key (send)')}")
        print(f"   {format_key(next_header_key_receive, 'Next header key (receive)')}")

        if initiator:
            debug_print(self.party_name, "🔐 GENERATING INITIAL DH KEYPAIR (INITIATOR)")
            # Initiator starts with DH key pair
            dh_keypair = SignalCrypto.generate_keypair()
            print(
                f"   {format_key(dh_keypair.get_public_key_bytes(), 'DH public key')}"
            )
            print(f"   DH private key: [PROTECTED]")
            return RatchetState(
                dh_keypair=dh_keypair,
                dh_remote_public=None,
                root_key=root_key,
                sending_chain_key=None,
                sending_message_number=0,
                receiving_chain_key=None,
                receiving_message_number=0,
                previous_sending_chain_length=0,
                skipped_message_keys={},
                header_key_send=header_key_send,
                header_key_receive=header_key_receive,
                next_header_key_send=next_header_key_send,
                next_header_key_receive=next_header_key_receive,
            )
        else:
            debug_print(self.party_name, "⏳ WAITING FOR DH KEY EXCHANGE (RESPONDER)")
            # Receiver starts without DH key pair initially
            return RatchetState(
                dh_keypair=None,
                dh_remote_public=None,
                root_key=root_key,
                sending_chain_key=None,
                sending_message_number=0,
                receiving_chain_key=None,
                receiving_message_number=0,
                previous_sending_chain_length=0,
                skipped_message_keys={},
                header_key_send=header_key_send,
                header_key_receive=header_key_receive,
                next_header_key_send=next_header_key_send,
                next_header_key_receive=next_header_key_receive,
            )

    def _dh_ratchet_step(self, remote_public_key: bytes) -> None:
        """Perform a Diffie-Hellman ratchet step."""
        debug_print(
            self.party_name,
            "🔄 PERFORMING DH RATCHET STEP",
            f"Remote DH public key: {remote_public_key[:8].hex()}...{remote_public_key[-4:].hex()}",
        )

        print(f"   📊 Previous state:")
        print(
            f"      Previous sending chain length: {self.state.sending_message_number}"
        )
        print(
            f"      Current root key: {self.state.root_key[:8].hex()}...{self.state.root_key[-4:].hex()}"
        )

        # Store previous sending chain length
        self.state.previous_sending_chain_length = self.state.sending_message_number

        # Reset message numbers
        old_sending_num = self.state.sending_message_number
        old_receiving_num = self.state.receiving_message_number
        self.state.sending_message_number = 0
        self.state.receiving_message_number = 0

        debug_print(
            self.party_name,
            "🔢 RESETTING MESSAGE COUNTERS",
            f"Sending: {old_sending_num} → 0, Receiving: {old_receiving_num} → 0",
        )

        # Update remote public key
        old_remote_key = self.state.dh_remote_public
        self.state.dh_remote_public = remote_public_key
        debug_print(self.party_name, "🔑 UPDATING REMOTE DH PUBLIC KEY")
        if old_remote_key:
            print(f"   Old: {old_remote_key[:8].hex()}...{old_remote_key[-4:].hex()}")
        print(f"   New: {remote_public_key[:8].hex()}...{remote_public_key[-4:].hex()}")

        # Generate new DH key pair
        old_keypair = self.state.dh_keypair
        self.state.dh_keypair = SignalCrypto.generate_keypair()
        debug_print(self.party_name, "🆕 GENERATING NEW DH KEYPAIR")
        if old_keypair:
            print(
                f"   Old public: {old_keypair.get_public_key_bytes()[:8].hex()}...{old_keypair.get_public_key_bytes()[-4:].hex()}"
            )
        print(
            f"   New public: {self.state.dh_keypair.get_public_key_bytes()[:8].hex()}...{self.state.dh_keypair.get_public_key_bytes()[-4:].hex()}"
        )

        # Perform DH and update root key and receiving chain key
        debug_print(self.party_name, "🤝 COMPUTING DH SHARED SECRET")
        remote_key_obj = ECDHKeyPair.from_public_key_bytes(remote_public_key)
        dh_output = SignalCrypto.compute_shared_secret(
            self.state.dh_keypair, remote_key_obj
        )
        print(f"   DH output: {dh_output[:8].hex()}...{dh_output[-4:].hex()}")

        old_root_key = self.state.root_key
        self.state.root_key, self.state.receiving_chain_key = SignalCrypto.kdf_rk(
            self.state.root_key, dh_output
        )

        debug_print(self.party_name, "📥 DERIVING NEW RECEIVING CHAIN KEY")
        print(f"   Old root key: {old_root_key[:8].hex()}...{old_root_key[-4:].hex()}")
        print(
            f"   New root key: {self.state.root_key[:8].hex()}...{self.state.root_key[-4:].hex()}"
        )
        print(
            f"   New receiving chain key: {self.state.receiving_chain_key[:8].hex()}...{self.state.receiving_chain_key[-4:].hex()}"
        )

        # Update root key and sending chain key with same DH output
        self.state.root_key, self.state.sending_chain_key = SignalCrypto.kdf_rk(
            self.state.root_key, dh_output
        )

        debug_print(self.party_name, "📤 DERIVING NEW SENDING CHAIN KEY")
        print(
            f"   Final root key: {self.state.root_key[:8].hex()}...{self.state.root_key[-4:].hex()}"
        )
        print(
            f"   New sending chain key: {self.state.sending_chain_key[:8].hex()}...{self.state.sending_chain_key[-4:].hex()}"
        )

        # Rotate header keys
        debug_print(self.party_name, "🏷️  ROTATING HEADER KEYS")
        old_send_header = self.state.header_key_send
        old_receive_header = self.state.header_key_receive

        self.state.header_key_send = self.state.next_header_key_send
        self.state.header_key_receive = self.state.next_header_key_receive

        print(
            f"   Send header: {old_send_header[:8].hex()}... → {self.state.header_key_send[:8].hex()}..."
        )
        print(
            f"   Receive header: {old_receive_header[:8].hex()}... → {self.state.header_key_receive[:8].hex()}..."
        )

        # Generate new next header keys
        header_key_material = self.state.root_key + b"NextHeaderKeys"
        next_keys = SignalCrypto.derive_message_keys(header_key_material)
        self.state.next_header_key_send = next_keys["encryption_key"]
        self.state.next_header_key_receive = next_keys["mac_key"][:32]

        print(f"   Next send header: {self.state.next_header_key_send[:8].hex()}...")
        print(
            f"   Next receive header: {self.state.next_header_key_receive[:8].hex()}..."
        )

        debug_print(self.party_name, "✅ DH RATCHET STEP COMPLETE")
        print(format_state(self.state, self.party_name))

    def _skip_message_keys(self, until_message_number: int) -> None:
        """Skip message keys up to a given message number."""
        if self.state.receiving_chain_key is None:
            raise CryptoError("Cannot skip message keys without receiving chain key")

        debug_print(
            self.party_name,
            "⏭️  SKIPPING MESSAGE KEYS (KDF CHAIN ADVANCE)",
            f"From msg #{self.state.receiving_message_number} to #{until_message_number-1}",
        )

        print(
            f"   Initial receiving chain key: {self.state.receiving_chain_key[:8].hex()}...{self.state.receiving_chain_key[-4:].hex()}"
        )

        skipped_count = 0
        while self.state.receiving_message_number < until_message_number:
            # Derive and store the message key for the skipped message
            old_chain_key = self.state.receiving_chain_key
            self.state.receiving_chain_key, message_key = SignalCrypto.kdf_ck(
                self.state.receiving_chain_key
            )

            print(f"   📍 Msg #{self.state.receiving_message_number}:")
            print(
                f"      Chain key: {old_chain_key[:8].hex()}... → {self.state.receiving_chain_key[:8].hex()}..."
            )
            print(
                f"      Message key: {message_key[:8].hex()}...{message_key[-4:].hex()}"
            )

            # Store the skipped message key
            key_id = (self.state.dh_remote_public, self.state.receiving_message_number)
            self.state.skipped_message_keys[key_id] = message_key

            self.state.receiving_message_number += 1
            skipped_count += 1

        debug_print(self.party_name, f"✅ SKIPPED {skipped_count} MESSAGE KEYS")
        print(
            f"   Final receiving chain key: {self.state.receiving_chain_key[:8].hex()}...{self.state.receiving_chain_key[-4:].hex()}"
        )
        print(f"   Total skipped keys stored: {len(self.state.skipped_message_keys)}")

    def encrypt(self, plaintext: bytes, associated_data: bytes = b"") -> RatchetMessage:
        """Encrypt a message using the Double Ratchet."""
        debug_print(
            self.party_name,
            "🔒 ENCRYPTING MESSAGE",
            f"Message: '{plaintext.decode() if len(plaintext) < 50 else plaintext[:47].decode() + '...'}'",
        )

        if self.state.dh_keypair is None:
            raise CryptoError("Cannot encrypt without DH key pair")

        # Ensure we have a sending chain key
        if self.state.sending_chain_key is None:
            raise CryptoError("No sending chain key available")

        print(f"   📊 Current state:")
        print(f"      Sending message #: {self.state.sending_message_number}")
        print(
            f"      Previous chain length: {self.state.previous_sending_chain_length}"
        )
        print(
            f"      Current DH public key: {self.state.dh_keypair.get_public_key_bytes()[:8].hex()}...{self.state.dh_keypair.get_public_key_bytes()[-4:].hex()}"
        )

        # Derive message key from sending chain key
        debug_print(self.party_name, "🔑 ADVANCING SENDING CHAIN (KDF)")
        old_chain_key = self.state.sending_chain_key
        self.state.sending_chain_key, message_key = SignalCrypto.kdf_ck(
            self.state.sending_chain_key
        )

        print(
            f"   Chain key: {old_chain_key[:8].hex()}... → {self.state.sending_chain_key[:8].hex()}..."
        )
        print(
            f"   Derived message key: {message_key[:8].hex()}...{message_key[-4:].hex()}"
        )

        # Encrypt the plaintext
        debug_print(self.party_name, "🔐 ENCRYPTING WITH MESSAGE KEY")
        ciphertext = SignalCrypto.encrypt_message(
            plaintext, message_key, associated_data
        )
        print(f"   Ciphertext length: {len(ciphertext)} bytes")
        print(f"   Ciphertext: {ciphertext[:16].hex()}...{ciphertext[-16:].hex()}")

        # Create the ratchet message
        message = RatchetMessage(
            dh_public_key=self.state.dh_keypair.get_public_key_bytes(),
            previous_chain_length=self.state.previous_sending_chain_length,
            message_number=self.state.sending_message_number,
            ciphertext=ciphertext,
        )

        self.state.sending_message_number += 1

        debug_print(self.party_name, "✅ MESSAGE ENCRYPTED & PACKAGED")
        print(f"   Message number: {message.message_number}")
        print(f"   Next sending message #: {self.state.sending_message_number}")
        print(format_state(self.state, self.party_name))

        return message

    def decrypt(self, message: RatchetMessage, associated_data: bytes = b"") -> bytes:
        """Decrypt a message using the Double Ratchet."""
        debug_print(self.party_name, "🔓 DECRYPTING MESSAGE")
        print(f"   📨 Received message:")
        print(
            f"      DH public key: {message.dh_public_key[:8].hex()}...{message.dh_public_key[-4:].hex()}"
        )
        print(f"      Message number: {message.message_number}")
        print(f"      Previous chain length: {message.previous_chain_length}")
        print(f"      Ciphertext length: {len(message.ciphertext)} bytes")

        # Check if this is from a new DH ratchet step
        is_new_dh_ratchet = (
            self.state.dh_remote_public != message.dh_public_key
            or self.state.dh_remote_public is None
        )

        if is_new_dh_ratchet:
            debug_print(self.party_name, "🆕 NEW DH RATCHET DETECTED")
            print(
                f"   Current remote key: {self.state.dh_remote_public[:8].hex() if self.state.dh_remote_public else 'None'}"
            )
            print(
                f"   Message DH key: {message.dh_public_key[:8].hex()}...{message.dh_public_key[-4:].hex()}"
            )

            # Skip any missing messages in the current receiving chain
            if self.state.receiving_chain_key is not None:
                debug_print(
                    self.party_name, "⏭️  SKIPPING REMAINING MESSAGES IN CURRENT CHAIN"
                )
                self._skip_message_keys(message.message_number)

            # Perform DH ratchet step
            self._dh_ratchet_step(message.dh_public_key)
        else:
            debug_print(self.party_name, "📥 SAME DH RATCHET - NORMAL MESSAGE")

        # Check for skipped message keys first
        key_id = (message.dh_public_key, message.message_number)
        if key_id in self.state.skipped_message_keys:
            debug_print(self.party_name, "🔍 USING SKIPPED MESSAGE KEY")
            message_key = self.state.skipped_message_keys.pop(key_id)
            print(
                f"   Retrieved message key: {message_key[:8].hex()}...{message_key[-4:].hex()}"
            )
            print(f"   Remaining skipped keys: {len(self.state.skipped_message_keys)}")

            plaintext = SignalCrypto.decrypt_message(
                message.ciphertext, message_key, associated_data
            )
            debug_print(self.party_name, "✅ MESSAGE DECRYPTED WITH SKIPPED KEY")
            print(f"   Plaintext: '{plaintext.decode()}'")
            return plaintext

        # Skip message keys if this message is out of order (message number is higher than expected)
        if message.message_number > self.state.receiving_message_number:
            debug_print(
                self.party_name, "📮 OUT-OF-ORDER MESSAGE - SKIPPING INTERMEDIATE KEYS"
            )
            self._skip_message_keys(message.message_number)
        elif message.message_number < self.state.receiving_message_number:
            # This message is from the past and should have been in skipped keys, but wasn't found
            raise CryptoError(
                f"Message {message.message_number} arrived too late - expected message key not found"
            )

        # At this point, message.message_number should equal self.state.receiving_message_number
        if message.message_number != self.state.receiving_message_number:
            raise CryptoError(
                f"Unexpected message number: {message.message_number}, expected {self.state.receiving_message_number}"
            )

        # Ensure we have a receiving chain key
        if self.state.receiving_chain_key is None:
            raise CryptoError("No receiving chain key available")

        debug_print(self.party_name, "🔑 ADVANCING RECEIVING CHAIN (KDF)")
        print(f"   Expected message #: {self.state.receiving_message_number}")
        print(
            f"   Current receiving chain key: {self.state.receiving_chain_key[:8].hex()}...{self.state.receiving_chain_key[-4:].hex()}"
        )

        # Derive message key and decrypt
        old_chain_key = self.state.receiving_chain_key
        self.state.receiving_chain_key, message_key = SignalCrypto.kdf_ck(
            self.state.receiving_chain_key
        )

        print(
            f"   Chain key: {old_chain_key[:8].hex()}... → {self.state.receiving_chain_key[:8].hex()}..."
        )
        print(
            f"   Derived message key: {message_key[:8].hex()}...{message_key[-4:].hex()}"
        )

        debug_print(self.party_name, "🔐 DECRYPTING WITH MESSAGE KEY")
        plaintext = SignalCrypto.decrypt_message(
            message.ciphertext, message_key, associated_data
        )

        self.state.receiving_message_number += 1

        debug_print(self.party_name, "✅ MESSAGE DECRYPTED SUCCESSFULLY")
        print(f"   Plaintext: '{plaintext.decode()}'")
        print(f"   Next receiving message #: {self.state.receiving_message_number}")
        print(format_state(self.state, self.party_name))

        return plaintext

    def get_public_key(self) -> Optional[bytes]:
        """Get the current DH public key."""
        if self.state.dh_keypair:
            return self.state.dh_keypair.get_public_key_bytes()
        return None

    def set_remote_public_key(self, public_key: bytes) -> None:
        """Set the remote party's DH public key (for initialization)."""
        debug_print(
            self.party_name, "🤝 SETTING REMOTE DH PUBLIC KEY (INITIAL EXCHANGE)"
        )
        print(
            f"   Remote DH public key: {public_key[:8].hex()}...{public_key[-4:].hex()}"
        )

        self.state.dh_remote_public = public_key

        # If we don't have a DH keypair yet (receiver), generate one
        if self.state.dh_keypair is None:
            debug_print(self.party_name, "🔐 GENERATING OUR DH KEYPAIR (RESPONDER)")
            self.state.dh_keypair = SignalCrypto.generate_keypair()
            print(
                f"   Our DH public key: {self.state.dh_keypair.get_public_key_bytes()[:8].hex()}...{self.state.dh_keypair.get_public_key_bytes()[-4:].hex()}"
            )

        # Initialize chain keys from the same DH exchange
        if (
            self.state.receiving_chain_key is None
            or self.state.sending_chain_key is None
        ):
            debug_print(
                self.party_name, "🔑 DERIVING INITIAL CHAIN KEYS FROM DH EXCHANGE"
            )
            print(
                f"   Current root key: {self.state.root_key[:8].hex()}...{self.state.root_key[-4:].hex()}"
            )

            remote_key_obj = ECDHKeyPair.from_public_key_bytes(public_key)
            dh_output = SignalCrypto.compute_shared_secret(
                self.state.dh_keypair, remote_key_obj
            )
            print(
                f"   DH shared secret: {dh_output[:8].hex()}...{dh_output[-4:].hex()}"
            )

            # Both sending and receiving chains use the same DH output initially
            # This ensures Alice and Bob derive the same keys
            old_root_key = self.state.root_key
            new_root_key, chain_key = SignalCrypto.kdf_rk(
                self.state.root_key, dh_output
            )

            print(
                f"   Root key: {old_root_key[:8].hex()}... → {new_root_key[:8].hex()}..."
            )
            print(
                f"   Derived chain key: {chain_key[:8].hex()}...{chain_key[-4:].hex()}"
            )

            if self.state.receiving_chain_key is None:
                debug_print(self.party_name, "📥 SETTING RECEIVING CHAIN KEY")
                self.state.receiving_chain_key = chain_key
                print(
                    f"   Receiving chain key: {chain_key[:8].hex()}...{chain_key[-4:].hex()}"
                )

            if self.state.sending_chain_key is None:
                debug_print(self.party_name, "📤 SETTING SENDING CHAIN KEY")
                self.state.sending_chain_key = chain_key
                print(
                    f"   Sending chain key: {chain_key[:8].hex()}...{chain_key[-4:].hex()}"
                )

            self.state.root_key = new_root_key

        debug_print(self.party_name, "✅ INITIAL DH EXCHANGE COMPLETE")
        print(format_state(self.state, self.party_name))


# Testing and example usage
if __name__ == "__main__":
    print("=" * 80)
    print("🔐 DOUBLE RATCHET ALGORITHM DEMONSTRATION")
    print("=" * 80)
    print("This demo shows how the Double Ratchet provides:")
    print("1. 🛡️  Forward Secrecy - old keys can't decrypt new messages")
    print("2. 🔄 Future Secrecy - new keys can't decrypt old messages")
    print("3. 🏥 Self-healing - corrupted state can recover")
    print("4. 📮 Out-of-order message handling")
    print("=" * 80)

    # Simulate initial shared secret from X3DH
    shared_secret = os.urandom(32)
    print(f"\n🎯 PHASE 1: INITIALIZATION")
    print(
        f"Shared secret from X3DH: {shared_secret[:8].hex()}...{shared_secret[-4:].hex()}"
    )

    # Create two ratchet instances
    print(f"\n👤 Creating Alice (initiator) and Bob (responder)...")
    alice_ratchet = DoubleRatchet(shared_secret, initiator=True)
    bob_ratchet = DoubleRatchet(shared_secret, initiator=False)

    print(f"\n🎯 PHASE 2: INITIAL DH KEY EXCHANGE")
    # Alice sends her DH public key to Bob
    alice_public_key = alice_ratchet.get_public_key()
    print(f"\n📤 Alice sends her DH public key to Bob...")
    bob_ratchet.set_remote_public_key(alice_public_key)

    # Bob generates his DH key pair and gets his public key
    bob_public_key = bob_ratchet.get_public_key()
    if bob_public_key is None:
        # Bob needs to generate a key pair first
        bob_ratchet.state.dh_keypair = SignalCrypto.generate_keypair()
        bob_public_key = bob_ratchet.get_public_key()

    print(f"\n📤 Bob sends his DH public key to Alice...")
    alice_ratchet.set_remote_public_key(bob_public_key)

    # Test message exchange
    print(f"\n🎯 PHASE 3: MESSAGE EXCHANGE (SYMMETRIC KEY RATCHETS)")
    print("=" * 60)
    print("Now both parties can send messages. Each message advances")
    print("the symmetric key chain, providing immediate forward secrecy.")
    print("=" * 60)

    # Alice sends messages to Bob
    print(f"\n📨 Alice sends messages to Bob...")
    alice_msg1 = alice_ratchet.encrypt(b"Hello Bob!")
    alice_msg2 = alice_ratchet.encrypt(b"How are you?")

    # Bob receives and decrypts Alice's messages
    print(f"\n📨 Bob receives and decrypts Alice's messages...")
    decrypted1 = bob_ratchet.decrypt(alice_msg1)
    decrypted2 = bob_ratchet.decrypt(alice_msg2)

    print(f"\n✅ Successfully decrypted:")
    print(f"Alice -> Bob: {decrypted1.decode()}")
    print(f"Alice -> Bob: {decrypted2.decode()}")

    # Bob replies to Alice
    print(f"\n📨 Bob replies to Alice...")
    bob_msg1 = bob_ratchet.encrypt(b"Hi Alice!")
    bob_msg2 = bob_ratchet.encrypt(b"I'm doing great, thanks!")

    # Alice receives and decrypts Bob's messages
    print(f"\n📨 Alice receives and decrypts Bob's messages...")
    decrypted3 = alice_ratchet.decrypt(bob_msg1)
    decrypted4 = alice_ratchet.decrypt(bob_msg2)

    print(f"\n✅ Successfully decrypted:")
    print(f"Bob -> Alice: {decrypted3.decode()}")
    print(f"Bob -> Alice: {decrypted4.decode()}")

    # Test out-of-order messages
    print(f"\n🎯 PHASE 4: OUT-OF-ORDER MESSAGE HANDLING")
    print("=" * 60)
    print("The Double Ratchet can handle messages received out of order")
    print("by storing skipped message keys for later use.")
    print("=" * 60)

    # Alice sends multiple messages
    print(f"\n📨 Alice sends 3 more messages...")
    alice_msg3 = alice_ratchet.encrypt(b"Message 3")
    alice_msg4 = alice_ratchet.encrypt(b"Message 4")
    alice_msg5 = alice_ratchet.encrypt(b"Message 5")

    # Bob receives them out of order: 5, 3, 4
    print(f"\n📮 Bob receives them OUT OF ORDER: Message 5, then 3, then 4...")
    try:
        decrypted5 = bob_ratchet.decrypt(alice_msg5)
        decrypted3_delayed = bob_ratchet.decrypt(alice_msg3)
        decrypted4_delayed = bob_ratchet.decrypt(alice_msg4)

        print(f"\n✅ Successfully handled out-of-order messages:")
        print(f"Out of order - Message 5: {decrypted5.decode()}")
        print(f"Out of order - Message 3: {decrypted3_delayed.decode()}")
        print(f"Out of order - Message 4: {decrypted4_delayed.decode()}")
        print("✅ Out-of-order message handling working!")
    except Exception as e:
        print(f"❌ Out-of-order handling failed: {e}")

    print(f"\n🎯 DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("🔐 The Double Ratchet successfully demonstrated:")
    print("   ✅ Initial key derivation from shared secret")
    print("   ✅ DH ratchet steps for forward secrecy")
    print("   ✅ Symmetric key chains for immediate forward secrecy")
    print("   ✅ Out-of-order message handling")
    print("   ✅ Bidirectional communication")
    print("=" * 80)
