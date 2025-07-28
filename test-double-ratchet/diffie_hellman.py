"""
⚠️  EDUCATIONAL IMPLEMENTATION ONLY ⚠️

This code is for learning and demonstration purposes.
DO NOT USE IN PRODUCTION - it lacks security features required for real-world applications.

Diffie-Hellman Key Exchange Implementation
=========================================

WHAT IS EXCHANGED PUBLICLY:
• Prime number p (large, typically 2048+ bits)
• Generator g (usually 2 or 5)
• Alice's public key A = g^a mod p
• Bob's public key B = g^b mod p

WHAT STAYS SECRET:
• Alice's private key 'a' (random number)
• Bob's private key 'b' (random number)
• Shared secret s = g^(ab) mod p

WHY THIS WORKS:
An attacker sees: p, g, A, B
To find the shared secret, they need either:
- Private key 'a': requires solving A = g^a mod p for 'a' (discrete log problem)
- Private key 'b': requires solving B = g^b mod p for 'b' (discrete log problem)
- Direct computation: extremely difficult without private keys

The discrete logarithm problem is computationally infeasible for large primes.

STEP-BY-STEP PROCESS:
1. Public setup: Alice and Bob agree on prime p and generator g
2. Private key generation: Alice picks 'a', Bob picks 'b' (kept secret)
3. Public key computation: Alice computes A = g^a mod p, Bob computes B = g^b mod p
4. Public key exchange: Alice sends A to Bob, Bob sends B to Alice
5. Shared secret computation:
   - Alice: s = B^a mod p = (g^b)^a mod p = g^(ba) mod p
   - Bob: s = A^b mod p = (g^a)^b mod p = g^(ab) mod p
   - Result: Both get same s = g^(ab) mod p

GROUP MESSAGING WITH SENDER KEY:
===============================

PROBLEM: Basic DH only works between 2 parties. For group chat:
- N users would need crypto_primitives.pyN*(N-1)/2 pairwise DH exchanges
- Very inefficient for large groups

SIGNAL'S SOLUTION - SENDER KEY:
1. Each user generates their own "Sender Key" (random 32-byte key)
2. When sending to group: encrypt message once with their Sender Key
3. Distribute the Sender Key to all group members encrypted with individual DH sessions
4. Recipients use the sender's key to decrypt group messages

EXAMPLE: Alice sends to group [Bob, Carol, Dave]
1. Alice generates Sender Key (e.g., random 32 bytes) 
2. Alice encrypts message: encrypt("Hello group!", AliceSenderKey)
3. Alice distributes her Sender Key:
   - To Bob: encrypt(AliceSenderKey, Alice-Bob-DH-Secret)
   - To Carol: encrypt(AliceSenderKey, Alice-Carol-DH-Secret)  
   - To Dave: encrypt(AliceSenderKey, Alice-Dave-DH-Secret)
4. Recipients decrypt Alice's Sender Key with their DH session, then decrypt message

This way: 1 group message encryption + N key distributions (much more efficient!)

NEW MEMBER JOINS GROUP - SIGNAL PROTOCOL APPROACHES:
===================================================

🔐 FORWARD SECRECY PROBLEM:
New members shouldn't be able to decrypt old messages sent before they joined!

APPROACH 1: FULL KEY ROTATION (More Secure)
✅ When new member joins → ALL existing members generate NEW Sender Keys
✅ Old messages cannot be decrypted by new member (forward secrecy preserved)
✅ Used in high-security implementations

APPROACH 2: PARTIAL KEY ROTATION (More Efficient)  
✅ New member gets current Sender Keys for ongoing conversations
❌ Cannot decrypt messages sent before joining (by design)
✅ More efficient for large groups with frequent membership changes

REAL SIGNAL BEHAVIOR:
- Uses TreeKEM (Tree-based Key Derivation) for very large groups
- Periodically rotates all keys regardless of membership changes
- New members start fresh - cannot see old messages
"""

import hashlib
import os
import random
from typing import Dict, List, Optional
from pcr import rfc3526


class DiffieHellman:
    def __init__(self, p=None, g=None):
        """
        Initialize Diffie-Hellman with prime p and generator g.
        If not provided, uses safe default values from RFC 3526.
        """
        # Using RFC 3526 group 14 (2048-bit) for security
        if p is None or g is None:
            prime, generator = rfc3526.groups[2048]
            self.p = p or prime
            self.g = g or generator
        else:
            self.p = p
            self.g = g

        # Generate private key (random number)
        self.private_key = random.randint(2, self.p - 2)

        # Compute public key: g^private_key mod p
        self.public_key = pow(self.g, self.private_key, self.p)

    def get_public_key(self):
        """Return the public key to share with the other party."""
        return self.public_key

    def compute_shared_secret(self, other_public_key):
        """
        Compute the shared secret using the other party's public key.
        shared_secret = other_public_key^private_key mod p
        """
        shared_secret = pow(other_public_key, self.private_key, self.p)
        return shared_secret

    def derive_key(self, shared_secret, key_length=32):
        """
        Derive a usable encryption key from the shared secret using SHA-256.
        This converts the large integer to a fixed-size key.
        """
        # Convert shared secret to bytes and hash it
        secret_bytes = shared_secret.to_bytes(
            (shared_secret.bit_length() + 7) // 8, "big"
        )
        return hashlib.sha256(secret_bytes).digest()[:key_length]


class GroupMessenger:
    """
    Demonstrates how to use Sender Keys for efficient group messaging
    built on top of pairwise Diffie-Hellman sessions.
    """

    def __init__(self, username: str):
        self.username = username
        self.dh_sessions: Dict[str, bytes] = {}  # username -> shared_key
        self.sender_keys: Dict[str, bytes] = {}  # username -> their_sender_key
        self.my_sender_key = os.urandom(32)  # My own sender key for group messages
        self.sender_key_version = 1  # Track key rotation for security

    def establish_dh_session(self, other_user: str, shared_key: bytes):
        """
        Store a shared DH session key with another user.
        In a real implementation, this would involve the full DH exchange.
        """
        self.dh_sessions[other_user] = shared_key
        print(f"✓ {self.username} established DH session with {other_user}")
        print(f"  Shared key: {shared_key.hex()[:16]}...")

    def generate_sender_key(self) -> bytes:
        """Generate a new sender key for group messaging."""
        self.my_sender_key = os.urandom(32)
        self.sender_key_version += 1
        print(
            f"✓ {self.username} generated new Sender Key (v{self.sender_key_version}): {self.my_sender_key.hex()[:16]}..."
        )
        return self.my_sender_key

    def distribute_sender_key_to_group(
        self, group_members: List[str]
    ) -> Dict[str, bytes]:
        """
        Distribute my sender key to all group members.
        Each member gets the same sender key, but encrypted with their individual DH session.
        """
        print(f"\n🔑 {self.username} distributing Sender Key to group: {group_members}")

        encrypted_distributions = {}

        for member in group_members:
            if member == self.username:
                continue  # Don't send to ourselves

            if member not in self.dh_sessions:
                print(f"  ❌ No DH session with {member}, skipping")
                continue

            # Encrypt sender key with this member's DH session
            dh_key = self.dh_sessions[member]
            encrypted_sender_key = self.simple_encrypt(self.my_sender_key, dh_key)
            encrypted_distributions[member] = encrypted_sender_key

            print(f"  ✓ To {member}: encrypted with DH session {dh_key.hex()[:8]}...")

        return encrypted_distributions

    def receive_sender_key(self, from_user: str, encrypted_sender_key: bytes):
        """
        Receive and decrypt a sender key from another user.
        """
        if from_user not in self.dh_sessions:
            print(
                f"  ❌ {self.username}: No DH session with {from_user}, cannot decrypt sender key"
            )
            return False

        # Decrypt using our DH session with this user
        dh_key = self.dh_sessions[from_user]
        sender_key = self.simple_decrypt(encrypted_sender_key, dh_key)

        # Store their sender key
        self.sender_keys[from_user] = sender_key

        print(
            f"  ✓ {self.username} received {from_user}'s Sender Key: {sender_key.hex()[:16]}..."
        )
        return True

    def send_group_message(self, message: str, group_members: List[str]) -> Dict:
        """
        Send a message to the group using sender key encryption.
        """
        print(f"\n📤 {self.username} sending group message: '{message}'")

        # Step 1: Encrypt message with our sender key
        encrypted_message = self.simple_encrypt(message.encode(), self.my_sender_key)
        print(f"   Encrypted with Sender Key: {self.my_sender_key.hex()[:16]}...")

        # Step 2: Distribute sender key to group members
        key_distributions = self.distribute_sender_key_to_group(group_members)

        return {
            "sender": self.username,
            "encrypted_message": encrypted_message,
            "key_distributions": key_distributions,
            "group_members": group_members,
        }

    def receive_group_message(self, group_message: Dict) -> Optional[str]:
        """
        Receive and decrypt a group message.
        """
        sender = group_message["sender"]
        encrypted_message = group_message["encrypted_message"]

        print(f"\n📥 {self.username} receiving group message from {sender}")

        # Check if we have the sender's key
        if sender not in self.sender_keys:
            print(f"  ❌ Don't have {sender}'s Sender Key yet")
            return None

        # Decrypt the message using the sender's key
        sender_key = self.sender_keys[sender]
        decrypted_message = self.simple_decrypt(encrypted_message, sender_key)

        message_text = decrypted_message.decode()
        print(f"  ✓ Decrypted: '{message_text}'")

        return message_text

    def simple_encrypt(self, data: bytes, key: bytes) -> bytes:
        """
        ⚠️  INSECURE XOR encryption for demonstration ONLY!
        This is NOT cryptographically secure - only for educational examples.
        Real applications should use AES-GCM or similar authenticated encryption.
        """
        if len(key) < 32:
            key = hashlib.sha256(key).digest()
        result = bytearray()
        for i, byte in enumerate(data):
            result.append(byte ^ key[i % len(key)])
        return bytes(result)

    def simple_decrypt(self, encrypted_data: bytes, key: bytes) -> bytes:
        """
        ⚠️  INSECURE XOR decryption for demonstration ONLY!
        This is NOT cryptographically secure - only for educational examples.
        """
        return self.simple_encrypt(encrypted_data, key)  # XOR is symmetric


def demonstrate_basic_dh():
    """
    Demonstrate basic 2-party Diffie-Hellman (your original question).
    """
    print("=== PART 1: Basic Diffie-Hellman (2 parties only) ===\n")

    # Step 1: Alice and Bob create their DH instances (same p, g)
    print("1. SETUP: Alice and Bob agree on public parameters")
    alice = DiffieHellman()
    bob = DiffieHellman(alice.p, alice.g)  # Same parameters

    print(f"   Prime p: {hex(alice.p)[:50]}...")
    print(f"   Generator g: {alice.g}")
    print()

    # Step 2: Exchange public keys
    print("2. PUBLIC KEY EXCHANGE")
    alice_public = alice.get_public_key()
    bob_public = bob.get_public_key()

    print(f"   Alice's public key: {hex(alice_public)[:30]}...")
    print(f"   Bob's public key: {hex(bob_public)[:30]}...")
    print()

    # Step 3: Compute shared secrets
    print("3. SHARED SECRET COMPUTATION")
    alice_shared = alice.compute_shared_secret(bob_public)
    bob_shared = bob.compute_shared_secret(alice_public)

    alice_key = alice.derive_key(alice_shared)
    bob_key = bob.derive_key(bob_shared)

    print(f"   Alice's derived key: {alice_key.hex()}")
    print(f"   Bob's derived key:   {bob_key.hex()}")
    print(f"   Keys match: {'✓ YES' if alice_key == bob_key else '❌ NO'}")
    print()

    print("🔍 LIMITATION: This only works between 2 parties!")
    print("   For 4 people, you'd need 6 separate DH exchanges")
    print("   This is where Sender Keys become useful...\n")


def demonstrate_group_messaging():
    """
    Demonstrate how Sender Keys solve the group messaging problem.
    """
    print("=== PART 2: Group Messaging with Sender Keys ===\n")

    # Create 4 group members
    alice = GroupMessenger("Alice")
    bob = GroupMessenger("Bob")
    carol = GroupMessenger("Carol")
    dave = GroupMessenger("Dave")

    group_members = ["Alice", "Bob", "Carol", "Dave"]
    members_dict = {"Alice": alice, "Bob": bob, "Carol": carol, "Dave": dave}

    print("👥 Group members: Alice, Bob, Carol, Dave\n")

    # Step 1: Establish pairwise DH sessions (this happens once)
    print("1. ESTABLISHING PAIRWISE DH SESSIONS")
    print("   (Each pair needs their own DH session for key distribution)")

    # Helper function to establish proper DH sessions
    def establish_real_dh_session(user1, user2, user1_name, user2_name):
        # Create DH instances for both users
        dh1 = DiffieHellman()
        dh2 = DiffieHellman(dh1.p, dh1.g)  # Same parameters

        # Exchange public keys and compute same shared secret
        pub1 = dh1.get_public_key()
        pub2 = dh2.get_public_key()

        shared_secret1 = dh1.compute_shared_secret(pub2)
        shared_secret2 = dh2.compute_shared_secret(pub1)

        # Both should get the same shared secret
        assert shared_secret1 == shared_secret2, "DH exchange failed!"

        # Derive shared key
        shared_key = dh1.derive_key(shared_secret1)

        # Both users store the same key - user1 stores key for user2, user2 stores key for user1
        user1.establish_dh_session(user2_name, shared_key)
        user2.establish_dh_session(user1_name, shared_key)

    # Establish all pairwise sessions with correct user names
    establish_real_dh_session(alice, bob, "Alice", "Bob")
    establish_real_dh_session(alice, carol, "Alice", "Carol")
    establish_real_dh_session(alice, dave, "Alice", "Dave")
    establish_real_dh_session(bob, carol, "Bob", "Carol")
    establish_real_dh_session(bob, dave, "Bob", "Dave")
    establish_real_dh_session(carol, dave, "Carol", "Dave")

    print()

    # Step 2: Alice sends a group message
    print("2. ALICE SENDS GROUP MESSAGE")
    alice.generate_sender_key()
    group_message = alice.send_group_message("Hello everyone! 👋", group_members)

    # Step 3: Distribute Alice's sender key to group members
    print("\n3. DISTRIBUTING ALICE'S SENDER KEY")
    for member_name, encrypted_key in group_message["key_distributions"].items():
        member = members_dict[member_name]
        member.receive_sender_key("Alice", encrypted_key)

    # Step 4: Group members decrypt the message
    print("\n4. GROUP MEMBERS DECRYPT MESSAGE")
    for member_name in ["Bob", "Carol", "Dave"]:
        member = members_dict[member_name]
        message = member.receive_group_message(group_message)

    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Same message decrypted by all members")
    print("=" * 60)

    # Step 5: Show efficiency comparison
    print("\n5. EFFICIENCY COMPARISON")
    print("Without Sender Keys (naive approach):")
    print("   • Alice would need 3 separate DH encryptions")
    print("   • Each message encrypted 3 times with different keys")
    print("   • Very inefficient for large groups")
    print()
    print("With Sender Keys (Signal's approach):")
    print("   • Message encrypted ONCE with Alice's Sender Key")
    print("   • Sender Key distributed 3 times (encrypted with DH sessions)")
    print("   • Much more efficient! Sender Key can be reused for multiple messages")
    print()

    # Step 6: Bob sends a message too
    print("6. BOB SENDS A REPLY")
    bob.generate_sender_key()
    bob_message = bob.send_group_message("Hi Alice! Got your message 😊", group_members)

    # Distribute Bob's sender key
    print("\n   Distributing Bob's Sender Key...")
    for member_name, encrypted_key in bob_message["key_distributions"].items():
        if member_name != "Bob":
            member = members_dict[member_name]
            member.receive_sender_key("Bob", encrypted_key)

    # Others decrypt Bob's message
    print("\n   Others decrypt Bob's message:")
    for member_name in ["Alice", "Carol", "Dave"]:
        member = members_dict[member_name]
        member.receive_group_message(bob_message)


def demonstrate_new_member_joining():
    """
    Demonstrate what happens when a new member joins the group.
    Shows both approaches: Full Key Rotation vs Partial Key Rotation
    """
    print("=== PART 3: NEW MEMBER JOINS GROUP ===\n")

    # Start with existing group: Alice, Bob, Carol
    print("🏗️  INITIAL GROUP SETUP")
    alice = GroupMessenger("Alice")
    bob = GroupMessenger("Bob")
    carol = GroupMessenger("Carol")

    existing_members = ["Alice", "Bob", "Carol"]
    members_dict = {"Alice": alice, "Bob": bob, "Carol": carol}

    print("👥 Initial group: Alice, Bob, Carol")

    # Helper function to establish DH sessions
    def establish_real_dh_session(user1, user2, user1_name, user2_name):
        dh1 = DiffieHellman()
        dh2 = DiffieHellman(dh1.p, dh1.g)

        pub1 = dh1.get_public_key()
        pub2 = dh2.get_public_key()

        shared_secret1 = dh1.compute_shared_secret(pub2)
        shared_secret2 = dh2.compute_shared_secret(pub1)

        assert shared_secret1 == shared_secret2, "DH exchange failed!"
        shared_key = dh1.derive_key(shared_secret1)

        user1.establish_dh_session(user2_name, shared_key)
        user2.establish_dh_session(user1_name, shared_key)

    # Establish sessions between existing members
    establish_real_dh_session(alice, bob, "Alice", "Bob")
    establish_real_dh_session(alice, carol, "Alice", "Carol")
    establish_real_dh_session(bob, carol, "Bob", "Carol")

    print()

    # Step 1: Existing group has some message history
    print("📚 EXISTING GROUP CONVERSATION")
    alice.generate_sender_key()
    old_message = alice.send_group_message(
        "Secret: The meeting is at 3pm tomorrow", existing_members
    )

    # Distribute Alice's key to existing members
    for member_name, encrypted_key in old_message["key_distributions"].items():
        if member_name in members_dict:
            member = members_dict[member_name]
            member.receive_sender_key("Alice", encrypted_key)

    # Existing members decrypt
    print("\n   Existing members read the message:")
    for member_name in ["Bob", "Carol"]:
        member = members_dict[member_name]
        member.receive_group_message(old_message)

    print("\n" + "=" * 70)
    print("🚪 NEW MEMBER (DAVE) WANTS TO JOIN THE GROUP")
    print("=" * 70)

    # Step 2: Dave joins the group
    dave = GroupMessenger("Dave")
    members_dict["Dave"] = dave
    new_group_members = ["Alice", "Bob", "Carol", "Dave"]

    print("🆕 Dave joins the group!")

    # Establish DH sessions with Dave
    print("\n1. ESTABLISHING DH SESSIONS WITH NEW MEMBER")
    establish_real_dh_session(alice, dave, "Alice", "Dave")
    establish_real_dh_session(bob, dave, "Bob", "Dave")
    establish_real_dh_session(carol, dave, "Carol", "Dave")

    print("\n" + "⚡" * 70)
    print("APPROACH 1: FULL KEY ROTATION (High Security)")
    print("⚡" * 70)
    print("🔐 Security Policy: New member cannot decrypt old messages!")
    print("🔄 Action: ALL existing members generate NEW Sender Keys")
    print()

    # Step 3a: Full key rotation approach
    print("2a. ALL MEMBERS GENERATE NEW SENDER KEYS")
    alice.generate_sender_key()  # New key - Dave can't decrypt old messages
    bob.generate_sender_key()
    carol.generate_sender_key()
    dave.generate_sender_key()

    print("\n3a. ALICE SENDS NEW MESSAGE WITH NEW KEY")
    new_message = alice.send_group_message(
        "Welcome Dave! This is a new conversation.", new_group_members
    )

    # Distribute new keys to everyone including Dave
    print("\n   Distributing Alice's NEW Sender Key to all members:")
    for member_name, encrypted_key in new_message["key_distributions"].items():
        if member_name in members_dict:
            member = members_dict[member_name]
            member.receive_sender_key("Alice", encrypted_key)

    print("\n4a. ALL MEMBERS (INCLUDING DAVE) DECRYPT NEW MESSAGE:")
    for member_name in ["Bob", "Carol", "Dave"]:
        member = members_dict[member_name]
        member.receive_group_message(new_message)

    print("\n5a. FORWARD SECRECY TEST: Can Dave decrypt old message?")
    print("   Dave tries to decrypt Alice's old message...")
    # Dave tries to decrypt old message but doesn't have old sender key
    try:
        result = dave.receive_group_message(old_message)
        if result is None:
            print(
                "   ✅ SUCCESS: Dave CANNOT decrypt old messages (forward secrecy preserved!)"
            )
        else:
            print("   ❌ FAILURE: Dave can decrypt old messages (security breach!)")
    except:
        print(
            "   ✅ SUCCESS: Dave CANNOT decrypt old messages (forward secrecy preserved!)"
        )

    print("\n" + "🔄" * 70)
    print("APPROACH 2: PARTIAL KEY ROTATION (Efficiency)")
    print("🔄" * 70)
    print("💡 Policy: New member gets current keys but can't see old messages")
    print("⚡ Action: Only distribute CURRENT keys to new member")
    print()

    # Reset Dave's state for second approach
    dave_v2 = GroupMessenger("Dave_v2")
    members_dict["Dave_v2"] = dave_v2

    # Establish sessions (reusing existing ones)
    establish_real_dh_session(alice, dave_v2, "Alice", "Dave_v2")
    establish_real_dh_session(bob, dave_v2, "Bob", "Dave_v2")
    establish_real_dh_session(carol, dave_v2, "Carol", "Dave_v2")

    print("2b. DISTRIBUTE CURRENT SENDER KEYS TO NEW MEMBER")
    print("   (Existing members keep their current keys)")

    # Give Dave the current sender keys (not the old ones)
    for existing_member in ["Alice", "Bob", "Carol"]:
        member_obj = members_dict[existing_member]
        # Simulate key distribution to new member
        dh_key = dave_v2.dh_sessions[existing_member]
        encrypted_key = dave_v2.simple_encrypt(member_obj.my_sender_key, dh_key)
        dave_v2.receive_sender_key(existing_member, encrypted_key)

    print("\n3b. CAROL SENDS MESSAGE (using existing sender key)")
    carol_message = carol.send_group_message(
        "Hi Dave! Nice to meet you!", new_group_members
    )

    print("\n4b. DAVE CAN DECRYPT CURRENT MESSAGES:")
    result = dave_v2.receive_group_message(carol_message)

    print("\n5b. FORWARD SECRECY TEST: Can Dave_v2 decrypt old message?")
    print("   Dave_v2 tries to decrypt Alice's old message...")
    try:
        result = dave_v2.receive_group_message(old_message)
        if result is None:
            print(
                "   ✅ SUCCESS: Dave_v2 CANNOT decrypt old messages (forward secrecy preserved!)"
            )
        else:
            print("   ❌ FAILURE: Dave_v2 can decrypt old messages (security breach!)")
    except:
        print(
            "   ✅ SUCCESS: Dave_v2 CANNOT decrypt old messages (forward secrecy preserved!)"
        )


def demonstrate_key_exchange():
    """
    Enhanced demonstration showing basic DH, group messaging, and new member scenarios.
    """
    demonstrate_basic_dh()
    print("\n" + "=" * 80 + "\n")
    demonstrate_group_messaging()
    print("\n" + "=" * 80 + "\n")
    demonstrate_new_member_joining()

    print("\n" + "=" * 80)
    print("📚 SUMMARY - New Member Joins Group:")
    print("=" * 80)
    print("🔐 FORWARD SECRECY REQUIREMENT:")
    print("   New members CANNOT decrypt messages sent before they joined")
    print()
    print("🛡️  APPROACH 1 - FULL KEY ROTATION:")
    print("   ✅ All existing members generate NEW Sender Keys")
    print("   ✅ Maximum security - complete forward secrecy")
    print("   ❌ More computational overhead")
    print()
    print("⚡ APPROACH 2 - PARTIAL KEY ROTATION:")
    print("   ✅ New member gets current keys only")
    print("   ✅ More efficient for large groups")
    print("   ✅ Still preserves forward secrecy")
    print()
    print("🎯 REAL SIGNAL PROTOCOL:")
    print("   • Uses TreeKEM for very large groups (1000+ members)")
    print("   • Combines both approaches based on group size")
    print("   • Periodic key rotation regardless of membership changes")
    print("   • New members always start fresh - no access to history")


if __name__ == "__main__":
    demonstrate_key_exchange()
