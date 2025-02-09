"""
ZkSync Era SSO (Smart Sign-On) Wallet Simulation
This code demonstrates how session keys work in zkSync Era for implementing SSO functionality.

Key Concepts:
1. Session keys are actually private keys with limited permissions
2. When sending transactions via SSO, the sender address remains the original wallet address
3. The receiver cannot distinguish between direct wallet transactions and SSO transactions
4. The system uses a combination of blockchain core (Kernel) and system contracts for SSO

Flow:
User Wallet → SSO → Application
1. User connects their wallet to the app
2. SSO validates their session key
3. User gets access to the app with limited permissions
"""

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List


# Simulate blockchain kernel level
class Kernel:
    """
    Represents the zkSync Era blockchain kernel (core protocol level).
    In real zkSync, session key validation happens at the protocol level,
    making it highly secure and efficient.
    """

    def validate_session_key(self, session_key: str, main_address: str) -> bool:
        # In real zkSync, this validation would be done at protocol level
        # It verifies that the session key is associated with the main wallet address
        return (
            SystemContract.sessions.get(session_key, {}).get("main_address")
            == main_address
        )


# Simulate system smart contract
class SystemContract:
    """
    Simulates the zkSync Era system contract that manages session keys.
    In reality, this would be a smart contract deployed on zkSync Era.

    Key points:
    - Stores the relationship between session keys and main wallets
    - Manages permissions and validity periods
    - Acts as a registry for all active session keys
    """

    sessions: Dict[str, Dict] = (
        {}
    )  # Maps session_key -> {main_address, permissions, created_at}

    @classmethod
    def register_session_key(
        cls, session_key: str, main_address: str, permissions: Dict
    ):
        """
        Register a new session key with its permissions.
        In real implementation, this would be a smart contract call.
        """
        cls.sessions[session_key] = {
            "main_address": main_address,  # Original wallet address
            "permissions": permissions,  # Limited permissions for this session
            "created_at": datetime.now(),  # Timestamp for expiry calculation
        }


# Simulate user wallet
class Wallet:
    """
    Represents a user's wallet that can create session keys for SSO.

    Key points:
    - Main wallet has full control and can generate multiple session keys
    - Session keys have limited permissions (time, spend limits, etc.)
    - When using session keys, transactions still come from the main wallet address
    """

    def __init__(self):
        # Main private key - in real world, this would be your actual wallet key
        self.main_private_key = secrets.token_hex(32)
        self.address = f"0x{secrets.token_hex(20)}"  # Simulate address generation
        self.balance = 1000  # Simulate some ETH

    def generate_session_key(self, permissions: Dict) -> str:
        """
        Generate a new session key (private key) with limited permissions.

        Flow:
        1. Generate new private key for session
        2. Register it with system contract
        3. Return the session private key to the user

        The session key is a real private key but with limited permissions
        compared to the main wallet key.
        """
        session_private_key = secrets.token_hex(32)

        # Register with system contract
        SystemContract.register_session_key(
            session_private_key, self.address, permissions
        )

        return session_private_key


# Simulate zkSync Era blockchain
class ZkSync:
    """
    Simulates the zkSync Era blockchain with SSO capabilities.

    Key points:
    - Transactions via session keys look identical to regular transactions
    - Receivers can't distinguish between direct and SSO transactions
    - All transactions are validated through the kernel
    """

    def __init__(self):
        self.kernel = Kernel()

    def send_transaction(
        self, from_address: str, to_address: str, amount: float, session_key: str = None
    ):
        """
        Send a transaction, either directly or via session key.

        Important:
        - The from_address is always the main wallet address, not a contract address
        - Session key transactions are indistinguishable from regular transactions
        - Receivers cannot tell if a transaction used SSO or not
        """
        # If using session key, validate it through kernel
        if session_key:
            if not self.kernel.validate_session_key(session_key, from_address):
                raise Exception("Invalid session key")

            # Check session permissions
            session = SystemContract.sessions[session_key]
            if amount > session["permissions"]["spend_limit"]:
                raise Exception("Amount exceeds session key spend limit")

            if datetime.now() > session["created_at"] + timedelta(
                days=session["permissions"]["time_limit_days"]
            ):
                raise Exception("Session key expired")

        print("### Transaction Details")
        print(f"* Status: ✅ Success")
        print(f"* From: `{from_address}`")  # Always shows main wallet address
        print(f"* To: `{to_address}`")
        print(f"* Amount: **{amount} ETH**")


def main():
    """
    Example demonstrating zkSync Era SSO wallet usage:

    1. Create a main wallet (this would be user's actual wallet in real world)
    2. Generate a session key with limited permissions (for SSO)
    3. Demonstrate transaction sending with session key validation
    4. Show how permission limits are enforced

    Note: In a real application, the session key would be used to authenticate
    with various dApps without needing the main wallet for every action.
    """
    # Initialize zkSync
    zksync = ZkSync()

    # Create user wallet
    my_wallet = Wallet()
    print("\n## Wallet Creation")
    print(f"* Main wallet address: `{my_wallet.address}`")

    # Generate session key with permissions
    session_permissions = {
        "spend_limit": 100,  # Max 100 ETH per transaction
        "time_limit_days": 7,  # Valid for 7 days
    }

    session_key = my_wallet.generate_session_key(session_permissions)
    print("\n## Session Key Generation")
    print(f"* Generated session key: `{session_key}`")
    print("* Permissions:")
    print(f"  * Spend limit: {session_permissions['spend_limit']} ETH")
    print(f"  * Time limit: {session_permissions['time_limit_days']} days")

    # Try to send transaction using session key
    try:
        # Create a random receiver address
        receiver_address = f"0x{secrets.token_hex(20)}"

        # Example 1: Successful transaction within limits
        print("\n## Transaction Test 1: Within Limits")
        print("* Attempting to send 50 ETH...")
        zksync.send_transaction(
            from_address=my_wallet.address,  # Note: Still uses main wallet address
            to_address=receiver_address,
            amount=50,
            session_key=session_key,
        )

        # Example 2: Failed transaction exceeding limits
        print("\n## Transaction Test 2: Exceeding Limits")
        print("* Attempting to send 150 ETH (should fail)...")
        zksync.send_transaction(
            from_address=my_wallet.address,
            to_address=receiver_address,
            amount=150,
            session_key=session_key,
        )
    except Exception as e:
        print(f"* ❌ Transaction failed: `{str(e)}`")


if __name__ == "__main__":
    main()
