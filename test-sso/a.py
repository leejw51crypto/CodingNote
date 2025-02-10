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
import random


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
    )  # Maps session_key -> {main_address, contract_address, permissions, created_at}

    @classmethod
    def register_session_key(
        cls, session_key: str, main_address: str, permissions: Dict
    ):
        """
        Register a new session key with its permissions.
        In real implementation, this would be a smart contract call.
        """
        # Generate a unique smart contract address for this session
        contract_address = f"0x{secrets.token_hex(20)}"

        cls.sessions[session_key] = {
            "main_address": main_address,  # Original wallet address
            "contract_address": contract_address,  # Smart contract address for SSO
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
        - When using SSO (session key), the from_address is the smart contract address
        - Regular transactions use the main wallet address
        - True sender can be verified through transaction calldata and SSO contract
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

            # Create transaction response with SSO details
            tx_response = TransactionResponse(
                tx_hash=f"0x{secrets.token_hex(32)}",
                status=True,
                from_address=session["contract_address"],
                to_address=to_address,
                amount=amount,
                block_number=random.randint(1000000, 9999999),
                timestamp=datetime.now(),
                is_sso_tx=True,
                true_sender=from_address,
                sso_contract=session["contract_address"],
                calldata={
                    "session_key_id": session_key[:10] + "...",
                    "nonce": random.randint(1, 1000),
                    "signature": f"0x{secrets.token_hex(64)}",
                },
                paymaster_data={
                    "contract_version": "v1.0.0",
                    "implementation": f"0x{secrets.token_hex(20)}",
                    "session_registry": f"0x{secrets.token_hex(20)}",
                },
            )
        else:
            # Regular transaction response
            tx_response = TransactionResponse(
                tx_hash=f"0x{secrets.token_hex(32)}",
                status=True,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                block_number=random.randint(1000000, 9999999),
                timestamp=datetime.now(),
                is_sso_tx=False,
            )

        tx_response.display()


@dataclass
class TransactionResponse:
    """
    Represents a detailed transaction response including SSO-specific information.
    This shows how receivers can identify the true sender of SSO transactions.

    For SSO transactions:
    - from_address is the SSO smart contract address
    - true_sender is the original wallet address
    - Verification can be done through system contract calls and paymaster data
    """

    tx_hash: str
    status: bool
    from_address: str  # Smart contract address for SSO txs
    to_address: str
    amount: float
    block_number: int
    timestamp: datetime
    # SSO specific fields
    is_sso_tx: bool
    true_sender: str = None  # Original wallet address for SSO txs
    sso_contract: str = None
    calldata: Dict = None  # Contains SSO verification data
    paymaster_data: Dict = None  # Contains additional verification data

    def display(self):
        print("\n### Transaction Details")
        print(f"* Status: {'✅ Success' if self.status else '❌ Failed'}")
        print(f"* Transaction Hash: `{self.tx_hash}`")
        print(f"* Block Number: {self.block_number}")
        print(f"* Timestamp: {self.timestamp}")

        if self.is_sso_tx:
            print(f"* From (SSO Contract): `{self.from_address}`")
        else:
            print(f"* From: `{self.from_address}`")

        print(f"* To: `{self.to_address}`")
        print(f"* Amount: **{self.amount} ETH**")

        if self.is_sso_tx:
            print("\n### SSO Transaction Information")
            print("* This is an SSO (Smart Sign-On) transaction")
            print(f"* True Sender (Original Wallet): `{self.true_sender}`")
            print(f"* SSO Contract: `{self.sso_contract}`")

            print("\n### Transaction Calldata")
            print("* Verification Data:")
            print(f"  * Session Key ID: `{self.calldata['session_key_id']}`")
            print(f"  * Nonce: {self.calldata['nonce']}")
            print(f"  * Signature: `{self.calldata['signature']}`")

            print("\n### Paymaster Data")
            print("* SSO System Contract Data:")
            print(f"  * Contract Version: `{self.paymaster_data['contract_version']}`")
            print(f"  * Implementation: `{self.paymaster_data['implementation']}`")
            print(f"  * Session Registry: `{self.paymaster_data['session_registry']}`")

            print("\n### How to Verify True Sender")
            print(
                "1. This transaction was sent through SSO contract `{self.from_address}`"
            )
            print("2. To verify the true sender, you can:")
            print("   a) Call the SSO system contract:")
            print(f"      * getSessionInfo({self.calldata['session_key_id']})")
            print("   b) Check session registry:")
            print(f"      * Contract: `{self.paymaster_data['session_registry']}`")
            print(
                f"      * Method: isValidSession({self.true_sender}, {self.calldata['session_key_id']})"
            )
            print("   c) Verify transaction signature:")
            print(f"      * Signer: `{self.true_sender}`")
            print(f"      * Signature: `{self.calldata['signature']}`")
            print("3. On-chain verification:")
            print(f"   * The SSO contract at `{self.sso_contract}`")
            print(f"   * Confirms wallet `{self.true_sender}` as true sender")
            print("   * Through zkSync Era's native SSO verification")


def main():
    """
    Example demonstrating zkSync Era SSO wallet usage:

    1. Create a main wallet (this would be user's actual wallet in real world)
    2. Generate a session key with limited permissions (for SSO)
    3. Demonstrate transaction sending with session key validation
    4. Show how permission limits are enforced
    5. Demonstrate how receivers can identify true sender in SSO transactions

    Note: In a real application:
    - The session key would be used to authenticate with various dApps
    - Receivers can verify the true sender through transaction calldata
    - The SSO contract provides cryptographic proof of the true sender
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
