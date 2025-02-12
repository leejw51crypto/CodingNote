"""
ZkSync Era SSO (Smart Sign-On) Wallet Simulation
This code demonstrates the house access model of zkSync Era SSO:

🏠 House Access Model:
├── Owner's Key (Passkey)
│   └── Proves ownership
│   └── Required for important changes
│   └── Hardware-backed security
│   └── Biometric verification
│
└── Guest Pass (Session Key)
    └── Limited access
    └── Convenient for frequent use
    └── Pre-authorized by owner
    └── Time and value restricted
"""

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
import random


class ImportantOperation(Enum):
    """Operations that require Owner's Key (Passkey)"""

    CREATE_SESSION = "create_session"  # Create new guest pass
    REVOKE_SESSION = "revoke_session"  # Revoke guest access
    CHANGE_SETTINGS = "change_settings"  # Modify house settings
    HIGH_VALUE_TX = "high_value_tx"  # High-value transfers
    RECOVERY = "recovery"  # Account recovery


class PasskeyStorage(Enum):
    """Where the owner's key (passkey) can be securely stored"""

    PLATFORM_AUTHENTICATOR = "platform"  # iOS/MacOS Keychain, Windows Hello
    PASSWORD_MANAGER = "1password"  # 1Password integration
    SECURITY_KEY = "security_key"  # Yubikey etc


@dataclass
class PasskeyCredential:
    """
    Owner's Key (Passkey) - Like a secure house key
    - Proves ownership of the house (wallet)
    - Required for important changes
    - Stored in secure hardware
    - Requires biometric verification
    """

    credential_id: str  # WebAuthn credential ID
    public_key: str  # Public key bytes
    storage: PasskeyStorage  # Where it's stored
    requires_biometric: bool  # Whether biometric auth is required
    origin: str  # Website that created the passkey

    def authorize_operation(
        self, operation: ImportantOperation, use_biometric: bool = True
    ) -> str:
        """Authorize important operations (like using the owner's key)"""
        if self.requires_biometric and not use_biometric:
            raise Exception("🔐 Biometric verification required for owner's key")
        # In reality, this would use the hardware security module
        return f"auth_{secrets.token_hex(16)}"


@dataclass
class GuestPassConfig:
    """
    Guest Pass (Session Key) Configuration
    - Like configuring what a guest can access in your house
    - Limited to specific areas (contracts)
    - Time-restricted access
    - Value limits for safety
    """

    expiry: timedelta  # How long the guest pass is valid
    allowed_rooms: List[str]  # Allowed contract addresses (areas of house)
    allowed_actions: List[str]  # Allowed functions (what they can do)
    max_daily_spend: float  # Daily spending limit
    max_per_action: float  # Max value per transaction
    max_gas_per_action: float  # Max gas fee per action


@dataclass
class SmartWalletHome:
    """
    Smart Wallet (Like a Smart Home)
    - Secured by owner's key (passkey)
    - Can issue guest passes (session keys)
    - Tracks all access and usage
    """

    address: str  # Home address
    owner_key: PasskeyCredential  # Owner's key (passkey)
    guest_passes: Dict[str, GuestPassConfig]  # Active guest passes
    guest_usage: Dict[str, float]  # Track guest activity
    important_changes: List[Dict]  # Log of important changes

    def validate_guest_access(
        self, guest_pass_id: str, room: str = None, action: str = None, value: float = 0
    ) -> bool:
        """Validate if a guest's access attempt is allowed"""
        if guest_pass_id not in self.guest_passes:
            return False

        guest_pass = self.guest_passes[guest_pass_id]
        current_time = datetime.now()

        # Check if guest pass is still valid
        if current_time > current_time + guest_pass.expiry:
            return False

        # Check if trying to access allowed room
        if room and room not in guest_pass.allowed_rooms:
            return False

        # Check if action is allowed
        if action and action not in guest_pass.allowed_actions:
            return False

        # Check value limits
        if value > guest_pass.max_per_action:
            return False

        # Check daily usage limit
        daily_usage = self.guest_usage.get(guest_pass_id, 0)
        if daily_usage + value > guest_pass.max_daily_spend:
            return False

        return True

    def log_important_change(self, operation: ImportantOperation, details: Dict):
        """Log important changes that required owner's key"""
        self.important_changes.append(
            {
                "timestamp": datetime.now(),
                "operation": operation.value,
                "details": details,
            }
        )


class HomeSecuritySystem:
    """
    Auth Server (Like a Home Security System)
    - Manages owner verification
    - Issues and tracks guest passes
    - Monitors all access attempts
    """

    def __init__(self):
        self.homes: Dict[str, SmartWalletHome] = {}
        self.access_logs: List[Dict] = []

    def register_new_home(self, origin: str) -> SmartWalletHome:
        """Register new home with owner's key (like setting up a new smart home)"""
        owner_key = PasskeyCredential(
            credential_id=secrets.token_hex(32),
            public_key=f"pk_{secrets.token_hex(32)}",
            storage=PasskeyStorage.PLATFORM_AUTHENTICATOR,
            requires_biometric=True,
            origin=origin,
        )

        home = SmartWalletHome(
            address=f"0x{secrets.token_hex(20)}",
            owner_key=owner_key,
            guest_passes={},
            guest_usage={},
            important_changes=[],
        )

        self.homes[home.address] = home
        return home

    def issue_guest_pass(self, home_address: str, config: GuestPassConfig) -> str:
        """Issue a new guest pass (like programming a temporary access code)"""
        if home_address not in self.homes:
            raise Exception("Home not found")

        home = self.homes[home_address]
        # Require owner's key authorization
        auth = home.owner_key.authorize_operation(ImportantOperation.CREATE_SESSION)

        guest_pass_id = secrets.token_hex(16)
        home.guest_passes[guest_pass_id] = config
        home.guest_usage[guest_pass_id] = 0.0

        home.log_important_change(
            ImportantOperation.CREATE_SESSION,
            {"guest_pass_id": guest_pass_id, "config": str(config)},
        )

        return guest_pass_id


@dataclass
class AccessAttempt:
    """Record of an access attempt (like a security log)"""

    timestamp: datetime
    success: bool
    home_address: str
    access_type: str  # "owner" or "guest"
    guest_pass_id: Optional[str]
    action: str
    value: float
    auth_data: Dict = None

    def display(self):
        print(f"\n### Access Attempt @ {self.timestamp}")
        print(f"* Status: {'✅ Granted' if self.success else '❌ Denied'}")
        print(f"* Home Address: `{self.home_address}`")
        print(f"* Access Type: {self.access_type.upper()}")

        if self.access_type == "owner":
            print("* 🔑 Using Owner's Key (Passkey)")
            if self.auth_data:
                print(f"* Verification: {self.auth_data.get('status', 'Unknown')}")
        else:
            print("* 🎟️ Using Guest Pass (Session Key)")
            print(f"* Guest Pass ID: `{self.guest_pass_id}`")
            if self.auth_data and "checks" in self.auth_data:
                print("* Access Checks:")
                for check, status in self.auth_data["checks"].items():
                    print(f"  * {check}: {status}")

        print(f"* Action: {self.action}")
        print(f"* Value: {self.value} ETH")


def main():
    """
    Demonstrate the house access model of zkSync Era SSO:
    1. Set up new home with owner's key
    2. Use owner's key for high-value transaction
    3. Create guest pass for limited access
    4. Show different access patterns
    """
    security_system = HomeSecuritySystem()

    # Register new home with owner's key
    print("\n## 🏠 New Smart Wallet Home Registration")
    home = security_system.register_new_home("https://example.com")
    print(f"* Home Address: `{home.address}`")
    print(f"* Owner's Key ID: `{home.owner_key.credential_id}`")
    print(f"* Security System: {home.owner_key.storage.value}")
    print(f"* Biometric Lock: {'✅' if home.owner_key.requires_biometric else '❌'}")

    # Owner using their key for high-value transaction
    print("\n## 🔑 Owner's Key Usage (High-Value Operation)")
    owner_access = AccessAttempt(
        timestamp=datetime.now(),
        success=True,
        home_address=home.address,
        access_type="owner",
        guest_pass_id=None,
        action="transfer",
        value=5.0,  # 5 ETH (high value)
        auth_data={
            "status": "✅ Verified with Biometrics",
            "operation": ImportantOperation.HIGH_VALUE_TX.value,
        },
    )
    owner_access.display()

    # Create guest pass for limited access
    print("\n## 🎟️ Creating Guest Pass")
    guest_config = GuestPassConfig(
        expiry=timedelta(days=1),
        allowed_rooms=[f"0x{secrets.token_hex(20)}"],  # Specific contract
        allowed_actions=["transfer", "stake"],
        max_daily_spend=0.5,  # Max 0.5 ETH per day
        max_per_action=0.1,  # Max 0.1 ETH per action
        max_gas_per_action=0.01,  # Max 0.01 ETH gas per action
    )

    guest_pass_id = security_system.issue_guest_pass(home.address, guest_config)
    print(f"* Guest Pass ID: `{guest_pass_id}`")
    print("* Access Limits:")
    print(f"  * Time Limit: {guest_config.expiry}")
    print(f"  * Max Per Action: {guest_config.max_per_action} ETH")
    print(f"  * Daily Limit: {guest_config.max_daily_spend} ETH")

    # Guest using their pass for small transaction
    print("\n## 🎟️ Guest Pass Usage")
    guest_access = AccessAttempt(
        timestamp=datetime.now(),
        success=True,
        home_address=home.address,
        access_type="guest",
        guest_pass_id=guest_pass_id,
        action="transfer",
        value=0.05,  # 0.05 ETH (small value)
        auth_data={
            "status": "✅ Valid Guest Pass",
            "checks": {
                "Time": "✅ Pass Still Valid",
                "Value": "✅ Within Limits",
                "Action": "✅ Allowed",
                "Daily Usage": "✅ Within Daily Limit",
            },
        },
    )
    guest_access.display()


if __name__ == "__main__":
    main()
