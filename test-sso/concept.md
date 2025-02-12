# ZkSync SSO Wallet Architecture

## Key Types Overview

### Regular ECDSA vs WebAuthn Passkey

#### Regular ECDSA (Not Allowed)
```typescript
const regularKey = {
    privateKey: "can be exported",
    publicKey: "derived from private",
    storage: "in memory/file"
};
```

#### WebAuthn Passkey (Required)
```typescript
const passkey = {
    privateKey: "never exportable",
    publicKey: "from hardware",
    storage: "secure hardware only",
    requires: "biometric/PIN verification"
};
```

### WebAuthn Security Features

**Enforced Security:**
- Private key never leaves hardware
- Requires user verification
- Hardware attestation
- Protected against extraction

**Advantages over Regular ECDSA:**
- Hardware protection
- Biometric requirement
- Attestation
- Export protection

## Implementation Details

### 1. Core Security Model

```typescript
// Must use hardware-backed passkey
const ssoWallet = {
    auth: "passkeys",          // Hardware required
    method: "WebAuthn",        // Web Authentication standard
    requires: "secure hardware" // TPM/Secure Enclave
};
```

### 2. WebAuthn Registration Process

```typescript
const credential = await navigator.credentials.create({
    publicKey: {
        challenge: new Uint8Array(32),
        rp: { name: "ZkSync SSO" },
        user: {
            id: new Uint8Array(32),
            name: "user@example.com"
        },
        pubKeyCredParams: [
            { type: "public-key", alg: -7 }  // ES256 (P-256)
        ],
        authenticatorSelection: {
            authenticatorAttachment: "platform",  // Must use hardware
            requireResidentKey: true,            // Must be resident key
            userVerification: "required"         // Must verify user (biometric/PIN)
        }
    }
});
```

## Storage Options

### Passkey Storage Locations

```
├── 1Password (Password Manager)
├── Platform Authenticator
│   ├── iOS/MacOS Keychain
│   ├── Windows Hello
│   └── Android Keystore
└── Security Keys (Yubikey etc)
```

### 1Password Benefits
- 🔄 Sync across devices
- 🔒 Encrypted storage
- 📱 Cross-platform support
- 🔑 Backup/recovery options

## Wallet Architecture

### 1. Hardware Passkey (1Password)

```typescript
// Never changes - permanent identity
const passkey = {
    type: "WebAuthn",
    storage: "1Password",
    state: "permanent",      // Never changes
    controls: "wallet identity" // Derives wallet address
};
```

### 2. Session Keys

```typescript
// Temporary, can create multiple
const sessionKey = {
    type: "ECDSA",
    state: "temporary",     // Can change
    expiry: "24h",         // Time limited
    authorized: "by passkey", // Need passkey to create
    limits: {
        maxValue: "0.1 ETH",
        allowedCalls: [...]
    }
};
```

### Wallet Structure
```
SSO Wallet
├── Passkey (1Password) 
│   └── Never changes (permanent identity)
│
└── Session Keys
    ├── Session Key 1 (temporary)
    ├── Session Key 2 (temporary)
    └── Can create/revoke anytime
```

## Transaction Signing

### 1. Passkey Signing (High-value transactions)

```typescript
// MUST use passkey for:
await ssoWallet.signWithPasskey({
    to: recipient,
    value: parseEther("10"),  // Large amount
    // Requires 1Password/hardware prompt
});
// Proves wallet ownership
// messageHash + signature => wallet address
```

### 2. Session Key Signing (Frequent/Limited transactions)

```typescript
// CAN use session key for:
await sessionWallet.sign({
    to: GAME_CONTRACT,
    value: parseEther("0.01"),  // Small amount
    // No hardware prompt
    // Must be within session limits
});
// Authorized operator
// Has limits & expiry
// No hardware prompt needed
```

### Usage Scenarios

**High Security/Value → Use Passkey (1Password)**
- Large transfers
- Security changes
- Session management

**Frequent/Limited → Use Session Key**
- DApp interactions
- Gaming moves
- Small transfers

## Technical Implementation

### 1. Passkey Address Derivation

```typescript
// Wallet address comes from passkey
const walletAddress = CREATE2.computeAddress(
    keccak256(passKeyPubKey),  // Salt from passkey
    IMPLEMENTATION_CODE
);

// Verification proves ownership of wallet
const isValid = verifyWebAuthnSignature(
    messageHash,
    signature,
    passKeyPubKey
) && recoveredAddress == walletAddress;
```

### 2. Session Key Verification

```typescript
// Session key signature recovers to session key address
const sessionKeyAddress = ecrecover(
    messageHash,
    signature
);  // NOT the wallet address!

// Must check if this session key is authorized
const isValid = sessionValidator.isAuthorized(
    walletAddress,    // Original wallet (from passkey)
    sessionKeyAddress // Recovered session key
);
```

## Access Model Analogy

```
🏠 House Access:
├── Owner's Key (Passkey)
│   └── Proves ownership
│   └── Required for important changes
│
└── Guest Pass (Session Key)
    └── Limited access
    └── Convenient for frequent use
    └── Pre-authorized by owner
```








