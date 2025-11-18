# Lua-Rust Wallet Bindings Protocol

This document describes the complete Rust binding interface exposed to Lua for the ECDSA wallet system.

## Table of Contents

- [Overview](#overview)
- [Data Types](#data-types)
- [Functions](#functions)
  - [wallet.generate_mnemonic()](#walletgenerate_mnemonic)
  - [wallet.create_wallet()](#walletcreate_wallet)
  - [wallet.generate_addresses()](#walletgenerate_addresses)
  - [wallet.generate_btc_address()](#walletgenerate_btc_address)
  - [wallet.read_hidden()](#walletread_hidden)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

---

## Overview

The Rust backend exposes a module called `wallet` to Lua with five main functions for wallet operations. All cryptographic operations are performed in Rust for security, while Lua handles UI and workflow logic.

**Cryptographic Standards:**
- BIP39: Mnemonic phrase generation
- BIP32: Hierarchical deterministic key derivation
- BIP44: Multi-account hierarchy (Ethereum path: m/44'/60'/0'/0/{index})
- secp256k1: ECDSA curve for key pairs
- Keccak256 (SHA3): Ethereum address generation

---

## Data Types

### WalletHandle (UserData)
Opaque Rust object that stores mnemonic phrase securely. This handle keeps sensitive data in Rust memory rather than exposing it directly to Lua.

**Methods:**
- `handle:get_mnemonic()` - Returns the mnemonic string (use sparingly, for display only)
- `handle:word_count()` - Returns the number of words in the mnemonic

### Lua String
Standard Lua string type used for text data.

### Lua Number
Standard Lua number type (64-bit floating point). Used for indices and counts.

### Lua Table
Lua table (dictionary/array) used for structured data return values.

### Optional Parameters
Parameters marked as optional will use default values if `nil` is passed or parameter is omitted.

---

## Functions

### wallet.generate_mnemonic()

Generates a new BIP39 mnemonic phrase using cryptographically secure random entropy. Returns an opaque WalletHandle that stores the mnemonic securely in Rust.

**Signature:**
```lua
function wallet.generate_mnemonic() -> WalletHandle
```

**Parameters:**
- None

**Returns:**
- `WalletHandle`: Opaque handle to the generated mnemonic (stored in Rust)

**Example:**
```lua
local handle = wallet.generate_mnemonic()
print(handle:get_mnemonic())  -- Display the mnemonic (explicit call)
-- Output: "abandon ability able about above absent absorb abstract absurd abuse access accident..."

-- Use the handle directly with other wallet functions
local my_wallet = wallet.create_wallet(handle, 0)
```

**Error Conditions:**
- Returns error if random number generator fails
- Returns error if entropy generation fails

**Implementation Details:**
- Uses 32 bytes (256 bits) of entropy
- Generates exactly 24 words
- Words are from BIP39 English wordlist
- Thread-safe random number generation
- Mnemonic stored in Rust memory, not Lua

**Security Note:**
The mnemonic is stored in Rust memory, not in Lua tables. This provides better security for sensitive data. Use `handle:get_mnemonic()` only when you need to display or export the mnemonic.

---

### wallet.import_mnemonic()

Imports an existing BIP39 mnemonic phrase and creates a WalletHandle for it.

**Signature:**
```lua
function wallet.import_mnemonic(mnemonic: string) -> WalletHandle
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mnemonic` | string | Yes | - | BIP39 mnemonic phrase (12-24 words) |

**Returns:**
- `WalletHandle`: Opaque handle to the imported mnemonic

**Example:**
```lua
local mnemonic_str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
local handle = wallet.import_mnemonic(mnemonic_str)

-- Now use the handle with other functions
local wallet = wallet.create_wallet(handle, 0)
print(wallet.address)
```

**Error Conditions:**
- Invalid mnemonic phrase (wrong words, invalid checksum)
- Wrong number of words

**Use Cases:**
- Recovering existing wallets from backup
- Importing mnemonics from user input
- Using known test mnemonics

---

### wallet.create_wallet()

Creates a wallet from a WalletHandle at a specific derivation index.

**Signature:**
```lua
function wallet.create_wallet(handle: WalletHandle, wallet_index?: number) -> table
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `handle` | WalletHandle | Yes | - | WalletHandle from generate_mnemonic() or import_mnemonic() |
| `wallet_index` | number | No | 0 | Derivation index (non-negative integer) |

**Returns:**
A Lua table with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `wallet_index` | number | The derivation index used |
| `private_key` | string | Hex-encoded private key (64 characters, 32 bytes) |
| `public_key` | string | Hex-encoded uncompressed public key (130 chars, 65 bytes with 0x04 prefix) |
| `address` | string | Ethereum-style address (42 characters with 0x prefix) |

**Note:** The mnemonic is NOT included in the returned table for security. It remains stored in the WalletHandle.

**Derivation Path:**
`m/44'/60'/0'/0/{wallet_index}`
- 44' = BIP44
- 60' = Ethereum coin type
- 0' = Account 0
- 0 = External chain (receiving addresses)
- {wallet_index} = Address index

**Example:**
```lua
-- Import mnemonic and create wallet at default index (0)
local handle = wallet.import_mnemonic("abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about")
local wallet = wallet.create_wallet(handle)
print(wallet.address)          -- 0x9858effd232b4033e47d90003d41ec34ecaeda94
print(wallet.wallet_index)     -- 0
print(wallet.private_key)      -- 64 hex characters
print(wallet.public_key)       -- 130 hex characters

-- Create wallet at index 5
local wallet5 = wallet.create_wallet(handle, 5)
print(wallet5.wallet_index)    -- 5
```

**Error Conditions:**
- Invalid WalletHandle
- Invalid wallet_index (negative numbers)
- Key derivation failure
- Invalid derivation path

**Notes:**
- Same mnemonic + same index = same address (deterministic)
- Different indices produce different addresses
- All addresses can be derived from single mnemonic

---

### wallet.generate_addresses()

Efficiently generates multiple addresses from a WalletHandle in one call.

**Signature:**
```lua
function wallet.generate_addresses(handle: WalletHandle, count?: number) -> table[]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `handle` | WalletHandle | Yes | - | WalletHandle from generate_mnemonic() or import_mnemonic() |
| `count` | number | No | 5 | Number of addresses to generate (positive integer) |

**Returns:**
An array table (1-indexed) where each element contains:

| Field | Type | Description |
|-------|------|-------------|
| `index` | number | Derivation index (0-based) |
| `address` | string | Ethereum address (0x...) |
| `private_key` | string | Hex-encoded private key |
| `public_key` | string | Hex-encoded public key |

**Example:**
```lua
-- Generate 5 addresses (default)
local handle = wallet.generate_mnemonic()
local addresses = wallet.generate_addresses(handle)
for i = 1, #addresses do
    local addr = addresses[i]
    print(string.format("[%d] %s", addr.index, addr.address))
end
-- Output:
-- [0] 0x78d137ceee0e2b28c3cb5bd10ec1b0d17d394645
-- [1] 0x0e28baff0e045c6243ac25f5f3453ebcac604826
-- [2] 0x...
-- [3] 0x...
-- [4] 0x...

-- Generate 10 addresses
local more = wallet.generate_addresses(handle, 10)
print(#more)  -- 10

-- Access specific address
print(more[1].address)    -- First address (index 0)
print(more[1].index)      -- 0
print(more[5].index)      -- 4
```

**Error Conditions:**
- Invalid WalletHandle
- Invalid count (negative or zero)
- Key derivation failure for any index

**Performance Notes:**
- More efficient than calling `create_wallet()` multiple times
- All addresses derived in single Rust function call
- Recommended for bulk address generation

**Array Indexing:**
```lua
-- Lua arrays are 1-indexed, but derivation indices are 0-based
addresses[1].index == 0  -- First address is at derivation index 0
addresses[2].index == 1  -- Second address is at derivation index 1
addresses[n].index == n-1
```

---

### wallet.generate_btc_address()

Generates a Bitcoin-style address from a public key (simplified implementation).

**Signature:**
```lua
function wallet.generate_btc_address(public_key_hex: string) -> string
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `public_key_hex` | string | Yes | - | Hex-encoded public key (130 or 66 chars) |

**Returns:**
- `string`: Bitcoin-style address (simplified format with hex encoding)

**Example:**
```lua
local wallet = wallet.create_wallet(mnemonic, 0)
local btc_addr = wallet.generate_btc_address(wallet.public_key)
print(btc_addr)
-- Output: 1a3b5c7d9e... (simplified format)
```

**Error Conditions:**
- Invalid hex string
- Invalid public key format
- Public key wrong length

**Notes:**
- Currently uses simplified Bitcoin address format
- Not production-ready for actual Bitcoin transactions
- Demonstrates address derivation concept
- Does not implement full Base58Check encoding

---

### wallet.read_hidden()

Reads hidden input from the user (like passwords or mnemonic phrases) without echoing to the terminal.

**Signature:**
```lua
function wallet.read_hidden(prompt: string) -> string
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | The prompt message to display to the user |

**Returns:**
- `string`: The user's input (not echoed to terminal)

**Example:**
```lua
-- Read mnemonic securely
local mnemonic = wallet.read_hidden("Enter your mnemonic phrase: ")
print("Mnemonic received (length: " .. #mnemonic .. ")")

-- Create wallet with the hidden mnemonic
local my_wallet = wallet.create_wallet(mnemonic, 0)
print("Address: " .. my_wallet.address)
```

**Error Conditions:**
- Failed to read from stdin
- Terminal input not available

**Security Notes:**
- Input is NOT echoed to the terminal (password-style input)
- More secure than regular io.read() for sensitive data
- Works on Unix/Linux/macOS and Windows
- Recommended for reading mnemonics and passwords

**Use Cases:**
```lua
-- Reading mnemonic securely
local mnemonic = wallet.read_hidden("Enter mnemonic: ")

-- Reading any sensitive input
local api_key = wallet.read_hidden("Enter API key: ")
```

---

## Error Handling

All Rust functions can return errors to Lua. Errors should be handled using `pcall`:

```lua
-- Safe way to call Rust functions
local success, result = pcall(function()
    return wallet.create_wallet(mnemonic, index)
end)

if success then
    -- result contains the wallet table
    print("Success:", result.address)
else
    -- result contains error message
    print("Error:", result)
end
```

**Common Error Messages:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| "Invalid mnemonic: ..." | Wrong mnemonic phrase | Check word spelling and count |
| "Failed to generate mnemonic: ..." | RNG failure | Retry operation |
| "Failed to derive key: ..." | Derivation error | Check index validity |
| "Invalid hex: ..." | Bad hex string | Verify hex format |
| "Invalid public key: ..." | Malformed key | Check key format and length |

---

## Usage Examples

### Example 1: Basic Wallet Creation

```lua
-- Generate and create wallet
local handle = wallet.generate_mnemonic()
print("Mnemonic:", handle:get_mnemonic())

local my_wallet = wallet.create_wallet(handle)
print("Address:", my_wallet.address)
print("Index:", my_wallet.wallet_index)
```

### Example 2: Multiple Addresses from One Mnemonic

```lua
-- Create addresses at different indices
local handle = wallet.generate_mnemonic()

for i = 0, 4 do
    local w = wallet.create_wallet(handle, i)
    print(string.format("Index %d: %s", i, w.address))
end
```

### Example 3: Bulk Address Generation

```lua
-- Generate 100 addresses efficiently
local handle = wallet.generate_mnemonic()
local addresses = wallet.generate_addresses(handle, 100)

-- Find addresses by criteria
for i = 1, #addresses do
    local addr = addresses[i]
    if string.sub(addr.address, 3, 4) == "00" then
        print("Vanity address found:", addr.address)
    end
end
```

### Example 4: Error Handling

```lua
local function safe_create_wallet(handle, index)
    local success, wallet = pcall(function()
        return wallet.create_wallet(handle, index)
    end)

    if not success then
        print("Error creating wallet:", wallet)
        return nil
    end

    return wallet
end

-- Import mnemonic from user
local mnemonic_str = wallet.read_hidden("Enter mnemonic: ")
local success, handle = pcall(function()
    return wallet.import_mnemonic(mnemonic_str)
end)

if success then
    local my_wallet = safe_create_wallet(handle, 0)
    if my_wallet then
        print("Wallet created:", my_wallet.address)
    end
else
    print("Invalid mnemonic:", handle)
end
```

### Example 5: Deterministic Verification

```lua
-- Verify deterministic generation
local handle = wallet.generate_mnemonic()

local wallet1 = wallet.create_wallet(handle, 10)
local wallet2 = wallet.create_wallet(handle, 10)

assert(wallet1.address == wallet2.address, "Addresses should match!")
assert(wallet1.private_key == wallet2.private_key, "Keys should match!")
```

### Example 6: Address Scanner

```lua
-- Scan for specific address patterns
local function find_vanity_address(handle, pattern, max_attempts)
    for i = 0, max_attempts - 1 do
        local w = wallet.create_wallet(handle, i)
        if string.match(w.address, pattern) then
            return w
        end
    end
    return nil
end

local handle = wallet.generate_mnemonic()
local vanity = find_vanity_address(handle, "^0x00", 1000)
if vanity then
    print("Found:", vanity.address, "at index", vanity.wallet_index)
end
```

---

## Best Practices

### 1. Security

```lua
-- ✅ GOOD: Use secure mnemonic generation with WalletHandle
local handle = wallet.generate_mnemonic()

-- ✅ GOOD: Only display mnemonic when needed
print("Backup your mnemonic:", handle:get_mnemonic())

-- ❌ BAD: Don't pass mnemonic strings around unnecessarily
-- local mnemonic_str = handle:get_mnemonic()
-- Pass handle instead!
```

### 2. Error Handling

```lua
-- ✅ GOOD: Always use pcall for user input
local success, handle = pcall(function()
    return wallet.import_mnemonic(user_mnemonic_str)
end)

if success then
    local wallet = wallet.create_wallet(handle, user_index)
end

-- ❌ BAD: Don't call directly with user input without validation
-- local handle = wallet.import_mnemonic(user_mnemonic_str)
```

### 3. Performance

```lua
-- ✅ GOOD: Use generate_addresses for bulk generation
local addresses = wallet.generate_addresses(handle, 100)

-- ❌ BAD: Don't loop with create_wallet
-- for i = 0, 99 do
--     local w = wallet.create_wallet(handle, i)
-- end
```

### 4. Index Management

```lua
-- ✅ GOOD: Track used indices
local used_indices = {}
for i = 0, 10 do
    used_indices[i] = true
end

-- ❌ BAD: Don't reuse indices without reason
-- Same index = same address (address reuse privacy concern)
```

### 5. Memory Management

```lua
-- ✅ GOOD: Generate addresses in batches
local batch_size = 100
for batch = 0, 10 do
    local start_idx = batch * batch_size
    -- Process batch...
end

-- ❌ BAD: Don't generate all at once if huge
-- local all = wallet.generate_addresses(mnemonic, 1000000)
```

### 6. Validation

```lua
-- ✅ GOOD: Validate parameters
local function create_safe(handle, index)
    if type(handle) ~= "userdata" then
        return nil, "handle must be WalletHandle"
    end
    if index and (type(index) ~= "number" or index < 0) then
        return nil, "index must be non-negative number"
    end
    return wallet.create_wallet(handle, index)
end
```

---

## Type Reference

### WalletInfo Table

```lua
{
    wallet_index = 0,                         -- number (integer)
    private_key = "abc123...",                -- string (64 hex chars)
    public_key = "04def456...",               -- string (130 hex chars)
    address = "0x789abc..."                   -- string (42 chars)
}

-- Note: mnemonic is NO LONGER in this table
-- It's stored in the WalletHandle instead
```

### AddressInfo Table

```lua
{
    index = 0,                                -- number (integer)
    address = "0x123...",                     -- string (42 chars)
    private_key = "abc...",                   -- string (64 hex chars)
    public_key = "04def..."                   -- string (130 hex chars)
}
```

---

## Constants

### Default Values
- Default wallet index: `0`
- Default address count: `5`
- Mnemonic word count: `24`
- Entropy bits: `256`

### String Formats
- Private key: 64 hexadecimal characters (lowercase)
- Public key: 130 hexadecimal characters (lowercase, with "04" prefix)
- Ethereum address: 42 characters (with "0x" prefix, lowercase)

### Derivation Path
- Standard path: `m/44'/60'/0'/0/{index}`
- Coin type: 60 (Ethereum)
- Account: 0
- Change: 0 (external/receiving)

---

## Version Information

**Protocol Version:** 1.0
**Rust Backend Version:** 0.1.0
**Lua Compatibility:** 5.4
**Last Updated:** 2025-11-11

---

## Additional Notes

1. All cryptographic operations happen in Rust for security
2. Lua is responsible for UI, workflow, and business logic
3. Keys are always returned as hexadecimal strings (lowercase)
4. Addresses use Ethereum format (checksumming not implemented)
5. All indices are 0-based for derivation, but Lua arrays are 1-indexed
6. Thread safety: Functions are thread-safe on Rust side
7. No state is maintained between calls (stateless API)

---

## Support

For issues or questions about the Rust-Lua bindings:
- Check error messages carefully
- Use `pcall` for all user-facing operations
- Verify parameter types before calling
- See usage examples in this document
- Review `test_wallet.lua` for comprehensive examples
