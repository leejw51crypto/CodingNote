# Rust TSS Wallet

⚠️ **SECURITY WARNING** ⚠️
This is an educational/demonstration project implementing a Threshold Signature Scheme (TSS) for Ethereum. It has not undergone security audits and SHOULD NOT be used in production environments without thorough review and testing. Use at your own risk.

A Threshold Signature Scheme (TSS) wallet implementation in Rust for Ethereum transactions.

## Features

- Shamir's Secret Sharing for private key splitting
- Threshold signature scheme (t-of-n)
- Ethereum address derivation
- Secure key generation and management
- Support for Ethereum transactions (in progress)

## Prerequisites

- Rust 1.70 or higher
- Cargo package manager

## Installation

1. Clone the repository
2. Install dependencies:
```bash
cargo build
```

## Environment Setup

Create a `.env` file in the project root with the following variables:

```env
MY_TO_ADDRESS=0x...  # Ethereum address to send to
MY_FULL_PRIVATEKEY=0x...  # Your Ethereum private key
```

## Usage

Run the project:

```bash
cargo run
```

This will:
1. Load your private key
2. Split it into shares using TSS
3. Generate the corresponding Ethereum address
4. Display the shares information

## Security Notes

- Never share your private key
- Keep the shares secure and distributed
- The threshold parameter determines how many shares are needed to sign transactions
- This is a demonstration project and should be thoroughly audited before production use

## Testing

Run the tests:

```bash
cargo test
```

## License

MIT 