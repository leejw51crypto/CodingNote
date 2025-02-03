# MPC Wallet (Threshold Signature Scheme)

This project implements a Multi-Party Computation (MPC) wallet using Threshold Signature Scheme (TSS) for Ethereum-compatible networks, specifically configured for Cronos testnet.

## Features

- Threshold Signature Scheme (TSS) implementation
- Support for splitting and combining private keys
- Ethereum transaction signing with multiple parties
- Configurable threshold and number of parties
- Cronos testnet integration

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd test-mpc-wallet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` file and add your private key:
```
MY_FULL_PRIVATEKEY=your_private_key_here  # Without 0x prefix
```

## Usage

The main script (`c.py`) implements TSS functionality for Ethereum transactions. To run:

```bash
source .env && python c.py
```

The script will:
1. Initialize TSS with specified threshold and number of parties
2. Generate partial signatures from each party
3. Combine signatures to create a valid transaction
4. Send the transaction to the Cronos testnet

## Security Considerations

- Keep your private key secure and never share it
- The `.env` file contains sensitive information - do not commit it to version control
- The threshold value determines how many parties are needed to sign a transaction
- Ensure secure communication channels between parties when sharing partial signatures

## Network Configuration

The project is configured to work with Cronos testnet:
- RPC Endpoint: https://evm-t3.cronos.org/
- Chain ID: 338

## Dependencies

- web3==6.15.1
- eth-account==0.11.0
- cryptography==42.0.5
- python-dotenv==1.0.1

## Testing Partial Signatures

You can test the partial signature functionality using both Python and Rust implementations:

### Python Implementation
```bash
python testjson.py
```
This will run the Python implementation of TSS partial signature generation and verification.

### Rust Implementation
```bash
cd rusttss
cargo run --example tssjsonread
```
This will run the Rust implementation which:
- Reads TSS test data from JSON
- Verifies partial signatures
- Combines signatures
- Validates the final combined signature

Both implementations should produce matching results, verifying the correctness of the TSS implementation across different languages.
