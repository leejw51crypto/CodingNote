import os
from web3 import Web3
from eth_account import Account
import json
from dotenv import load_dotenv
from decimal import Decimal

# Load environment variables
load_dotenv()

# Configure web3 with Cronos testnet
RPC_ENDPOINT = "https://evm-t3.cronos.org/"
CHAIN_ID = 338

# Get credentials from environment
private_key = os.getenv("MY_FULL_PRIVATEKEY")
if private_key.startswith("0x"):
    private_key = private_key[2:]  # Remove '0x' prefix if present
to_address = os.getenv("MY_TO_ADDRESS")

# Connect to network
w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
if not w3.is_connected():
    raise Exception("Failed to connect to the network")

# Create account from private key
account = Account.from_key(private_key)
print(f"From address: {account.address}")
print(f"To address: {to_address}")

# Show initial balances
sender_balance_before = w3.eth.get_balance(account.address)
receiver_balance_before = w3.eth.get_balance(to_address)
print("\nInitial balances:")
print(f"Sender balance: {w3.from_wei(sender_balance_before, 'ether')} TCRO")
print(f"Receiver balance: {w3.from_wei(receiver_balance_before, 'ether')} TCRO")

# Get nonce
nonce = w3.eth.get_transaction_count(account.address)

# Amount to send (0.1 TCRO)
amount_to_send = w3.to_wei(0.1, 'ether')

# Prepare transaction
transaction = {
    'nonce': nonce,
    'gasPrice': w3.eth.gas_price,
    'gas': 21000,  # Standard gas limit for ETH transfer
    'to': to_address,
    'value': amount_to_send,
    'chainId': CHAIN_ID
}

# Estimate gas (optional but recommended)
try:
    estimated_gas = w3.eth.estimate_gas(transaction)
    transaction['gas'] = estimated_gas
except Exception as e:
    print(f"Warning: Could not estimate gas: {e}")

# Sign transaction
signed_txn = w3.eth.account.sign_transaction(transaction, private_key)

# Send transaction
try:
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    print(f"\nTransaction sent! Hash: {tx_hash.hex()}")
    
    # Wait for transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Transaction confirmed in block {tx_receipt['blockNumber']}")
    gas_used = tx_receipt['gasUsed']
    gas_price = transaction['gasPrice']
    gas_cost = gas_used * gas_price
    print(f"Gas used: {gas_used} (Cost: {w3.from_wei(gas_cost, 'ether')} TCRO)")
    
    # Show final balances
    sender_balance_after = w3.eth.get_balance(account.address)
    receiver_balance_after = w3.eth.get_balance(to_address)
    print("\nFinal balances:")
    print(f"Sender balance: {w3.from_wei(sender_balance_after, 'ether')} TCRO")
    print(f"Receiver balance: {w3.from_wei(receiver_balance_after, 'ether')} TCRO")
    
    # Calculate and show changes
    print(f"Gas cost: {w3.from_wei(gas_cost, 'ether')} TCRO")
    
except Exception as e:
    print(f"Error sending transaction: {e}") 