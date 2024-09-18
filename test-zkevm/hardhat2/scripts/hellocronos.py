import os
import json
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account
import asyncio

load_dotenv()

async def create_contract(network_config):
    print("Running deploy script for the Hello contract")

    w3 = Web3(Web3.HTTPProvider(network_config['url']))

    # Load the artifact of the contract you want to deploy
    with open('artifacts-zk/contracts/HelloWorld.sol/HelloWorld.json', 'r') as file:
        contract_json = json.load(file)
    
    abi = contract_json['abi']
    bytecode = contract_json['bytecode']

    # Deploy the contract
    HelloWorld = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Get the latest nonce
    nonce = w3.eth.get_transaction_count(network_config['from'])
    print(f"Current nonce: {nonce}")

    # Construct transaction
    transaction = {
        'nonce': nonce,
        'gasPrice': w3.eth.gas_price,
        'gas': 3000000,
        'data': bytecode,
        'chainId': w3.eth.chain_id,
        'from': network_config['from'],
    }

    print(f"Transaction details before signing: {transaction}")

    # Sign the transaction
    signed_txn = w3.eth.account.sign_transaction(transaction, network_config['private_key'])
    
    try:
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        print(f"Transaction hash: {tx_hash.hex()}")
    except Exception as e:
        print(f"Error sending transaction: {str(e)}")
        #print(f"Transaction details: {transaction}")
        return
    
    # Wait for the transaction to be mined
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    contract_address = tx_receipt.contractAddress
    print(f"HelloWorld was deployed to {contract_address}")

    # Add a delay to ensure the contract is fully deployed
    await asyncio.sleep(10)

    # Interact with the contract
    hello_world_contract = w3.eth.contract(address=contract_address, abi=abi)

    # Get the initial greeting
    try:
        greeting = hello_world_contract.functions.getGreeting().call()
        print("Initial Greeting:", greeting)
    except Exception as e:
        print(f"Error getting initial greeting: {str(e)}")
        print("Contract ABI:", json.dumps(abi, indent=2))
        print("Contract address:", contract_address)
        print("Transaction receipt:", tx_receipt)
        return

    # Set a new greeting
    new_greeting = "Hello from zkSync!"
    tx = hello_world_contract.functions.setGreeting(new_greeting).build_transaction({
        'from': network_config['from'],
        'nonce': w3.eth.get_transaction_count(network_config['from']),
        'gas': 200000,
        'gasPrice': w3.eth.gas_price,
    })
   
    
    signed_tx = w3.eth.account.sign_transaction(tx, network_config['private_key'])
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Setting new greeting...")

    # Get the updated greeting
    greeting = hello_world_contract.functions.getGreeting().call()
    print("Updated Greeting:", greeting)

Account.enable_unaudited_hdwallet_features()
async def main():
    print("Hello, World!")
    mnemonic = os.getenv('MYMNEMONICS')
    if not mnemonic:
        return {"status": "error", "message": "MYMNEMONICS environment variable not set"}
    
    current_wallet_index = 0
    from_account = Account.from_mnemonic(mnemonic, account_path=f"m/44'/60'/0'/0/{current_wallet_index}")
    
    private_key = from_account._private_key.hex()
    address = from_account.address
    
    print(f"Derived address: {address}")
    
    cronos_rpc_url = os.getenv('MYCRONOSRPC')
    cronos_chain_id = int(os.getenv('MYCRONOSCHAINID'))
    if not cronos_rpc_url or not cronos_chain_id:
        raise ValueError("MYCRONOSRPC or MYCRONOSCHAINID environment variable is not set")

    # Get network configuration
    network_config = {
        'url': cronos_rpc_url,
        'chain_id': cronos_chain_id,
        'from': address,
        'private_key': private_key
    }

    # Call create_contract function
    await create_contract(network_config)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
