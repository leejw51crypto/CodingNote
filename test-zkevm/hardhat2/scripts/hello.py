import os
import json
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount
from zksync2.module.module_builder import ZkSyncBuilder
from zksync2.signer.eth_signer import PrivateKeyEthSigner
from zksync2.transaction.transaction_builders import TxCreateContract
from zksync2.manage_contracts.contract_encoder_base import ContractEncoder
from zksync2.core.types import EthBlockParams
import binascii
from eth_account.signers.local import LocalAccount

load_dotenv()

async def deploy_contract(zk_web3, account: LocalAccount, abi, bytecode):
    print("Running deploy script for the Hello contract")

    # Convert bytecode to bytes if it's a string
    #if isinstance(bytecode, str):
    #    bytecode = bytes.fromhex(bytecode.strip('0x'))
    if isinstance(bytecode, str):
        bytecode = bytes.fromhex(bytecode)

    # Pad the bytecode if necessary
    if len(bytecode) % 64 == 0:
        bytecode += b'\0'  # Add a single zero byte

    # Get chain id of zkSync network
    chain_id = zk_web3.zksync.chain_id

    # Signer is used to generate signature of provided transaction
    signer = PrivateKeyEthSigner(account, chain_id)

    # Get nonce of ETH address on zkSync network
    nonce = zk_web3.zksync.get_transaction_count(
        account.address, EthBlockParams.PENDING.value
    )

    # Get current gas price in Wei
    gas_price = zk_web3.zksync.gas_price

    
    # Create deployment contract transaction
    create_contract = TxCreateContract(
        web3=zk_web3,
        chain_id=chain_id,
        nonce=nonce,
        from_=account.address,
        gas_limit=0,  # UNKNOWN AT THIS STATE
        gas_price=gas_price,
        bytecode=bytecode,
    )
    
    

    # ZkSync transaction gas estimation
    estimate_gas = zk_web3.zksync.eth_estimate_gas(create_contract.tx)
    print(f"Fee for transaction is: {Web3.from_wei(estimate_gas * gas_price, 'ether')} ETH")

    # Convert transaction to EIP-712 format
    tx_712 = create_contract.tx712(estimate_gas)

    # Sign message
    signed_message = signer.sign_typed_data(tx_712.to_eip712_struct())

    # Encode signed message
    msg = tx_712.encode(signed_message)

    # Deploy contract
    tx_hash = zk_web3.zksync.send_raw_transaction(msg)

    # Wait for deployment contract transaction to be included in a block
    tx_receipt = zk_web3.zksync.wait_for_transaction_receipt(
        tx_hash, timeout=240, poll_latency=0.5
    )

    print(f"Tx status: {tx_receipt['status']}")
    contract_address = tx_receipt["contractAddress"]

    print(f"Deployed contract address: {contract_address}")

    return contract_address

Account.enable_unaudited_hdwallet_features()
async def main():
    print("Hello, World!")
    mnemonic = os.getenv('MYMNEMONICS')
    if not mnemonic:
        return {"status": "error", "message": "MYMNEMONICS environment variable not set"}
    
    current_wallet_index = 0
    from_account: LocalAccount = Account.from_mnemonic(mnemonic, account_path=f"m/44'/60'/0'/0/{current_wallet_index}")
    
    private_key = from_account._private_key.hex()
    address = from_account.address
    
    print(f"Derived address: {address}")
    
    # Get network configuration
    network_config = {
        'url': 'https://testnet.zkevm.cronos.org',
        'from': address,
        'private_key': private_key
    }

    # Load the artifact of the contract you want to deploy
    with open('artifacts-zk/contracts/HelloWorld.sol/HelloWorld.json', 'r') as file:
        contract_json = json.load(file)

    abi = contract_json['abi']
    bytecode = contract_json['bytecode']
    if bytecode.startswith('0x'):
        bytecode = bytecode[2:]

    # Connect to zkSync network
    zk_web3 = ZkSyncBuilder.build(network_config['url'])

    # Get account object by providing from private key
    account: LocalAccount = Account.from_key(private_key)

    # Deploy the contract
    deployed_address = await deploy_contract(zk_web3, account, abi, bytecode)
    print(f"Contract deployed at: {deployed_address}")

    # Create contract instance
    contract = zk_web3.eth.contract(address=deployed_address, abi=abi)

    # Get initial greeting
    initial_greeting = contract.functions.getGreeting().call()
    print(f"Initial greeting: {initial_greeting}")

    # Set new greeting
    new_greeting = "Hello from zkSync!"
    tx = contract.functions.setGreeting(new_greeting).build_transaction({
        'from': address,
        'nonce': zk_web3.eth.get_transaction_count(address),
    })
    signed_tx = zk_web3.eth.account.sign_transaction(tx, private_key)
    tx_hash = zk_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = zk_web3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"New greeting set: {new_greeting}")

    # Get updated greeting
    updated_greeting = contract.functions.getGreeting().call()
    print(f"Updated greeting: {updated_greeting}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
