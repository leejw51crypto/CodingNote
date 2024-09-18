from solcx import compile_standard, install_solc
import json

def compile_contract(contract_file):
    with open(contract_file, 'r') as file:
        contract_source = file.read()

    compiled_sol = compile_standard({
        "language": "Solidity",
        "sources": {
            "HelloWorld.sol": {
                "content": contract_source
            }
        },
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                }
            }
        }
    }, solc_version="0.8.0")

    with open('compiled_contract.json', 'w') as file:
        json.dump(compiled_sol, file)

    return compiled_sol

if __name__ == "__main__":
    compile_contract('contracts/HelloWorld.sol')
    print("Contract compiled successfully. Output saved to compiled_contract.json")
