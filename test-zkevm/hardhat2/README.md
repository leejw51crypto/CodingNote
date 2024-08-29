# Cronos zkEVM Hardhat Demo Repository

The goal of this repository is to demonstrate basic use of Cronos zkEVM to develop smart contract and implement on-chain transactions.

# Set-up

This repository uses Node 20 with Typescript and the following packages:

-   zksync-ethers (v6.9.0) together with ethers (v6) (
    see https://docs.zksync.io/build/sdks/js/zksync-ethers/getting-started.html)
-   hardhat (see https://docs.zksync.io/build/tooling/hardhat/migrating-to-zksync.html)

It was created using a standard hardhat project (https://hardhat.org/hardhat-runner/docs/guides/project-setup), and then migrated to be compatible with ZK Stack using the instructions described in the ZK Sync documentation ("migration guide").

IMPORTANT: See `package.json` for the full list of dependencies. (In particular, requires @matterlabs/hardhat-zksync-verify for contract verification).

To install the dependencies of this repository, type `npm install`.

The zksync-ethers package is constantly moving and unfortunately, this means that this repository may present dependency conflicts over time. At the time of writing, `npm install` returns some warnings but no errors. Pay attention to the version numbers in `package.json` if you are trying to recreate this project at home.

# Basic blockchain reading and writing operations

You can find the Cronos zkEVM testnet blockchain explorer at: https://explorer.zkevm.cronos.org/testnet/.

The basic reading and writing scripts are in the /scripts folder:

-   s01_read_blockchain.ts: read wallet balances, blocks and transactions.
-   s02_basic_transactions.ts: transfer zkTCRO, deposit zkTCRO from L1 to L2, withdraw zkTCRO from L2 to L1.

# Smart contract development

## Hardhat config

The settings for the Cronos zkEVM testnet network are as follows:

```json lines
{
    "cronosZkEvmTestnet": {
        "url": "https://testnet.zkevm.cronos.org",
        "ethNetwork": "sepolia",
        // or a Sepolia RPC endpoint from Infura/Alchemy/Chainstack etc.
        "zksync": true,
        "verifyURL": "https://explorer-api.testnet.zkevm.cronos.org/api/v1/contract/verify/hardhat?apikey={api_key}"
    }
}
```

In order to obtain an API key for contract verification, please visit the Cronos zkEVM Developer Portal at: [https://developers.zkevm.cronos.org/](https://developers.zkevm.cronos.org/).

Alternatively, you can verify contracts by visiting the user interface at [https://explorer.zkevm.cronos.org/testnet/verifyContract](https://explorer.zkevm.cronos.org/testnet/verifyContract).

## Compilation and deployment

The smart contracts in this repository are written in Solidity and are based on the OpenZeppelin library. Considering
that `@matterlabs/hardhat-zksync-upgradable` does not currently support OpenZeppelin libraries above v4.9.5, we are only
using `@openzeppelin/contracts-upgradeable@4.9.5` and `@openzeppelin/contracts@4.9.5`.

To compile all the contracts in the /contracts directory, run:

```shell
npx hardhat compile --network cronosZkEvmTestnet
```

To deploy and verify the contract, run:

```shell
# Deploy to testnet and verify
npx hardhat deploy-zksync --script deployMyERC20Token.ts --network cronosZkEvmTestnet

# Deploy to mainnet and verify
npx hardhat deploy-zksync --script deployMyERC20Token.ts --network cronosZkEvmMainnet
```

# Interacting with the deployed contract

A basic reading and writing script is included in the /scripts folder:

-   s03_smart_contract_read_and_write.ts: read contract, write contract.

# Going further

Now that you have seen a few working examples of using Cronos zkEVM, you should be able to translate the zkSync
documentation into Cronos zkEVM code.

For the zkSync documentation, refer to: https://docs.zksync.io/build/sdks/js/zksync-ethers/getting-started.html
