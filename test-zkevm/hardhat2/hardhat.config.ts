import { HardhatUserConfig } from "hardhat/config";

import "@nomicfoundation/hardhat-toolbox";
import "@matterlabs/hardhat-zksync";
// import "@matterlabs/hardhat-zksync-node";
// import "@matterlabs/hardhat-zksync-deploy";
// import "@matterlabs/hardhat-zksync-solc";
// import "@matterlabs/hardhat-zksync-verify";

// Import dotenv to read .env file
import * as dotenv from "dotenv";
// Add this import
import { ethers } from "ethers";

dotenv.config();

const cronos_zkevm_testnet_apikey: string = <string>(
    process.env.CRONOS_ZKEVM_DEVELOPER_PORTAL_TESTNET_API_KEY
);
const cronos_zkevm_mainnet_apikey: string = <string>(
    process.env.CRONOS_ZKEVM_DEVELOPER_PORTAL_MAINNET_API_KEY
);

const config: HardhatUserConfig = {
    defaultNetwork: "cronosZkEvmTestnet",
    networks: {
        hardhat: {
            zksync: false,
        },
        cronosZkEvmTestnet: {
            url: "https://testnet.zkevm.cronos.org",
            ethNetwork: process.env.ETHEREUM_SEPOLIA_URL,
            zksync: true,
            verifyURL:
                "https://explorer-api.testnet.zkevm.cronos.org/api/v1/contract/verify/hardhat?apikey=" +
                cronos_zkevm_testnet_apikey,
            accounts: process.env.MY_MNEMONICS ? {
                    mnemonic: process.env.MY_MNEMONICS,
                    path: "m/44'/60'/0'/0",
                    initialIndex: 0,
                    count: 20,
                    passphrase: "",
            } : undefined,
        },
        cronosZkEvmMainnet: {
            url: "https://mainnet.zkevm.cronos.org",
            ethNetwork: process.env.ETHEREUM_MAINNET_URL,
            zksync: true,
            verifyURL:
                "https://explorer-api.zkevm.cronos.org/api/v1/contract/verify/hardhat?apikey=" +
                cronos_zkevm_mainnet_apikey,
        },
        cronosZkEvmMainnetWithApiKey: {
            url: "https://xxxx.zkevm.cronos.org",
            ethNetwork: process.env.ETHEREUM_MAINNET_URL,
            zksync: true,
            verifyURL:
                "https://explorer-api.zkevm.cronos.org/api/v1/contract/verify/hardhat?apikey=" +
                cronos_zkevm_mainnet_apikey,
            accounts: process.env.MY_MNEMONICS ? {
                mnemonic: process.env.MY_MNEMONICS,
                path: "m/44'/60'/0'/0",
                initialIndex: 0,
                count: 20,
                passphrase: "",
            } : undefined,
            httpHeaders: {
                "X-API-KEY": process.env.X_API_KEY || "",
            },
        },
        zkSyncSepoliaTestnet: {
            // If you want to try your deployment on ZKsync Era testnet for reference
            url: "https://sepolia.era.zksync.dev",
            ethNetwork: "sepolia", // or a Sepolia RPC endpoint from Infura/Alchemy/Chainstack etc.
            zksync: true,
            verifyURL:
                "https://explorer.sepolia.era.zksync.dev/contract_verification",
        },
        zkSyncMainnet: {
            // If you want to try your deployment on ZKsync Era mainnet for reference
            url: "https://mainnet.era.zksync.io",
            ethNetwork: "mainnet", // or a Ethereum RPC endpoint from Infura/Alchemy/Chainstack etc.
            zksync: true,
            verifyURL:
                "https://zksync2-mainnet-explorer.zksync.io/contract_verification",
        },
    },
    zksolc: {
        // For Cronos zkEVM, currently only supports zksolc version up to 1.5.2 for contract verification
        version: "1.4.1",
        settings: {
            // find all available options in the official documentation
            // https://era.zksync.io/docs/tools/hardhat/hardhat-zksync-solc.html#configuration
            metadata: {
                // Set bytecodeHash to "none" so that the bytecode is the same at every compilation of a given code base
                bytecodeHash: "none",
            },
        },
    },
    solidity: {
        // For Cronos zkEVM, currently only supports solidity version up to 0.8.26 for contract verification
        version: "0.8.24",
        settings: {
            optimizer: {
                enabled: true,
                runs: 200,
            },
            viaIR: true,
        },
    },
};

export default config;
