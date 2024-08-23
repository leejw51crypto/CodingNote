import { HardhatUserConfig } from 'hardhat/config';
import '@nomicfoundation/hardhat-toolbox';

const config: HardhatUserConfig = {
  solidity: {
    compilers: [
      {
        version: '0.6.6',
      },
      {
        version: '0.6.2',
      },
      {
        version: '0.5.0',
      },
      {
        version: '0.8.24',
      },
    ],
  },
  networks: {
    my: {
      gas: 'auto',
      gasPrice: 'auto',
      gasMultiplier: 1,
      url: process.env.MYCRONOSRPC,
      chainId: process.env.MYCRONOSCHAINID
        ? parseInt(process.env.MYCRONOSCHAINID)
        : 0,
      accounts: {
        mnemonic: process.env.MYMNEMONICS,
        path: "m/44'/60'/0'/0",
        initialIndex: 0,
        count: 5,
        passphrase: ''
      }
    },

    cronosZkEvmTestnet: {
      url: "https://testnet.zkevm.cronos.org",
      chainId: 282,
      accounts: {
        mnemonic: process.env.MYMNEMONICS,
        path: "m/44'/60'/0'/0",
        initialIndex: 0,
        count: 5,
        passphrase: ''
      }
    },

    cronosZkEvmMainnet: {
      url: "https://mainnet.zkevm.cronos.org",
      chainId: 388,
      accounts: {
        mnemonic: process.env.MYMNEMONICS,
        path: "m/44'/60'/0'/0",
        initialIndex: 0,
        count: 5,
        passphrase: ''
      },
      gasPrice: 2500000000000 // 2500 gwei
    },
  },
};

export default config;