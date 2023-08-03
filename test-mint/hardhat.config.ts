import { HardhatUserConfig } from 'hardhat/config';
import '@nomicfoundation/hardhat-toolbox';

const config: HardhatUserConfig = {
  solidity: '0.8.2',
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
    }
  }
};

export default config;
