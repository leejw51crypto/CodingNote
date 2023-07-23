import { HardhatUserConfig } from 'hardhat/config';
import '@nomicfoundation/hardhat-toolbox';

const config: HardhatUserConfig = {
  solidity: '0.6.6',
  networks: {
    my: {
      gas:2862121,
      gasPrice:1943806396054,
      gasMultiplier: 1,
      mycroamount: "1.2",
      mycronosaddress: process.env.MYCRONOSADDRESS,
      mycronosaddress2: process.env.MYCRONOSADDRESS2,
      mycronosaddress3: process.env.MYCRONOSADDRESS3,
      mycontractswap: process.env.MYCONTRACTSWAP,
      mycontractcro: process.env.MYCONTRACTWCRO,
      mycontractvvs: process.env.MYCONTRACTVVS,
      mycontractusdc: process.env.MYCONTRACTUSDC,
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
