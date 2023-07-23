const { ethers } = require('hardhat');
import config from '../hardhat.config';

async function main() {
  const contractAddress = config.networks.my.mycontractusdc;
  const address = config.networks.my.mycronosaddress;

  const contractBalance = await ethers.provider.getBalance(contractAddress);
  const addressBalance = await ethers.provider.getBalance(address);

  const data = {
    'Contract Address': contractAddress,
    'Balance of Contract': ethers.utils.formatEther(contractBalance),
    Address: address,
    'Balance of Address': ethers.utils.formatEther(addressBalance)
  };

  console.table(data);
}

main();
