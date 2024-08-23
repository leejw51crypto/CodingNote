import { Provider } from "zksync-ethers";
import { ethers } from 'hardhat';
import { showSigners, ask } from './util';

async function getLatestBlock(provider: Provider) {
  try {
    const latestBlock = await provider.getBlock("latest");
    console.log("Latest block:", latestBlock);
    return latestBlock;
  } catch (error) {
    console.error("Error fetching latest block:", error);
    throw error;
  }
}

async function sendAmount() {
  const signers = await ethers.getSigners();
  await showSigners(signers);

  let from = signers[0];
  let to = signers[1];

  const amount = ethers.parseEther("0.1"); // 0.001 ETH
  const data = ethers.encodeBytes32String("Hello, Cronos zkEVM!"); // 32 bytes of string data

  console.log(`Sending ${ethers.formatEther(amount)} ETH from ${from.address} to ${to.address}`);
  console.log(`Including data: ${ethers.decodeBytes32String(data)}`);

  try {
    const tx = await from.sendTransaction({
      to: to.address,
      value: amount,
      data: data
    });

    console.log(`Transaction sent: ${tx.hash}`);
    await tx.wait();
    console.log(`Transaction confirmed`);

    const fromBalance = await ethers.provider.getBalance(from.address);
    const toBalance = await ethers.provider.getBalance(to.address);

    console.log(`New balance of sender: ${ethers.formatEther(fromBalance)} ETH`);
    console.log(`New balance of recipient: ${ethers.formatEther(toBalance)} ETH`);
  } catch (error) {
    console.error("Error sending transaction:", error);
  }
}

async function main() {
  
  const provider = new Provider("https://testnet.zkevm.cronos.org");
  const latestBlock = await getLatestBlock(provider);
  if (latestBlock) {
    console.log("Block height:", latestBlock.number);
  } 
  sendAmount();
}

main().catch((error) => {
  console.error("An error occurred in the main function:", error);
  process.exit(1);
});