import { Provider } from "zksync-ethers";

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

async function main() {
  const provider = new Provider("https://testnet.zkevm.cronos.org");
  const latestBlock = await getLatestBlock(provider);
  if (latestBlock) {
    console.log("Block height:", latestBlock.number);
  } 
}

main().catch((error) => {
  console.error("An error occurred in the main function:", error);
  process.exit(1);
});