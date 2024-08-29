import { ethers } from "hardhat";
import { HardhatRuntimeEnvironment } from "hardhat/types";
import { Deployer } from "@matterlabs/hardhat-zksync-deploy";
import { Provider as ZkProvider, Wallet as ZkWallet } from "zksync-ethers";
import * as dotenv from "dotenv";

dotenv.config();

function getPrivateKeyFromMnemonic(mnemonic: string, index: number = 0): { privateKey: string, publicKey: string, address: string } {
  // Create the HDNode from the mnemonic
  const hdNode = ethers.HDNodeWallet.fromPhrase(mnemonic);

  // Derive the account directly using the full path
  const fullPath = `m/44'/60'/0'/0/${index}`;
  // ethers memnoics from mnemonics
  let ethersMnemonic = ethers.Mnemonic.fromPhrase(mnemonic);
  const wallet = ethers.HDNodeWallet.fromMnemonic(ethersMnemonic, fullPath);

  return {
      privateKey: wallet.privateKey.slice(2),
      publicKey: wallet.publicKey.slice(2),
      address: wallet.address.slice(2)
  };
}

async function create_contract(hre: HardhatRuntimeEnvironment) {
  console.log(`Running deploy script for the Hello contract`);

  // Get the network configuration
  const network = hre.network.config as any;

  // Initialize the providers
  const l1Provider = new ethers.JsonRpcProvider(network.ethNetwork);
  const l2Provider = new ZkProvider(network.url);

  // Get the mnemonic from the network configuration
  const mnemonic = network.accounts.mnemonic;

  // Use the getPrivateKeyFromMnemonic function to get the private key
  const { privateKey } = getPrivateKeyFromMnemonic(mnemonic);

  const zkWallet = new ZkWallet(privateKey, l2Provider, l1Provider);

  // Create deployer object
  const deployer = new Deployer(hre, zkWallet);

  // Load the artifact of the contract you want to deploy.
  const artifact = await deployer.loadArtifact("HelloWorld");

  // Estimate contract deployment fee
  const deploymentFee = await deployer.estimateDeployFee(artifact, []);

  // Deploy this contract. The returned object will be of a `Contract` type, similarly to ones in `ethers`.
  const parsedFee = ethers.formatEther(deploymentFee.toString());
  console.log(`The deployment is estimated to cost ${parsedFee} ETH`);

  const helloContract = await deployer.deploy(artifact, []);

  // Show the contract info.
  const contractAddress = await helloContract.getAddress();
  console.log(`${artifact.contractName} was deployed to ${contractAddress}`);

  // Assuming you have the contract ABI and provider
  const contractABI = artifact.abi;
  const provider = l2Provider;
  const helloWorldContract = new ethers.Contract(contractAddress, contractABI, zkWallet);
  
  // Get the initial greeting
  let greeting = await helloWorldContract.getGreeting();
  console.log("Initial Greeting:", greeting);

  // Set a new greeting
  const newGreeting = "Hello from zkSync!";
  await helloWorldContract.setGreeting(newGreeting);
  console.log("Setting new greeting...");

  // Get the updated greeting
  greeting = await helloWorldContract.getGreeting();
  console.log("Updated Greeting:", greeting);
}

async function main() {
  console.log("Hello, World!");

  // Get 5 accounts from the Hardhat network
  const accounts = await hre.ethers.getSigners();

  for (let i = 0; i < 5; i++) {
    const account = accounts[i];
    console.log(`Account ${i + 1}:`);
    console.log(account);
    console.log();
  }

  // Call create_contract function with hre
  await create_contract(hre);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});