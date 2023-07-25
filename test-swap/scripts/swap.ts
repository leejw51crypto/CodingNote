import { ethers } from 'hardhat';
import { VVSRouter, IERC20 } from '../typechain-types';
import config from '../hardhat.config';

const MY_SWAP_CONTRACT_ADDRESS = config.networks.my.mycontractswap;
const PATH = [
  config.networks.my.mycontractcro,
  config.networks.my.mycontractvvs,
  config.networks.my.mycontractusdc
];
const TO = config.networks.my.mycronosaddress;
const SEND_AMOUNT = config.networks.my.mycroamount;

export async function mainSwap(contractAddress: string | undefined) {
  const endtime=Math.floor(Date.now() / 1000) + 60 * 20;
  console.log("endtime=", endtime);
  const signers = await ethers.getSigners();
  const signer = signers[0];
  const mySwapContract = await ethers.getContractAt(
    'VVSRouter',
    contractAddress,
    signer
  );

  await approveWETH(signer, contractAddress);
  console.log('approved ------------------------');
  //"swapExactETHForTokens(uint256,address[],address,uint256)": FunctionFragment;
  const tx = await mySwapContract.swapExactETHForTokens(
    ethers.utils.parseUnits('1', 'wei'),
    PATH,
    TO,
    Math.floor(Date.now() / 1000) + 60 * 20,
    { value: ethers.utils.parseEther(SEND_AMOUNT), gasLimit: 210000 }
  );

  const receipt = await tx.wait();
  console.table([{ 'Transaction hash': receipt.transactionHash }]);
}

async function approveWETH(signer: Signer, contractAddress: string) {
  //const wethContract = new ethers.Contract(PATH[0], ERC20_ABI, signer);
  const wethContract: IERC20 = await ethers.getContractAt(
    'IERC20',
    PATH[0],
    signer
  );
  await wethContract.approve(
    contractAddress,
    ethers.utils.parseEther(SEND_AMOUNT)
  );
}

async function main() {
  console.table('swap source');
  console.table({
    url: config.networks.my.url,
    chainId: config.networks.my.chainId,
    'my cronos address': TO,
    'send amount cro': SEND_AMOUNT
  });

  await mainSwap(MY_SWAP_CONTRACT_ADDRESS);
}

async function handleError(error: Error): Promise<void> {
  console.error(error);
  process.exitCode = 1;
}

// run
main().catch(handleError);
