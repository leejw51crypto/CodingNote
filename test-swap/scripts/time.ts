import { ethers } from 'hardhat';
import { VVSRouter, IERC20 } from '../typechain-types';
import config from '../hardhat.config';


async function main() {
  console.log(Math.floor(Date.now() / 1000) + 60 * 20);
}

async function handleError(error: Error): Promise<void> {
  console.error(error);
  process.exitCode = 1;
}

// run
main().catch(handleError);
