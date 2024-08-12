import { ethers } from 'hardhat';
import { BigNumber } from 'ethers';
import { MyErc1155 } from '../typechain-types';
import { showSigners } from './util';
import { ask } from './util';
export let Erc1155ContractAddress: string | undefined;


async function showErc1155Balance(
  tokenString: string,
  contract: MyErc1155,
  tokenid: BigNumber,
  address: string
) {
  const goldBalance = await contract.balanceOf(address, tokenid);
  console.table(
    [
      {
        TokenID: tokenid,
        Address: address,
        Balance: goldBalance
      }
    ],
    ['TokenID', 'Address', 'Balance']
  );
}

// write showErc1155Balances
async function showErc1155Balances(
  tokenString: string,
  contract: MyErc1155,
  tokenid: BigNumber,
  addresses: string[]
) {
  let items: any = [];

  for (let i = 0; i < addresses.length; i++) {
    const address = addresses[i];
    const goldBalance = await contract.balanceOf(address, tokenid);
    let item = {
      TokenID: tokenid,
      Address: address,
      Balance: goldBalance
    };
    items.push(item);
    console.log(`fetching balance for tokenid=${tokenid} address=${address}`);
  }
  console.table(items, ['TokenID', 'Address', 'Balance']);
}

async function erc1155Transfer(
  contract: MyErc1155,
  from: string,
  to: string,
  tokenid: string,
  amount: string
) {
  const signerContract = contract.connect(ethers.provider.getSigner(from));
  return signerContract.safeTransferFrom(from, to, tokenid, amount, '0x');
}

export async function main1155(
  contractaddress: string | undefined,
  ) {
  let deployedContract = undefined;

  if (contractaddress) {
    deployedContract = await ethers.getContractAt('MyErc1155', contractaddress);

    console.table([
      {
        'ERC1155 contract address': contractaddress,
        'Contract from address': contractaddress
      }
    ]);
  } else {
    console.table([
      {
        Message: 'erc1155 create new contract'
      }
    ]);
    const erc1155Contract = await ethers.getContractFactory('MyErc1155');
    deployedContract = await erc1155Contract.deploy();
  }
  Erc1155ContractAddress = deployedContract.address;

  const [owner] = await ethers.getSigners();
  const balance = await owner.getBalance();

  console.table([
    {
      'Owner address': owner.address,
      'Owner balance': balance.toString()
    }
  ]);

  const signers = await ethers.getSigners();
  //await showSigners(signers);

  const myErc1155: MyErc1155 = await deployedContract.deployed();

  console.table([
    {
      'NFT1155 deployed address': signers[0].address,
      'Deployed contract': deployedContract.address
    }
  ]);

  let fromaddress=signers[0].address;
  let toaddress=signers[1].address;

  const gold = await myErc1155.GOLD();
  await showErc1155Balances('GOLD', myErc1155, gold, [
    fromaddress,
    toaddress
  ]);

  
  let doGame = false
  // ask user, if user enter y, then doGame is true
  let doGameString = await ask('Do game? (y/n) ');
  if (doGameString === 'y') {
    doGame = true;
  }
  if (doGame) {
    const trasnferResult = await erc1155Transfer(
      myErc1155,
      fromaddress,
      toaddress,
      gold.toString(),
      '1'
    );
    await new Promise((r) => setTimeout(r, 5000));
    await showErc1155Balances('GOLD', myErc1155, gold, [
      fromaddress,
      toaddress
    ]);
  }  
  
}
