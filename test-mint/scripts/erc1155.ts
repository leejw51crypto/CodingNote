import { ethers } from 'hardhat';
import { BigNumber } from 'ethers';
import { MyErc1155 } from '../typechain-types';
import { showSigners } from './util';

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
    console.table(items, ['TokenID', 'Address', 'Balance']);
  }
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
  doGame = false
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

  const gold = await myErc1155.GOLD();
  await showErc1155Balances('GOLD', myErc1155, gold, [
    owner.address,
    signers[2].address
  ]);

  const doBasic = false;
  if (doBasic) {
    const trasnferResult = await erc1155Transfer(
      myErc1155,
      signers[0].address,
      signers[2].address,
      gold.toString(),
      '1'
    );
    await new Promise((r) => setTimeout(r, 5000));
    await showErc1155Balances('GOLD', myErc1155, gold, [
      owner.address,
      signers[2].address
    ]);
  }

  if (doGame) {
    for (let i = 0; i < 10; i++) {
      console.log(`Loop ${i}`);
      if (0 == i % 2) {
        await erc1155Transfer(
          myErc1155,
          signers[2].address,
          signers[0].address,
          gold.toString(),
          '1'
        );
      } else {
        await erc1155Transfer(
          myErc1155,
          signers[0].address,
          signers[2].address,
          gold.toString(),
          '1'
        );
      }
    }
    // sleep 5 seconds
    await new Promise((r) => setTimeout(r, 5000));
    await showErc1155Balances('GOLD', myErc1155, gold, [
      signers[0].address,
      signers[2].address
    ]);
  }
}
