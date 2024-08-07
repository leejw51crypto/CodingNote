import { ethers } from 'hardhat';
import { BigNumber } from 'ethers';
import { MyErc721 } from '../typechain-types';
import { ask, showSigners } from './util';
import * as readline from 'readline';

export let Erc721ContractAddress: string | undefined;


export async function query721(contractaddress: string | undefined,) {
  let deployedContract =  await ethers.getContractAt('MyErc721', contractaddress);
  // print deployed contract address
  console.table([{
    'erc721 contract address': contractaddress,
  }],['erc721 contract address']);
  let i=0;
  const uri = await deployedContract.tokenURI(i);
  const owner = await deployedContract.ownerOf(i);
  console.table([{
    'token id': i,
    'token uri': uri,
    'owner': owner,
  }],['token id','token uri','owner']);

}

export async function main721(
  contractaddress: string | undefined,
  doGame = false
) {
  const signers = await ethers.getSigners();

  let deployedContract = undefined;

  if (contractaddress) {
    deployedContract = await ethers.getContractAt('MyErc721', contractaddress);

    console.table({
      'erc721 contract address': contractaddress,
      'get contract from address': contractaddress
    });
  } else {
    console.log('erc721 create new contract');
    const erc721Contrat = await ethers.getContractFactory('MyErc721');
    deployedContract = await erc721Contrat.deploy();
  }
  Erc721ContractAddress = deployedContract.address;

  const [owner] = await ethers.getSigners();
  const balance = await owner.getBalance();

  console.table([
    { 'Owner address': owner.address, 'Owner balance': balance.toString() }
  ]);

  let generateCountString = await ask('How many tokens to generate? ');
  let generateCount = Number(generateCountString);

  const myErc721: MyErc721 = await deployedContract.deployed();

  console.table([{ generateCount: generateCount }]);
  console.table([
    { address: signers[0].address, contract: deployedContract.address }
  ]);

  const mintResults = [];
  for (let i = 0; i < generateCount; i++) {
    console.log(`mint ${i}`);
    const mintResult = await myErc721.safeMint(signers[0].address, `myuri${i}`);
    mintResults.push({
      hash: mintResult.hash.substring(0, 8) + '...',
      nonce: mintResult.nonce,
      chainId: mintResult.chainId,
      type: mintResult.type,
      from: mintResult.from.substring(0, 8) + '...'
      // result: JSON.stringify(mintResult),
    });
  }

  if (generateCount > 0) {
    console.table(mintResults, ['hash', 'nonce', 'chainId', 'type', 'from']);

    await new Promise((r) => setTimeout(r, 5000));
  }

  // read count from terminal
  let countString = await ask('How many tokens to show? ');
  let count = Number(countString);

  if (doGame) {
    let startid = 0;
    const firstowner = await myErc721.ownerOf(0);
    if (firstowner == signers[0].address) {
      startid = 0;
    } else {
      startid = 1;
    }
    const targettokenid = BigNumber.from(2);
    for (let i = startid; i < 2; i++) {
      console.log(`Loop ${i}`);
      if (0 == i % 2) {
        await erc721Transfer(
          myErc721,
          signers[0].address,
          signers[1].address,
          targettokenid.toString()
        );
      } else {
        await erc721Transfer(
          myErc721,
          signers[1].address,
          signers[0].address,
          targettokenid.toString()
        );
      }
      await new Promise((r) => setTimeout(r, 5000));
      await showErc721(myErc721, count);
    }
  }

  await showErc721(myErc721, count);

  const id = await myErc721._tokenIdCounter();
  console.log(`tokenid= ${id}`);
}

async function showErc721(contract: MyErc721, count: number) {
  const data = [];
  for (let i = 0; i < count; i++) {
    try {
      const uri = await contract.tokenURI(i);
      const owner = await contract.ownerOf(i);
      data.push({
        Token: i,
        Owner: owner,
        URI: uri
      });
    } catch (e) {
      break;
    }
    console.table(data);
  }
}

async function erc721Transfer(
  contract: MyErc721,
  from: string,
  to: string,
  tokenid: string
) {
  const signerContract = contract.connect(ethers.provider.getSigner(from));
  return signerContract.transferFrom(from, to, BigNumber.from(tokenid));
}
