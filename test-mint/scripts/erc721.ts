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
  contractaddress: string | undefined
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
      type: mintResult.type, // 0: legay, 1: eip2930, 2: eip 1559
      from: mintResult.from.substring(0, 8) + '...'
      // result: JSON.stringify(mintResult),
    });
  }

  if (generateCount > 0) {
    console.table(mintResults, ['hash', 'nonce', 'chainId', 'type', 'from']);

    await new Promise((r) => setTimeout(r, 5000));
  }

  const tokenCounter = await myErc721._tokenIdCounter();
  console.log(`Current token counter: ${tokenCounter}`);


  // read count from terminal
  let countString = await ask('How many tokens to show? ');
  let count = Number(countString);

  
  let doGame = false
  // ask user, if user enter y, then doGame is true
  let doGameString = await ask('Do game? (y/n) ');
  if (doGameString === 'y') {
    doGame = true;
  }
  if (doGame) {
    let tokenid=0;
    // read tokenid from terminal
    let tokenidString = await ask('change owner , tokenid= ');
    tokenid = Number(tokenidString);
    const firstowner = await myErc721.ownerOf(tokenid);
    console.log(`first owner= ${firstowner}`);
    let fromaddress="";
    let toaddress="";
    if (firstowner === signers[0].address) {
      fromaddress = signers[0].address;
      toaddress = signers[1].address;
    } else {
      fromaddress = signers[1].address;
      toaddress = signers[0].address;
    }
    
    await showErc721(myErc721, count);
    await erc721Transfer(
          myErc721,
          fromaddress,
          toaddress,
          tokenid.toString()
     );
    
     
  }

  await new Promise((r) => setTimeout(r, 5000));
  await showErc721(myErc721, count);    
  const id = await myErc721._tokenIdCounter();
  console.log(`current tokenid= ${id}`);
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
    console.log("fetching tokenid=",i);
  }
  console.table(data);
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
