import { query721, main721, Erc721ContractAddress } from './erc721';
import { main1155, Erc1155ContractAddress } from './erc1155';
import { ethers } from 'hardhat';
import { showSigners, ask } from './util';

function displayEnv() {
  console.table(
    [
      { Variable: 'MYCONTRACT721', Value: process.env.MYCONTRACT721 },
      { Variable: 'MYCONTRACT1155', Value: process.env.MYCONTRACT1155 },
      { Variable: 'MYCRONOSRPC', Value: process.env.MYCRONOSRPC },
      { Variable: 'MYCRONOSCHAINID', Value: process.env.MYCRONOSCHAINID },
      { Variable: 'MYMNEMONICS', Value: 'your mnemonics' }
    ],
    ['Variable', 'Value']
  );
}

async function runQueryContract() {
  console.log("query contract erc721");
  await query721(process.env.MYCONTRACT721);
}

function showMenu() {
  console.table(
    [
      {
        Option: '0',
        Description: 'show singers'
      },
      {
        Option: '1',
        Description: 'deploy erc721 contract'
      },
      {
        Option: '2',
        Description: 'deploy erc1155 contract'
      },
      {
        Option: '3',
        Description: 'run query contract'
      },
      {
        Option: '4',
        Description: 'parse artifact file'
      },
      {
        Option: '9',
        Description: 'show env'
      },
      {
        Option: 'q',
        Description: 'exit'
      }
    ],
    ['Option', 'Description']
  );
}


import { promises as fs } from 'fs';

async function readFile(path: string): Promise<string> {
  const data = await fs.readFile(path, 'utf8');
  return data;
}


async function parseArtifact() {
  console.log("parse artifact file");
  const filename= await ask("filename=");
  // print filename
  console.table([{
    'filename': filename,
  }],['filename']);

  const rawText= await readFile(filename);
  const abifilename= await ask("abi filename=");
  const topJson= JSON.parse(rawText);
  const abiJson= JSON.stringify(topJson.abi);
  const bytecodeJson= JSON.stringify(topJson.bytecode);
  await fs.writeFile(abifilename, abiJson);
  await fs.writeFile(abifilename+".bytecode", bytecodeJson);


}
async function main() {
  displayEnv();
  while (true) {
    showMenu();
    const answer = await ask("select=");
    console.log(`chose ${answer}`);

    // dispatch by answer
    if (answer === '0') {
      const signers = await ethers.getSigners();
      await showSigners(signers);
    } else if (answer === '1') {
      await main721(process.env.MYCONTRACT721,true);
    } else if (answer === '2') {
      await main1155(process.env.MYCONTRACT1155,true);
    } else if (answer === '9') {
      displayEnv();
    } else if (answer==='3') {
      await runQueryContract();
    }
    else if (answer === '4') {
      await parseArtifact();
    }
    else if (answer === 'q') {
      process.exit(0);
    }

  }
}

async function handleError(error: Error): Promise<void> {
  console.error(error);
  process.exitCode = 1;
}

// run
main().catch(handleError);
