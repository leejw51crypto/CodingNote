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

  
async function main() {
    displayEnv();
}  


async function handleError(error: Error): Promise<void> {
    console.error(error);
    process.exitCode = 1;
  }
  
  // run
  main().catch(handleError);
  