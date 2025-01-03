import { ethers } from 'hardhat';
import * as readline from 'readline';

export async function showSigners(signers: any[]) {
  const data = [];

  for (let i = 0; i < signers.length; i++) {
    const signer = signers[i];
    
    data.push({
      Index: i,
      Address: signer.address,
    
    });

    
  }
  console.table(data, ['Index', 'Address']);
}
export function ask(question: string): Promise<string> {
  const readlineInterface = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });

  return new Promise<string>((resolve) => {
    readlineInterface.question(question, (answer: string) => {
      readlineInterface.close();
      resolve(answer);
    });
  });
}

// write delay function
export function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
