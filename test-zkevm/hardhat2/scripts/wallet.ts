import { ethers } from 'ethers';
import { HardhatRuntimeEnvironment } from "hardhat/types";
import * as hre from "hardhat";

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

async function main() {
    // Get the network configuration
    const network = hre.network.config as any;

    // Get the mnemonic from the network configuration
    const mnemonic = network.accounts.mnemonic;

    if (!mnemonic) {
        console.error('Mnemonic not found in hardhat.config.ts');
        process.exit(1);
    }

    const { privateKey, publicKey, address } = getPrivateKeyFromMnemonic(mnemonic);
    console.table([
        // for debugging
        // { 'Key Type': 'Private Key', 'Value': privateKey, 'Length (bytes)': privateKey.length / 2 },
        { 'Key Type': 'Public Key', 'Value': publicKey, 'Length (bytes)': publicKey.length / 2 },
        { 'Key Type': 'Wallet Address', 'Value': address, 'Length (bytes)': address.length / 2 }
    ]);
}

main().catch((error) => {
    console.error('Error:', error);
    process.exit(1);
});