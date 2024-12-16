import { ethers } from "ethers";
import promptSync from 'prompt-sync';

async function main() {
    const prompt = promptSync({ sigint: true });
    
    // Connect to Cronos zkEVM Testnet
    const provider = new ethers.JsonRpcProvider("https://testnet.zkevm.cronos.org");
    
    // Get addresses from console input
    const contractAddress = prompt('Enter contract address: ');
    const walletAddress = prompt('Enter wallet address: ');
    
    if (!contractAddress || !walletAddress) {
        console.error("Both addresses are required.");
        process.exit(1);
    }

    // Validate addresses
    if (!ethers.isAddress(contractAddress) || !ethers.isAddress(walletAddress)) {
        console.error("Invalid address format. Please provide valid Ethereum addresses.");
        process.exit(1);
    }

    try {
        // Get balance using balanceOf function signature
        const balanceOfData = ethers.id("balanceOf(address)").slice(0, 10) + 
            walletAddress.slice(2).padStart(64, '0');

        console.log("Calling contract with data:", balanceOfData);

        // First check if contract exists
        const code = await provider.getCode(contractAddress);
        console.log("Contract code exists:", code !== "0x");

        const balance = await provider.call({
            to: contractAddress,
            data: balanceOfData
        });

        console.log("Raw balance response:", balance);

        if (balance === "0x") {
            console.log("Balance is zero or contract call failed");
            return;
        }

        // Convert balance from hex to decimal and format it
        const balanceInWei = ethers.toBigInt(balance);
        const balanceInEth = ethers.formatUnits(balanceInWei, 18);
        
        console.log(`Balance: ${balanceInEth} vETH`);
        
    } catch (error) {
        console.error("Error fetching balance:", error);
        
        // Add more detailed error logging
        if (error instanceof Error) {
            console.error("Error details:", {
                message: error.message,
                code: (error as any).code,
                data: (error as any).data
            });
        }
    }
}

main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
});