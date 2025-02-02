"use client";

import { useState } from "react";
import { Button } from "./ui/Button";
import { Card } from "./ui/Card";

export default function WalletConnect() {
  const [isConnecting, setIsConnecting] = useState(false);

  const handleConnect = async () => {
    setIsConnecting(true);
    try {
      // TSS wallet connection logic will go here
      await new Promise((resolve) => setTimeout(resolve, 1000)); // Simulated delay
    } finally {
      setIsConnecting(false);
    }
  };

  return (
    <Card className="w-full max-w-md p-6 space-y-4">
      <div className="space-y-2">
        <h2 className="text-2xl font-bold">Connect Wallet</h2>
        <p className="text-gray-500 dark:text-gray-400">
          Connect your TSS wallet to get started
        </p>
      </div>
      <Button
        onClick={handleConnect}
        className="w-full"
        disabled={isConnecting}
      >
        {isConnecting ? "Connecting..." : "Connect TSS Wallet"}
      </Button>
    </Card>
  );
}
