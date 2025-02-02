"use client";

import React, { useState } from "react";
import { Card } from "./ui/Card";
import { Button } from "./ui/Button";
import { ThresholdSignatureScheme } from "../lib/tss/ThresholdSignatureScheme";
import { Party, PartialSignature } from "../lib/tss/types";
import { TSSSignatureInfo } from "./TSSSignatureInfo";
import { BigNumber } from "@ethersproject/bignumber";
import { ethers } from "ethers";
import { arrayify, hexlify, concat } from "@ethersproject/bytes";
import { keccak256 } from "@ethersproject/keccak256";

interface EthereumSignature {
  r: string;
  s: string;
  v: number;
}

interface TransactionInfo {
  txHash: string;
  senderBalanceBefore: string;
  senderBalanceAfter: string;
  receiverBalanceBefore: string;
  receiverBalanceAfter: string;
  gasCost: string;
}

export const TSSWallet: React.FC = () => {
  const [threshold, setThreshold] = useState(2);
  const [numParties, setNumParties] = useState(3);
  const [privateKey, setPrivateKey] = useState("");
  const [recipientAddress, setRecipientAddress] = useState("");
  const [amount, setAmount] = useState("");
  const [tss, setTss] = useState<ThresholdSignatureScheme>();
  const [tssAddress, setTssAddress] = useState<string>();
  const [parties, setParties] = useState<Party[]>([]);
  const [partialSignatures, setPartialSignatures] = useState<
    { party: Party; signature: PartialSignature }[]
  >([]);
  const [aggregatedSignature, setAggregatedSignature] =
    useState<EthereumSignature>();
  const [recoveredAddress, setRecoveredAddress] = useState<string>();
  const [txHash, setTxHash] = useState<string>();
  const [transactionInfo, setTransactionInfo] = useState<TransactionInfo>();
  const [error, setError] = useState<string>();

  const initializeTSS = async () => {
    try {
      setError(undefined);
      if (!privateKey) {
        setError("Please enter a private key");
        return;
      }

      const tssInstance = new ThresholdSignatureScheme(true); // Enable debug mode

      // Ensure private key has 0x prefix
      const cleanKey = privateKey.startsWith("0x")
        ? privateKey
        : `0x${privateKey}`;
      const privateKeyBN = BigNumber.from(cleanKey);

      const keyData = await tssInstance.setupExistingKey(
        privateKeyBN,
        threshold,
        numParties,
      );
      setTss(tssInstance);
      setTssAddress(keyData.tssAddress);
      setParties(keyData.parties);
      setPartialSignatures([]);
      setAggregatedSignature(undefined);
      setRecoveredAddress(undefined);
      setTxHash(undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to initialize TSS");
    }
  };

  const verifySignature = async () => {
    if (!tss || !parties.length || !recipientAddress || !amount) {
      setError("Please provide recipient address and amount");
      return;
    }

    try {
      setError(undefined);
      const provider = new ethers.JsonRpcProvider("https://evm-t3.cronos.org/");
      const chainId = 338;

      // Prepare transaction data
      const nonce = await provider.getTransactionCount(tssAddress!);
      const gasPrice = await provider
        .getFeeData()
        .then((data) => data.gasPrice ?? BigNumber.from("5000000000"));
      const amountWei = ethers.parseEther(amount);
      const gasLimit = 21000;

      // Create raw transaction array following RLP encoding order
      const rawTx = {
        nonce: nonce || 0,
        gasPrice: gasPrice.toString(),
        gasLimit: gasLimit,
        to: recipientAddress,
        value: amountWei.toString(),
        data: "0x",
        chainId: chainId,
        r: "0x",
        s: "0x",
        v: chainId,
      };

      // Get message hash using RLP encoding
      const messageHash = arrayify(
        keccak256(ethers.Transaction.from(rawTx).unsignedSerialized),
      );

      // Generate common seed from message hash and nonce
      const commonSeed = arrayify(
        keccak256(
          concat([messageHash, arrayify(BigNumber.from(nonce).toHexString())]),
        ),
      );

      // Generate partial signatures
      const signatures: { party: Party; signature: PartialSignature }[] = [];
      for (let i = 0; i < threshold; i++) {
        const party = parties[i];
        const signature = await tss.createPartialSignature(
          party,
          messageHash,
          commonSeed,
        );
        signatures.push({ party, signature });
      }
      setPartialSignatures(signatures);

      // Combine signatures
      const result = await tss.combineSignatures(
        signatures.map((s) => s.signature),
        parties.slice(0, threshold),
        messageHash,
      );

      if (
        result.success &&
        result.signature &&
        typeof result.signature === "object"
      ) {
        const sig = result.signature as EthereumSignature;

        // Normalize s value according to EIP-2 BEFORE v calculation
        const halfCurveN = BigNumber.from(
          "0x7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0",
        );
        let s = BigNumber.from(sig.s);
        if (s.gt(halfCurveN)) {
          s = BigNumber.from(
            "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
          ).sub(s);
        }

        // Try both possible v values for recovery (27 and 28)
        let finalSig: EthereumSignature | null = null;
        for (const possibleRecoveredV of [27, 28]) {
          // Calculate v value according to EIP-155
          const testV = chainId * 2 + 35 + (possibleRecoveredV - 27);
          const testSig = {
            r: sig.r,
            s: s.toHexString(),
            v: testV,
          };

          try {
            // Create the signed transaction for verification
            const signedTx = {
              ...rawTx,
              r: testSig.r,
              s: testSig.s,
              v: testV,
            };

            // Recover address from the signed transaction
            const testRecovered = ethers.recoverAddress(messageHash, testSig);
            console.log("testRecovered", testRecovered);
            console.log("tssAddress", tssAddress);
            if (testRecovered.toLowerCase() === tssAddress!.toLowerCase()) {
              finalSig = testSig;
              break;
            }
          } catch (err) {
            // Ignore invalid signature errors for the other v value
            console.log("Recovery error:", err);
          }
        }

        if (!finalSig) {
          throw new Error(
            "Failed to find valid v value for signature recovery",
          );
        }

        setAggregatedSignature(finalSig);
        setRecoveredAddress(tssAddress!); // We've already validated it matches
      } else {
        throw new Error(result.error || "Failed to combine signatures");
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to verify signature",
      );
    }
  };

  const generateRandomPrivateKey = () => {
    const randomBytes = Buffer.from(
      Array(32)
        .fill(0)
        .map(() => Math.floor(Math.random() * 256)),
    );
    setPrivateKey("0x" + randomBytes.toString("hex"));
  };

  const sendTransaction = async () => {
    if (!tss || !parties.length || !recipientAddress || !amount) {
      setError("Please provide recipient address and amount");
      return;
    }

    try {
      setError(undefined);
      const result = await tss.sendTransaction(
        parties,
        recipientAddress,
        amount,
        threshold,
      );

      if (!result.success) {
        throw new Error(result.error || "Transaction failed");
      }

      setTransactionInfo({
        txHash: result.txHash!,
        senderBalanceBefore: result.senderBalanceBefore!,
        senderBalanceAfter: result.senderBalanceAfter!,
        receiverBalanceBefore: result.receiverBalanceBefore!,
        receiverBalanceAfter: result.receiverBalanceAfter!,
        gasCost: result.gasCost!,
      });

      setTxHash(result.txHash);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to send transaction",
      );
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <div className="p-4">
          <h2 className="text-lg font-semibold mb-4">TSS Wallet Setup</h2>
          <div className="space-y-4">
            {/* Threshold and Number of Parties */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Threshold
                </label>
                <input
                  type="number"
                  min="2"
                  value={threshold}
                  onChange={(e) => setThreshold(parseInt(e.target.value))}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Number of Parties
                </label>
                <input
                  type="number"
                  min={threshold}
                  value={numParties}
                  onChange={(e) => setNumParties(parseInt(e.target.value))}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                />
              </div>
            </div>

            {/* Private Key Input with Generate Button */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Private Key (optional)
              </label>
              <div className="mt-1 flex gap-2">
                <input
                  type="password"
                  value={privateKey}
                  onChange={(e) => setPrivateKey(e.target.value)}
                  placeholder="0x..."
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                />
                <Button
                  onClick={generateRandomPrivateKey}
                  variant="outline"
                  size="sm"
                >
                  Generate
                </Button>
              </div>
            </div>

            {/* Original and TSS Addresses */}
            {tssAddress && (
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  TSS Wallet Address
                </label>
                <div className="mt-1 font-mono text-sm break-all bg-gray-50 p-2 rounded">
                  {tssAddress}
                </div>
              </div>
            )}

            {/* Transaction Fields */}
            {tssAddress && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Recipient Address
                  </label>
                  <input
                    type="text"
                    value={recipientAddress}
                    onChange={(e) => setRecipientAddress(e.target.value)}
                    placeholder="0x..."
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Amount (TCRO)
                  </label>
                  <input
                    type="text"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    placeholder="0.1"
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                  />
                </div>
              </>
            )}

            {/* Action Buttons */}
            <div className="flex gap-4 flex-wrap">
              <Button
                onClick={initializeTSS}
                variant={tssAddress ? "outline" : "default"}
              >
                {tssAddress ? "Re-Initialize TSS" : "Initialize TSS"}
              </Button>

              {tssAddress && (
                <>
                  <Button
                    onClick={verifySignature}
                    disabled={!recipientAddress || !amount}
                    variant={!recipientAddress || !amount ? "ghost" : "default"}
                  >
                    Verify Signature
                  </Button>

                  <Button
                    onClick={sendTransaction}
                    disabled={
                      !recipientAddress ||
                      !amount ||
                      !recoveredAddress ||
                      recoveredAddress.toLowerCase() !==
                        tssAddress.toLowerCase()
                    }
                    variant={
                      !recipientAddress ||
                      !amount ||
                      !recoveredAddress ||
                      recoveredAddress.toLowerCase() !==
                        tssAddress.toLowerCase()
                        ? "ghost"
                        : "default"
                    }
                  >
                    Send Transaction
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      </Card>

      <TSSSignatureInfo
        threshold={threshold}
        numParties={numParties}
        tssAddress={tssAddress}
        parties={parties}
        partialSignatures={partialSignatures}
        aggregatedSignature={aggregatedSignature}
        recoveredAddress={recoveredAddress}
        txHash={txHash}
        transactionInfo={transactionInfo}
        error={error}
      />
    </div>
  );
};
