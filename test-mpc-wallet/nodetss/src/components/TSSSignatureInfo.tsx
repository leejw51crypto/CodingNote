import React from "react";
import { Card } from "./ui/Card";
import { Party, PartialSignature } from "../lib/tss/types";

interface TSSSignatureInfoProps {
  threshold: number;
  numParties: number;
  tssAddress?: string;
  recoveredAddress?: string;
  parties?: Party[];
  partialSignatures?: { party: Party; signature: PartialSignature }[];
  aggregatedSignature?: {
    r: string;
    s: string;
    v: number;
  };
  txHash?: string;
  transactionInfo?: {
    txHash: string;
    senderBalanceBefore: string;
    senderBalanceAfter: string;
    receiverBalanceBefore: string;
    receiverBalanceAfter: string;
    gasCost: string;
  };
  error?: string;
}

export const TSSSignatureInfo: React.FC<TSSSignatureInfoProps> = ({
  threshold,
  numParties,
  tssAddress,
  recoveredAddress,
  parties,
  partialSignatures,
  aggregatedSignature,
  txHash,
  transactionInfo,
  error,
}) => {
  return (
    <div className="space-y-4">
      {/* Threshold Information */}
      <Card>
        <div className="p-4">
          <h2 className="text-lg font-semibold mb-2">
            Threshold Configuration
          </h2>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-gray-600">Required Signatures:</span>
              <span className="ml-2 font-mono">{threshold}</span>
            </div>
            <div>
              <span className="text-gray-600">Total Parties:</span>
              <span className="ml-2 font-mono">{numParties}</span>
            </div>
            {tssAddress && (
              <div className="col-span-2">
                <span className="text-gray-600">TSS Address:</span>
                <span className="ml-2 font-mono break-all">{tssAddress}</span>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Party Signatures */}
      {partialSignatures && partialSignatures.length > 0 && (
        <Card>
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">Party Signatures</h2>
            <div className="space-y-3">
              {partialSignatures.map(({ party, signature }) => (
                <div key={party.id} className="border-b last:border-b-0 pb-2">
                  <div className="font-medium">Party {party.id}</div>
                  <div className="grid grid-cols-1 gap-1 mt-1">
                    <div className="text-sm">
                      <span className="text-gray-600">r:</span>
                      <span className="ml-2 font-mono break-all">
                        {signature.r.toHexString()}
                      </span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-600">s:</span>
                      <span className="ml-2 font-mono break-all">
                        {Buffer.from(signature.s).toString("hex")}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* Aggregated Signature */}
      {aggregatedSignature && (
        <Card>
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">Aggregated Signature</h2>
            <div className="space-y-2">
              <div>
                <span className="text-gray-600">r:</span>
                <span className="ml-2 font-mono break-all">
                  {aggregatedSignature.r}
                </span>
              </div>
              <div>
                <span className="text-gray-600">s:</span>
                <span className="ml-2 font-mono break-all">
                  {aggregatedSignature.s}
                </span>
              </div>
              <div>
                <span className="text-gray-600">v:</span>
                <span className="ml-2 font-mono">{aggregatedSignature.v}</span>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Transaction Information */}
      {transactionInfo && (
        <Card>
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">Transaction Details</h2>
            <div className="space-y-2">
              <div>
                <span className="text-gray-600">Transaction Hash:</span>
                <span className="ml-2 font-mono break-all">
                  {transactionInfo.txHash}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Sender Balance Before:</span>
                <span className="ml-2 font-mono">
                  {transactionInfo.senderBalanceBefore} TCRO
                </span>
              </div>
              <div>
                <span className="text-gray-600">Sender Balance After:</span>
                <span className="ml-2 font-mono">
                  {transactionInfo.senderBalanceAfter} TCRO
                </span>
              </div>
              <div>
                <span className="text-gray-600">Receiver Balance Before:</span>
                <span className="ml-2 font-mono">
                  {transactionInfo.receiverBalanceBefore} TCRO
                </span>
              </div>
              <div>
                <span className="text-gray-600">Receiver Balance After:</span>
                <span className="ml-2 font-mono">
                  {transactionInfo.receiverBalanceAfter} TCRO
                </span>
              </div>
              <div>
                <span className="text-gray-600">Gas Cost:</span>
                <span className="ml-2 font-mono">
                  {transactionInfo.gasCost} TCRO
                </span>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Transaction Hash */}
      {txHash && !transactionInfo && (
        <Card>
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-2">Transaction</h2>
            <div>
              <span className="text-gray-600">Hash:</span>
              <span className="ml-2 font-mono break-all">{txHash}</span>
            </div>
          </div>
        </Card>
      )}

      {/* Error Message */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <div className="p-4">
            <h2 className="text-lg font-semibold text-red-700 mb-2">Error</h2>
            <div className="text-red-600">{error}</div>
          </div>
        </Card>
      )}
    </div>
  );
};
