import { type BigNumberish, type SignatureLike } from "ethers";
import { BigNumber } from "@ethersproject/bignumber";
import { arrayify, hexZeroPad, concat, hexlify } from "@ethersproject/bytes";
import { getAddress } from "@ethersproject/address";
import { keccak256 } from "@ethersproject/keccak256";
import { _TypedDataEncoder } from "@ethersproject/hash";
import { ec as EC } from "elliptic";
import { Party, TSSKeyData, PartialSignature } from "./types";
import { randomBytes } from "crypto";
import { ethers } from "ethers";

interface SignatureResult {
  success: boolean;
  error?: string;
  signature?: SignatureLike;
  txHash?: string;
}

interface TransactionResult {
  success: boolean;
  error?: string;
  txHash?: string;
  senderBalanceBefore?: string;
  senderBalanceAfter?: string;
  receiverBalanceBefore?: string;
  receiverBalanceAfter?: string;
  gasCost?: string;
}

export class ThresholdSignatureScheme {
  private curve: EC;
  private curveOrder: bigint;
  private g: any; // elliptic.js point
  private debug: boolean;
  private groupPublicKey: Uint8Array;
  private tssAddress: string;

  constructor(debug: boolean = false) {
    this.curve = new EC("secp256k1");
    this.curveOrder = BigInt(
      "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
    );
    this.g = this.curve.g;
    this.debug = debug;
    this.groupPublicKey = new Uint8Array(); // Initialize with empty array, will be set during setupExistingKey
    this.tssAddress = ""; // Will be set during setupExistingKey
  }

  private log(message: string, data?: any) {
    if (this.debug) {
      console.log(`[TSS] ${message}`);
      if (data) {
        console.log(data);
      }
    }
  }

  private logError(message: string, error?: any) {
    if (this.debug) {
      console.error(`[TSS Error] ${message}`);
      if (error) {
        console.error(error);
      }
    }
  }

  private modInverse(a: bigint, m: bigint): bigint {
    // Extended Euclidean Algorithm implementation
    let [old_r, r] = [a, m];
    let [old_s, s] = [1n, 0n];

    while (r !== 0n) {
      const quotient = old_r / r;
      [old_r, r] = [r, old_r - quotient * r];
      [old_s, s] = [s, old_s - quotient * s];
    }

    let result = old_s;
    if (result < 0n) {
      result += m;
    }
    return result;
  }

  private lagrangeCoefficient(
    parties: Party[],
    j: number,
    x: number = 0,
  ): bigint {
    let num = 1n;
    let den = 1n;

    for (const m of parties) {
      if (m.id !== j) {
        // Do modulo at each step to prevent overflow
        num = (num * BigInt(x - m.id)) % this.curveOrder;
        den = (den * BigInt(j - m.id)) % this.curveOrder;
      }
    }

    // Handle negative values in modular arithmetic
    if (den < 0n) {
      den = (den + this.curveOrder) % this.curveOrder;
    }
    if (num < 0n) {
      num = (num + this.curveOrder) % this.curveOrder;
    }

    return (num * this.modInverse(den, this.curveOrder)) % this.curveOrder;
  }

  private deriveEthereumAddress(groupPublicKey: Uint8Array): string {
    console.log("Deriving Ethereum address from group public key");
    console.log("Group public key length:", groupPublicKey.length, "bytes");
    console.log(
      "Group public key (hex):",
      Buffer.from(groupPublicKey).toString("hex"),
    );

    // Skip the first byte (format byte) and hash the remaining public key bytes
    const publicKeyBytes = groupPublicKey.slice(1);
    console.log(
      "Public key bytes length (after slice):",
      publicKeyBytes.length,
      "bytes",
    );
    console.log(
      "Public key bytes (hex):",
      Buffer.from(publicKeyBytes).toString("hex"),
    );

    const hash = keccak256(Buffer.from(publicKeyBytes));
    console.log("Keccak hash:", hash);

    const address = getAddress("0x" + hash.slice(-40));
    console.log("Derived address:", address);
    return address;
  }

  public async setupExistingKey(
    privateKey: BigNumber,
    threshold: number,
    numParties: number,
  ): Promise<TSSKeyData> {
    try {
      if (threshold < 2 || threshold > numParties) {
        const error = `Invalid threshold configuration: threshold=${threshold}, numParties=${numParties}`;
        this.logError(error);
        throw new Error(error);
      }

      this.log(
        `Setting up TSS with threshold ${threshold} out of ${numParties} parties`,
      );

      // Generate coefficients for polynomial
      const coefficients = [privateKey];
      for (let i = 1; i < threshold; i++) {
        const coeff = BigNumber.from(randomBytes(32)).mod(
          this.curveOrder.toString(),
        );
        coefficients.push(coeff);
      }

      const parties: Party[] = [];
      // Generate shares using polynomial evaluation
      for (let i = 0; i < numParties; i++) {
        const x = i + 1;
        let share = coefficients[0];

        for (let j = 1; j < threshold; j++) {
          const exp = BigNumber.from(x).pow(j).mod(this.curveOrder.toString());
          const term = coefficients[j].mul(exp).mod(this.curveOrder.toString());
          share = share.add(term).mod(this.curveOrder.toString());
        }

        // Generate public key for share
        const keyPair = this.curve.keyFromPrivate(
          share.toHexString().slice(2),
          "hex",
        );
        const publicKey = new Uint8Array(
          keyPair.getPublic().encode("array", true),
        );

        parties.push({
          id: x,
          xi: share,
          publicKey,
        });
        this.log(`Generated key share for party ${x}`);
      }

      // Generate group public key
      const groupKeyPair = this.curve.keyFromPrivate(
        privateKey.toHexString().slice(2),
        "hex",
      );
      this.groupPublicKey = new Uint8Array(
        groupKeyPair.getPublic().encode("array", false),
      );

      // Derive Ethereum address from public key
      this.tssAddress = this.deriveEthereumAddress(this.groupPublicKey);

      this.log(`TSS setup complete`, {
        address: this.tssAddress,
        threshold,
        numParties,
        partiesGenerated: parties.length,
      });

      return {
        parties,
        tssAddress: this.tssAddress,
        groupPublicKey: this.groupPublicKey,
        threshold,
        numParties,
      };
    } catch (error) {
      this.logError("Failed to setup TSS key shares", error);
      throw error;
    }
  }

  private generateBaseK(
    messageHash: Uint8Array,
    commonSeed: Uint8Array,
  ): BigNumber {
    // Use commonSeed directly as the deterministic input
    // This ensures all parties generate the same k value
    return this.generateKRFC6979(messageHash, BigNumber.from(commonSeed));
  }

  private generateK(
    messageHash: Uint8Array,
    privateKey: string,
    commonSeed?: Uint8Array,
  ): BigNumber {
    // If common seed is provided, use generateBaseK to ensure consistency
    if (commonSeed) {
      return this.generateBaseK(messageHash, commonSeed);
    }
    // Otherwise use private key for single-party signing
    return this.generateKRFC6979(messageHash, BigNumber.from(privateKey));
  }

  private generateKRFC6979(
    messageHash: Uint8Array,
    privateKey: BigNumber,
  ): BigNumber {
    // RFC 6979 implementation for deterministic k generation
    let V = Uint8Array.from(new Array(32).fill(1)); // V = 0x01 0x01 ... 0x01
    let K = Uint8Array.from(new Array(32).fill(0)); // K = 0x00 0x00 ... 0x00

    // Convert inputs to octets
    const x = Uint8Array.from(
      arrayify(hexZeroPad(privateKey.toHexString(), 32)),
    );
    const h1 = Uint8Array.from(
      arrayify(hexZeroPad(BigNumber.from(messageHash).toHexString(), 32)),
    );

    // Initial K = HMAC_K(V || 0x00 || int2octets(x) || bits2octets(h1))
    K = Uint8Array.from(
      arrayify(keccak256(concat([K, V, new Uint8Array([0]), x, h1]))),
    );

    // V = HMAC_K(V)
    V = Uint8Array.from(arrayify(keccak256(concat([K, V]))));

    // K = HMAC_K(V || 0x01 || int2octets(x) || bits2octets(h1))
    K = Uint8Array.from(
      arrayify(keccak256(concat([K, V, new Uint8Array([1]), x, h1]))),
    );

    // V = HMAC_K(V)
    V = Uint8Array.from(arrayify(keccak256(concat([K, V]))));

    // Generate k
    while (true) {
      let T = new Uint8Array(0);
      while (T.length < 32) {
        V = Uint8Array.from(arrayify(keccak256(concat([K, V]))));
        T = Uint8Array.from(concat([T, V]));
      }

      const k = BigNumber.from(T).mod(this.curveOrder.toString());
      if (!k.isZero() && k.lt(this.curveOrder.toString())) {
        return k;
      }

      K = Uint8Array.from(
        arrayify(keccak256(concat([K, V, new Uint8Array([0])]))),
      );
      V = Uint8Array.from(arrayify(keccak256(concat([K, V]))));
    }
  }

  public async createPartialSignature(
    party: Party,
    messageHash: Uint8Array,
    commonSeed?: Uint8Array,
  ): Promise<PartialSignature> {
    try {
      // Get first and last bytes of private key for debugging
      const xiHex = party.xi.toHexString();
      const xiFirstByte = xiHex.slice(0, 4);
      const xiLastByte = xiHex.slice(-2);
      this.log(
        `Creating partial signature for party ${party.id} (private key: ${xiFirstByte}...${xiLastByte})`,
      );

      // Always use generateBaseK when commonSeed is provided to ensure consistency
      const k = commonSeed
        ? this.generateBaseK(messageHash, commonSeed)
        : this.generateK(messageHash, party.xi.toHexString());
      const kPair = this.curve.keyFromPrivate(k.toHexString().slice(2), "hex");
      const R = kPair.getPublic();
      const r = BigNumber.from("0x" + R.getX().toString(16));

      const z = BigNumber.from(messageHash);
      const kInv = BigNumber.from(
        this.modInverse(BigInt(k.toString()), this.curveOrder).toString(),
      );
      const s = kInv
        .mul(z.add(r.mul(party.xi)))
        .mod(this.curveOrder.toString());

      const signature = {
        r,
        s: arrayify(hexZeroPad(s.toHexString(), 32)),
        k,
        R: new Uint8Array(R.encode("array", true)),
      };

      this.log(`Generated partial signature for party ${party.id}`, {
        r: r.toHexString(),
        s: hexlify(signature.s),
        R: {
          x: R.getX().toString(16),
          y: R.getY().toString(16),
        },
      });

      return signature;
    } catch (error) {
      this.logError(
        `Failed to create partial signature for party ${party.id}`,
        error,
      );
      throw error;
    }
  }

  public async combineSignatures(
    partialSignatures: PartialSignature[],
    parties: Party[],
    messageHash: Uint8Array,
  ): Promise<SignatureResult> {
    try {
      this.log(`Combining ${partialSignatures.length} partial signatures`);

      // Verify all r values are the same
      const r = partialSignatures[0].r;
      const R = partialSignatures[0].R;
      const firstR = this.curve.keyFromPublic(R).getPublic();

      for (const sig of partialSignatures.slice(1)) {
        if (
          !sig.r.eq(r) ||
          !arrayify(sig.R).every(
            (byte: number, index: number) => byte === R[index],
          )
        ) {
          throw new Error("Inconsistent R values in partial signatures");
        }
      }

      // Combine s values using Lagrange interpolation
      let sCombined = BigNumber.from(0);
      const participatingParties = parties.slice(0, partialSignatures.length);

      for (let i = 0; i < partialSignatures.length; i++) {
        const s_i = BigNumber.from(hexlify(partialSignatures[i].s));
        const lambda_i = this.lagrangeCoefficient(
          participatingParties,
          participatingParties[i].id,
        );

        // Convert lambda to negative if needed (following Python implementation)
        let lambda_bignum = BigNumber.from(lambda_i.toString());
        if (lambda_bignum.isNegative()) {
          lambda_bignum = BigNumber.from(this.curveOrder.toString()).add(
            lambda_bignum,
          );
        }

        // Get first and last bytes of private key for debugging
        const xiHex = participatingParties[i].xi.toHexString();
        const xiFirstByte = xiHex.slice(0, 4);
        const xiLastByte = xiHex.slice(-2);

        this.log(
          `Combined signature part ${i + 1}/${partialSignatures.length}`,
          {
            partyId: participatingParties[i].id,
            privateKey: `${xiFirstByte}...${xiLastByte}`,
            lambda: lambda_i.toString(16),
            lambda_adjusted: lambda_bignum.toHexString(),
            partialS: hexlify(partialSignatures[i].s),
            R: {
              x: firstR.getX().toString(16),
              y: firstR.getY().toString(16),
            },
          },
        );

        const contribution = s_i
          .mul(lambda_bignum)
          .mod(this.curveOrder.toString());
        sCombined = sCombined.add(contribution).mod(this.curveOrder.toString());
      }

      // Normalize s according to EIP-2
      const halfCurveOrder = BigNumber.from(this.curveOrder.toString()).div(2);
      if (sCombined.gt(halfCurveOrder)) {
        sCombined = BigNumber.from(this.curveOrder.toString()).sub(sCombined);
      }

      // Try both possible v values (27, 28)
      const chainId = 338; // Cronos testnet
      const possibleBaseV = [27, 28];
      let validSignature = null;
      let validV = null;

      for (const baseV of possibleBaseV) {
        // Calculate EIP-155 v value: v = chain_id * 2 + 35 + (base_v - 27)
        const v = chainId * 2 + 35 + (baseV - 27);

        const testSignature = {
          r: r.toHexString(),
          s: sCombined.toHexString(),
          v,
        };

        // Verify the signature
        try {
          const recoveredAddress = ethers.recoverAddress(
            messageHash,
            testSignature,
          );
          const expectedAddress = this.tssAddress;

          this.log(`Testing signature with v=${v} (base_v=${baseV})`, {
            recoveredAddress,
            expectedAddress,
            match:
              recoveredAddress.toLowerCase() === expectedAddress.toLowerCase(),
          });

          if (
            recoveredAddress.toLowerCase() === expectedAddress.toLowerCase()
          ) {
            validSignature = testSignature;
            validV = v;
            break;
          }
        } catch (error) {
          this.log(`Failed to verify signature with v=${v}`, error);
          continue;
        }
      }

      if (!validSignature) {
        throw new Error("Could not find valid v value for signature");
      }

      this.log(`Successfully combined signatures`, {
        finalSignature: validSignature,
        baseV: (validV as number) - (chainId * 2 + 35) + 27,
        eip155V: validV,
        chainId,
        txHash: keccak256(
          concat([
            arrayify(validSignature.r),
            arrayify(validSignature.s),
            arrayify(
              hexZeroPad(BigNumber.from(validSignature.v).toHexString(), 32),
            ),
          ]),
        ),
      });

      return {
        success: true,
        signature: validSignature,
        txHash: keccak256(
          concat([
            arrayify(validSignature.r),
            arrayify(validSignature.s),
            arrayify(
              hexZeroPad(BigNumber.from(validSignature.v).toHexString(), 32),
            ),
          ]),
        ),
      };
    } catch (error) {
      this.logError("Failed to combine signatures", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  public async sendTransaction(
    parties: Party[],
    toAddress: string,
    amountEth: string,
    threshold: number,
  ): Promise<TransactionResult> {
    try {
      // Connect to Cronos testnet
      const provider = new ethers.JsonRpcProvider("https://evm-t3.cronos.org/");
      const chainId = 338;

      // Use the known TSS wallet address
      const senderAddress = this.tssAddress;

      // Get initial balances
      const senderBalanceBefore = await provider.getBalance(senderAddress);
      const receiverBalanceBefore = await provider.getBalance(toAddress);
      this.log("Initial balances", {
        sender: ethers.formatEther(senderBalanceBefore.toString()),
        receiver: ethers.formatEther(receiverBalanceBefore.toString()),
      });

      // Prepare transaction
      const nonce = await provider.getTransactionCount(senderAddress);
      const gasPrice = await provider
        .getFeeData()
        .then((data) => data.gasPrice?.toString() ?? "5000000000");
      const amountWei = ethers.parseEther(amountEth);
      const gasLimit = 21000; // Standard ETH transfer

      // Create transaction object
      const unsignedTx = {
        to: toAddress,
        nonce: nonce,
        gasLimit: gasLimit,
        gasPrice: gasPrice,
        value: amountWei,
        data: "0x",
        chainId: chainId,
      };

      // Serialize transaction to get the message to sign
      const serializedUnsigned =
        ethers.Transaction.from(unsignedTx).unsignedSerialized;
      const messageHash = ethers.keccak256(serializedUnsigned);
      this.log("Message hash for signing", { hash: messageHash });

      // Generate a common seed from message hash and private key shares
      const commonSeed = arrayify(
        keccak256(
          concat([
            arrayify(messageHash),
            arrayify(BigNumber.from(nonce).toHexString()),
            // Add party public keys to the seed for additional entropy
            ...parties.slice(0, threshold).map((p) => p.publicKey),
          ]),
        ),
      );
      this.log("Generated common seed", { seed: hexlify(commonSeed) });

      // Generate partial signatures
      const partialSignatures: PartialSignature[] = [];
      for (let i = 0; i < threshold; i++) {
        const signature = await this.createPartialSignature(
          parties[i],
          arrayify(messageHash),
          commonSeed,
        );
        partialSignatures.push(signature);
        this.log(`Generated partial signature for party ${i + 1}/${threshold}`);
      }

      // Combine signatures
      const result = await this.combineSignatures(
        partialSignatures,
        parties,
        arrayify(messageHash),
      );
      if (!result.success || !result.signature) {
        throw new Error(result.error || "Failed to combine signatures");
      }

      // Extract signature components
      const sig = result.signature as { r: string; s: string; v: number };

      // Create signed transaction
      const signedTx = {
        ...unsignedTx,
        signature: {
          r: sig.r,
          s: sig.s,
          v: sig.v,
        },
      };

      // Serialize the signed transaction
      const serializedTx = ethers.Transaction.from(signedTx).serialized;

      // Verify the signature recovers to our sender address
      const recoveredAddress = ethers.recoverAddress(messageHash, {
        r: sig.r,
        s: sig.s,
        v: sig.v,
      });

      this.log("Signature verification", {
        recoveredAddress,
        expectedAddress: senderAddress,
        messageHash: messageHash,
        signature: sig,
      });

      if (recoveredAddress.toLowerCase() !== senderAddress.toLowerCase()) {
        throw new Error(
          `Invalid signature: recovered address ${recoveredAddress} does not match sender ${senderAddress}`,
        );
      }

      // Send transaction
      this.log("Broadcasting transaction...");
      const tx = await provider.broadcastTransaction(serializedTx);
      this.log("Transaction sent", { hash: tx.hash });

      // Wait for confirmation
      this.log("Waiting for confirmation...");
      const receipt = await tx.wait();
      if (!receipt) {
        throw new Error("Transaction failed - no receipt received");
      }
      this.log("Transaction confirmed", { blockNumber: receipt.blockNumber });

      // Get final balances
      const senderBalanceAfter = await provider.getBalance(senderAddress);
      const receiverBalanceAfter = await provider.getBalance(toAddress);
      const gasCost = BigNumber.from(gasPrice.toString()).mul(receipt.gasUsed);

      this.log("Final balances", {
        sender: ethers.formatEther(senderBalanceAfter.toString()),
        receiver: ethers.formatEther(receiverBalanceAfter.toString()),
        gasCost: ethers.formatEther(gasCost.toString()),
      });

      return {
        success: true,
        txHash: tx.hash,
        senderBalanceBefore: ethers.formatEther(senderBalanceBefore.toString()),
        senderBalanceAfter: ethers.formatEther(senderBalanceAfter.toString()),
        receiverBalanceBefore: ethers.formatEther(
          receiverBalanceBefore.toString(),
        ),
        receiverBalanceAfter: ethers.formatEther(
          receiverBalanceAfter.toString(),
        ),
        gasCost: ethers.formatEther(gasCost.toString()),
      };
    } catch (error) {
      this.logError("Failed to send transaction", error);
      return {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unknown error during transaction",
      };
    }
  }
}
