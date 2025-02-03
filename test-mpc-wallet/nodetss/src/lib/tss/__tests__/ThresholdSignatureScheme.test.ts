import { ThresholdSignatureScheme } from "../ThresholdSignatureScheme";
import { BigNumber } from "@ethersproject/bignumber";
import { arrayify, hexlify } from "@ethersproject/bytes";
import { keccak256 } from "@ethersproject/keccak256";
import { ethers } from "ethers";

describe("ThresholdSignatureScheme", () => {
  describe("combineSignatures", () => {
    it("should successfully combine partial signatures and verify against wallet address", async () => {
      // Initialize TSS with debug mode
      const tss = new ThresholdSignatureScheme(true);

      // Setup test parameters
      const threshold = 2;
      const numParties = 3;

      // Generate a random private key using ethers v6 API
      const privateKey = BigNumber.from(ethers.randomBytes(32));

      // Setup the TSS key shares
      const keyData = await tss.setupExistingKey(
        privateKey,
        threshold,
        numParties,
      );

      // Create message to sign
      const message = "hello world";
      const messageHash = arrayify(keccak256(ethers.toUtf8Bytes(message)));

      // Generate a common seed for deterministic k values
      const commonSeed = arrayify(
        keccak256(
          ethers.concat([
            messageHash,
            // Add some additional entropy
            ...keyData.parties.slice(0, threshold).map((p) => p.publicKey),
          ]),
        ),
      );

      // Generate partial signatures from first two parties (meeting threshold)
      const partialSignatures = [];
      for (let i = 0; i < threshold; i++) {
        const signature = await tss.createPartialSignature(
          keyData.parties[i],
          messageHash,
          commonSeed,
        );
        partialSignatures.push(signature);
      }

      // Combine the signatures
      const result = await tss.combineSignatures(
        partialSignatures,
        keyData.parties,
        messageHash,
      );

      // Verify the result
      expect(result.success).toBe(true);
      expect(result.signature).toBeDefined();
      expect(result.error).toBeUndefined();

      if (result.signature) {
        // Verify the signature recovers to the correct address
        const recoveredAddress = ethers.recoverAddress(
          messageHash,
          result.signature,
        );

        expect(recoveredAddress.toLowerCase()).toBe(
          keyData.tssAddress.toLowerCase(),
        );
      }
    });
  });
});
