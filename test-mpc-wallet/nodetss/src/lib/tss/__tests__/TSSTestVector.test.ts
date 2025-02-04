import { ThresholdSignatureScheme } from "../ThresholdSignatureScheme";
import { BigNumber } from "@ethersproject/bignumber";
import { arrayify, hexlify } from "@ethersproject/bytes";
import { ethers } from "ethers";
import fs from "fs";
import path from "path";

// Import types
import { Party } from "../types";

// Read test data from JSON file
const testDataPath = path.join(__dirname, "../../../../../tss_test_data.json");
const testData = JSON.parse(fs.readFileSync(testDataPath, "utf8"));

// Define test data interface
interface TestParty {
  id: number;
  private_share: string;
  public_key: string;
}

interface TestPartialSignature {
  party_id: number;
  r: string;
  s: string;
  k: string;
  R: string;
}

// Helper function to clean hex strings (remove double 0x if present)
function cleanHexString(hex: string): string {
  return hex.replace("0x0x", "0x");
}

describe("TSS Test Vector Verification", () => {
  let tss: ThresholdSignatureScheme;

  beforeEach(() => {
    // Initialize TSS with debug mode for more verbose output
    tss = new ThresholdSignatureScheme(true);
  });

  describe("Test Vector Setup Verification", () => {
    it("should verify test vector setup parameters", () => {
      expect(testData.setup.threshold).toBe(2);
      expect(testData.setup.num_parties).toBe(3);
      expect(testData.setup.tss_address).toBe(
        "0xA324E4581832Bd47Ae4E2Cc464BfF3E8BcCe25c0",
      );
      expect(testData.setup.group_public_key).toMatch(/^0x04/); // Should start with 0x04 for uncompressed point
    });

    it("should verify party shares and public keys", () => {
      testData.parties.forEach((party: TestParty) => {
        expect(party.id).toBeGreaterThan(0);
        expect(party.id).toBeLessThanOrEqual(testData.setup.num_parties);
        expect(party.private_share).toMatch(/^0x/);
        expect(party.public_key).toMatch(/^0x04/); // Uncompressed public key format
      });
    });
  });

  describe("Partial Signature Generation", () => {
    it("should generate matching partial signatures using test vector k value", async () => {
      // Convert test data parties to TSS parties format
      const parties = testData.parties.map((party: TestParty) => ({
        id: party.id,
        xi: BigNumber.from(party.private_share),
        publicKey: arrayify(party.public_key),
      }));

      // Get message hash and k value from test data
      const messageHash = arrayify(
        cleanHexString(testData.test_signing.message_hash),
      );
      const k = BigNumber.from(
        cleanHexString(testData.test_signing.partial_signatures[0].k),
      );

      // Generate and verify each partial signature
      for (let i = 0; i < testData.setup.threshold; i++) {
        const party = parties[i];
        const testSig = testData.test_signing.partial_signatures[i];

        // Create partial signature using our implementation with test k value
        const ourSig = await tss.createPartialSignatureWithK(
          party,
          messageHash,
          k,
        );

        // Compare with test data
        expect(ourSig.r.toHexString()).toBe(testSig.r);
        expect(hexlify(ourSig.s)).toBe(testSig.s);
        expect(hexlify(ourSig.R)).toBe(testSig.R);

        // Detailed logging for the first signature
        if (i === 0) {
          console.log("\nFirst Partial Signature Comparison:");
          console.log("Our r:", ourSig.r.toHexString());
          console.log("Test r:", testSig.r);
          console.log("Our s:", hexlify(ourSig.s));
          console.log("Test s:", testSig.s);
          console.log("Our R:", hexlify(ourSig.R));
          console.log("Test R:", testSig.R);
        }
      }
    });
  });

  describe("Signature Combination", () => {
    it("should combine partial signatures to match test vector", async () => {
      // Convert test data parties to TSS parties format
      const parties = testData.parties.map((party: TestParty) => ({
        id: party.id,
        xi: BigNumber.from(party.private_share),
        publicKey: arrayify(party.public_key),
      }));

      // Set the group public key and TSS address from test data
      (tss as any).groupPublicKey = arrayify(testData.setup.group_public_key);
      (tss as any).tssAddress = testData.setup.tss_address;

      // Get message hash and k value from test data
      const messageHash = arrayify(
        cleanHexString(testData.test_signing.message_hash),
      );
      const k = BigNumber.from(
        cleanHexString(testData.test_signing.partial_signatures[0].k),
      );

      // Generate partial signatures
      const partialSignatures = [];
      for (let i = 0; i < testData.setup.threshold; i++) {
        const signature = await tss.createPartialSignatureWithK(
          parties[i],
          messageHash,
          k,
        );
        partialSignatures.push(signature);
      }

      // Combine signatures
      const result = await tss.combineSignatures(
        partialSignatures,
        parties,
        messageHash,
      );

      // Verify the result
      expect(result.success).toBe(true);
      expect(result.signature).toBeDefined();

      if (result.signature) {
        const sig = result.signature as { r: string; s: string; v: number };

        console.log("\nCombined Signature Comparison:");
        console.log("Our r:", sig.r);
        console.log("Test r:", testData.test_signing.combined_signature.r);
        console.log("Our s:", sig.s);
        console.log("Test s:", testData.test_signing.combined_signature.s);
        console.log("Our v:", sig.v);
        console.log("Test v:", testData.test_signing.combined_signature.v);

        expect(sig.r).toBe(testData.test_signing.combined_signature.r);
        expect(sig.s).toBe(testData.test_signing.combined_signature.s);
        expect(sig.v).toBe(testData.test_signing.combined_signature.v);
        // show sig.r, sig.s with that of test_sigining
        console.log("sig.r:", sig.r);
        console.log(
          "test_signing.combined_signature.r:",
          testData.test_signing.combined_signature.r,
        );
        console.log("sig.s:", sig.s);
        console.log(
          "test_signing.combined_signature.s:",
          testData.test_signing.combined_signature.s,
        );
        // print messageHash in hex
        console.log("messageHash:", hexlify(messageHash));

        // console sog sig.v
        console.log("--------------------------------");
        console.log("sig.v:", sig.v);

        // Verify the signature recovers to the correct address
        // Convert EIP-155 v value back to base v value for recovery
        const chainId = 338;
        const base_v = ((sig.v - 35 - chainId * 2) % 2) + 27;
        const v2 = chainId * 2 + 35 + (base_v - 27);
        // print v2, sig.v
        console.log("v2:", v2);
        console.log("sig.v:", sig.v);

        console.log("base_v:", base_v);

        const normalized_v = (sig.v - (chainId * 2 + 35)) % 2;
        console.log("normalized_v:", normalized_v);
        const recoverySignature = {
          r: sig.r,
          s: sig.s,
          v: sig.v,
        };
        const recoveredAddress = ethers.recoverAddress(
          messageHash,
          recoverySignature,
        );
        expect(recoveredAddress.toLowerCase()).toBe(
          testData.setup.tss_address.toLowerCase(),
        );

        console.log("\nSignature Recovery:");
        console.log("Recovered address:", recoveredAddress);
        console.log("Expected address:", testData.setup.tss_address);
      }
    });
  });
});
