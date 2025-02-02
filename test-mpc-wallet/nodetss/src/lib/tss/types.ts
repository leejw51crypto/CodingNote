import { BigNumber } from "@ethersproject/bignumber";

export interface Party {
  id: number;
  xi: BigNumber; // Secret share of the private key
  publicKey: Uint8Array; // Public key share
}

export interface TSSKeyData {
  parties: Party[];
  tssAddress: string;
  groupPublicKey: Uint8Array;
  threshold: number;
  numParties: number;
}

export interface PartialSignature {
  r: BigNumber;
  s: Uint8Array;
  k: BigNumber;
  R: Uint8Array;
}
