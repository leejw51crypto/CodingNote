//import { makeCosmoshubPath, makeSignDoc } from "@cosmjs/amino";
import { makeCosmoshubPath } from "@cosmjs/amino";
import  { pathToString } from "@cosmjs/crypto";
import { toBase64 } from "@cosmjs/encoding";
// eslint-disable-next-line @typescript-eslint/naming-convention
import { LedgerSigner } from "@cosmjs/ledger-amino";
// eslint-disable-next-line @typescript-eslint/naming-convention
import TransportNodeHid from "@ledgerhq/hw-transport-node-hid";
import { MsgSendEncodeObject, SigningStargateClient } from "@cosmjs/stargate";
import { TxRaw, TxBody, AuthInfo} from "cosmjs-types/cosmos/tx/v1beta1/tx";
import {MsgSend} from "cosmjs-types/cosmos/bank/v1beta1/tx";

import {toHex} from "@cosmjs/encoding";


function showTxRaw(txrawbytes: Uint8Array) {
  // convert Tx from txrawbytes 
  const txraw = TxRaw.decode(txrawbytes);
  
  console.log("txraw josn:", JSON.stringify(txraw));

  const body = TxBody.decode(txraw.bodyBytes);
  const authInfo =AuthInfo.decode(txraw.authInfoBytes);
  console.log("body json:", JSON.stringify(body));
  console.log("authInfo json:", JSON.stringify(authInfo));
  console.log("signature:", JSON.stringify(txraw.signatures));
  // convert any to MsgSend
  //const msgSend = MsgSend.decode(body.messages[0]);
  const message= body.messages[0];
  // print message.typeUrl
  console.log("message typeUrl:", message.typeUrl);
  // print message.value
  console.log("message value:", JSON.stringify(message.value));
  const sendmessage= MsgSend.decode(message.value);
  console.log("message:", JSON.stringify(sendmessage));
  

}


// disalble eslint-disable-next-line @typescript-eslint/no-explicit-any
async function SendLedger(toaddr:string,amount="1", amountdenom="uatom", feeamount="1", feedenom="uatom", feegas="95000", memo="", chainid="theta-testnet-001",rpcendpoint="https://rpc.sentry-01.theta-testnet.polypore.xyz") {

  const defaultFee = {
    amount: [{ amount: feeamount, denom: feedenom}],
    gas: feegas,
  };

  const interactiveTimeout = 120_000;
  const accountNumbers = [0];
  const paths = accountNumbers.map(makeCosmoshubPath);


  const ledgerTransport = await TransportNodeHid.create(interactiveTimeout, interactiveTimeout);
  const signer = new LedgerSigner(ledgerTransport, { testModeAllowed: true, hdPaths: paths });

  const accounts = await signer.getAccounts();
  const printableAccounts = accounts.map((account) => ({ ...account, pubkey: toBase64(account.pubkey) }));
  console.info("Accounts from Ledger device:");
  console.table(printableAccounts.map((account, i) => ({ ...account, hdPath: pathToString(paths[i]) })));

  const fromaddr= accounts[0].address;
  console.log("fromaddr:", fromaddr);
  console.log("-----------------------------------------------------");

  const client = await SigningStargateClient.connectWithSigner(rpcendpoint,signer );
  const before = await client.getBalance(fromaddr, "uatom");
  const accountinfo =await client.getAccount(fromaddr);
  console.log('from address balance:', before);
  console.log('from address info:', accountinfo);

  const sendmsg: MsgSendEncodeObject = {
    typeUrl: "/cosmos.bank.v1beta1.MsgSend",
    value: {
      fromAddress: fromaddr,
      toAddress: toaddr,
      amount: [
        {
          "denom": amountdenom,
          "amount": amount
        }
      ],
    },
  };


  const txraw=await client.sign(fromaddr, [sendmsg],defaultFee, memo, { accountNumber:accountinfo.accountNumber, sequence: accountinfo.sequence,  chainId:chainid });
  console.log('txraw:', txraw);
  console.log('txraw json:', JSON.stringify(txraw));
  // convert txraw to uint8array
  const signedBytes = Uint8Array.from(TxRaw.encode(txraw).finish());
  const hexstring = toHex(signedBytes);
  console.log(`signedBytes in hex: ${hexstring}`);
  
  showTxRaw(signedBytes);
  const txreceipt=await client.broadcastTx(signedBytes);
  console.log(`tx receipt json: ${JSON.stringify(txreceipt)}`);

 
}


async function run() {
  // read environment variable S2 
  let toaddr = process.env.COSMOS_TO_ADDRESS;
  // if toaddr is not set, read from user 
  if (!toaddr) {
    console.log("Enter to address: ");
    toaddr = await new Promise((resolve) => {
      process.stdin.once("data", (data) => {
        resolve(data.toString().trim());
      });
    });
  }

  // print toaddr
  console.log("toaddr:", toaddr);

  await SendLedger(toaddr);
}

run().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
