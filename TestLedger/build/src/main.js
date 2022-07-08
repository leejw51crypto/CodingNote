"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tslib_1 = require("tslib");
const amino_1 = require("@cosmjs/amino");
const crypto_1 = require("@cosmjs/crypto");
const encoding_1 = require("@cosmjs/encoding");
const ledger_amino_1 = require("@cosmjs/ledger-amino");
const hw_transport_node_hid_1 = require("@ledgerhq/hw-transport-node-hid");
const stargate_1 = require("@cosmjs/stargate");
const tx_1 = require("cosmjs-types/cosmos/tx/v1beta1/tx");
const encoding_2 = require("@cosmjs/encoding");
function SendLedger(toaddr, amount = "1", amountdenom = "uatom", feeamount = "1", feedenom = "uatom", feegas = "95000", memo = "", chainid = "theta-testnet-001", rpcendpoint = "https://rpc.sentry-01.theta-testnet.polypore.xyz") {
    return (0, tslib_1.__awaiter)(this, void 0, void 0, function* () {
        const defaultFee = {
            amount: [{ amount: feeamount, denom: feedenom }],
            gas: feegas,
        };
        const interactiveTimeout = 120000;
        const accountNumbers = [0];
        const paths = accountNumbers.map(amino_1.makeCosmoshubPath);
        const ledgerTransport = yield hw_transport_node_hid_1.default.create(interactiveTimeout, interactiveTimeout);
        const signer = new ledger_amino_1.LedgerSigner(ledgerTransport, { testModeAllowed: true, hdPaths: paths });
        const accounts = yield signer.getAccounts();
        const printableAccounts = accounts.map((account) => (Object.assign(Object.assign({}, account), { pubkey: (0, encoding_1.toBase64)(account.pubkey) })));
        console.info("Accounts from Ledger device:");
        console.table(printableAccounts.map((account, i) => (Object.assign(Object.assign({}, account), { hdPath: (0, crypto_1.pathToString)(paths[i]) }))));
        const fromaddr = accounts[0].address;
        console.log("fromaddr:", fromaddr);
        console.log("-----------------------------------------------------");
        const client = yield stargate_1.SigningStargateClient.connectWithSigner(rpcendpoint, signer);
        const before = yield client.getBalance(fromaddr, "uatom");
        const accountinfo = yield client.getAccount(fromaddr);
        console.log('from address balance:', before);
        console.log('from address info:', accountinfo);
        const sendmsg = {
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
        const txraw = yield client.sign(fromaddr, [sendmsg], defaultFee, memo, { accountNumber: accountinfo.accountNumber, sequence: accountinfo.sequence, chainId: chainid });
        console.log('txraw:', txraw);
        console.log('txraw json:', JSON.stringify(txraw));
        const signedBytes = Uint8Array.from(tx_1.TxRaw.encode(txraw).finish());
        const hexstring = (0, encoding_2.toHex)(signedBytes);
        console.log(`signedBytes in hex: ${hexstring}`);
        const txreceipt = yield client.broadcastTx(signedBytes);
        console.log(`tx receipt json: ${JSON.stringify(txreceipt)}`);
    });
}
function run() {
    return (0, tslib_1.__awaiter)(this, void 0, void 0, function* () {
        let toaddr = process.env.COSMOS_TO_ADDRESS;
        if (!toaddr) {
            console.log("Enter to address: ");
            toaddr = yield new Promise((resolve) => {
                process.stdin.once("data", (data) => {
                    resolve(data.toString().trim());
                });
            });
        }
        console.log("toaddr:", toaddr);
        yield SendLedger(toaddr);
    });
}
run().then(() => process.exit(0), (err) => {
    console.error(err);
    process.exit(1);
});
//# sourceMappingURL=main.js.map