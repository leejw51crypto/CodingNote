"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tslib_1 = require("tslib");
const amino_1 = require("@cosmjs/amino");
const crypto_1 = require("@cosmjs/crypto");
const encoding_1 = require("@cosmjs/encoding");
const ledger_amino_1 = require("@cosmjs/ledger-amino");
const hw_transport_node_hid_1 = require("@ledgerhq/hw-transport-node-hid");
const interactiveTimeout = 120000;
const accountNumbers = [0, 1, 2, 10];
const paths = accountNumbers.map(amino_1.makeCosmoshubPath);
const defaultChainId = "testing";
const defaultFee = {
    amount: [{ amount: "100", denom: "ucosm" }],
    gas: "250",
};
const defaultMemo = "Some memo";
const defaultSequence = "0";
function signMsgSend(signer, accountNumber, fromAddress, toAddress) {
    return (0, tslib_1.__awaiter)(this, void 0, void 0, function* () {
        const msg = {
            type: "cosmos-sdk/MsgSend",
            value: {
                amount: [
                    {
                        amount: "1234567",
                        denom: "ucosm",
                    },
                ],
                from_address: fromAddress,
                to_address: toAddress,
            },
        };
        const signDoc = (0, amino_1.makeSignDoc)([msg], defaultFee, defaultChainId, defaultMemo, accountNumber, defaultSequence);
        const { signature } = yield signer.signAmino(fromAddress, signDoc);
        return signature;
    });
}
function run() {
    return (0, tslib_1.__awaiter)(this, void 0, void 0, function* () {
        const ledgerTransport = yield hw_transport_node_hid_1.default.create(interactiveTimeout, interactiveTimeout);
        const signer = new ledger_amino_1.LedgerSigner(ledgerTransport, { testModeAllowed: true, hdPaths: paths });
        const accounts = yield signer.getAccounts();
        const printableAccounts = accounts.map((account) => (Object.assign(Object.assign({}, account), { pubkey: (0, encoding_1.toBase64)(account.pubkey) })));
        console.info("Accounts from Ledger device:");
        console.table(printableAccounts.map((account, i) => (Object.assign(Object.assign({}, account), { hdPath: (0, crypto_1.pathToString)(paths[i]) }))));
        console.info("Showing address of first account on device");
        yield signer.showAddress();
        console.info("Showing address of 3rd account on device");
        yield signer.showAddress(paths[2]);
        const accountNumber0 = 0;
        const address0 = accounts[accountNumber0].address;
        console.info(`Signing on Ledger device with account index ${accountNumber0} (${address0}). Please review and approve on the device now.`);
        const signature0 = yield signMsgSend(signer, accountNumber0, address0, address0);
        console.info("Signature:", signature0);
        yield new Promise((resolve) => setTimeout(resolve, 1000));
        const accountNumber10 = 10;
        const address10 = accounts[accountNumbers.findIndex((n) => n === accountNumber10)].address;
        console.info(`Signing on Ledger device with account index ${accountNumber10} (${address10}). Please review and approve on the device now.`);
        const signature1 = yield signMsgSend(signer, accountNumber10, address10, address10);
        console.info("Signature:", signature1);
    });
}
run().then(() => process.exit(0), (err) => {
    console.error(err);
    process.exit(1);
});
//# sourceMappingURL=ledger.js.map