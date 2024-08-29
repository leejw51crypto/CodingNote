use anyhow::Result;
use ethers::abi::Abi;
use ethers::abi::Token;
use ethers::core::k256::ecdsa::SigningKey;
use ethers::providers::Http;
use ethers::signers::MnemonicBuilder;
use ethers::types::H160;
use rpassword::prompt_password;
use std::env;
use zksync_web3_rs::providers::{Middleware, Provider};
use zksync_web3_rs::signers::Signer;
use zksync_web3_rs::zks_provider::ZKSProvider;
use zksync_web3_rs::zks_wallet::{CallRequest, DeployRequest};
use zksync_web3_rs::ZKSWallet;

struct ZkEvmCore {
    zk_wallet: ZKSWallet<Provider<Http>, SigningKey>,
}

impl ZkEvmCore {
    async fn new(mnemonics: &str, index: u32) -> Result<Self> {
        let wallet = MnemonicBuilder::<ethers::signers::coins_bip39::English>::default()
            .phrase(mnemonics)
            .index(index)?
            .build()?;

        let l2_provider = Provider::try_from("https://testnet.zkevm.cronos.org")?;
        let chain_id = l2_provider.get_chainid().await?;
        let l2_wallet = wallet.with_chain_id(chain_id.as_u64());

        let zk_wallet = ZKSWallet::new(l2_wallet, None, Some(l2_provider.clone()), None)?;

        Ok(Self { zk_wallet })
    }

    async fn deploy_contract(
        &self,
        abi: Abi,
        contract_bin: Vec<u8>,
        args: Vec<String>,
    ) -> Result<H160> {
        let request =
            DeployRequest::with(abi, contract_bin, args).from(self.zk_wallet.l2_address());
        self.zk_wallet
            .deploy(&request)
            .await
            .map_err(|e| anyhow::anyhow!("Deploy error: {}", e))
    }

    async fn call_view_method(&self, contract_address: H160, method: &str) -> Result<Vec<String>> {
        let era_provider = self.zk_wallet.get_era_provider()?;
        let call_request = CallRequest::new(contract_address, method.to_owned());
        ZKSProvider::call(era_provider.as_ref(), &call_request)
            .await
            .map_err(|e| anyhow::anyhow!("Call error: {}", e))
            .and_then(tokens_to_strings)
    }

    async fn send_transaction(
        &self,
        contract_address: H160,
        method: &str,
        args: Option<Vec<String>>,
    ) -> Result<ethers::types::TxHash> {
        let receipt = self
            .zk_wallet
            .get_era_provider()?
            .clone()
            .send_eip712(
                &self.zk_wallet.l2_wallet,
                contract_address,
                method,
                args,
                None,
            )
            .await?
            .await?
            .ok_or_else(|| anyhow::anyhow!("Transaction receipt not found"))?;

        Ok(receipt.transaction_hash)
    }
}

// Helper function to convert Token to String
fn tokens_to_strings(tokens: Vec<Token>) -> Result<Vec<String>> {
    tokens
        .into_iter()
        .map(|token| match token {
            Token::String(s) => Ok(s),
            _ => Err(anyhow::anyhow!("Unexpected token type")),
        })
        .collect()
}

static CONTRACT_BIN: &str = include_str!("./Greeter.bin");
static CONTRACT_ABI: &str = include_str!("./Greeter.abi");

fn get_mnemonics() -> Result<String> {
    let mut mnemonics = prompt_password("Enter your mnemonics: ")?;
    if mnemonics.is_empty() {
        mnemonics = env::var("MY_MNEMONICS")?;
    }
    assert!(!mnemonics.is_empty(), "Mnemonics cannot be empty");
    Ok(mnemonics)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    println!("🔐 Welcome to the ZkSync Wallet Demo! 🚀");

    let mnemonics = get_mnemonics()?;
    let index: u32 = 0;

    println!("🔧 Setting up ZkEvmCore...");
    let core = ZkEvmCore::new(&mnemonics, index).await?;

    // Deploy contract:
    let contract_address = {
        println!("📝 Deploying smart contract...");
        let abi = Abi::load(CONTRACT_ABI.as_bytes())?;
        let contract_bin = hex::decode(CONTRACT_BIN)?.to_vec();
        let address = core
            .deploy_contract(abi, contract_bin, vec!["Hey".to_owned()])
            .await?;

        println!("✅ Contract deployed successfully!");
        println!("📍 Contract address: {:#?}", address);

        address
    };

    // Call the greet view method:
    {
        println!("👋 Calling greet() method...");
        let greet = core
            .call_view_method(contract_address, "greet()(string)")
            .await?;
        println!("🗨️ Greeting: {}", greet[0]);
    }

    // Perform a signed transaction calling the setGreeting method
    {
        println!("✍️ Setting new greeting...");
        let tx_hash = core
            .send_transaction(
                contract_address,
                "setGreeting(string)",
                Some(vec!["Hello".into()]),
            )
            .await?;
        println!("🔗 setGreeting transaction hash: {:#?}", tx_hash);
    };

    // Call the greet view method again:
    {
        println!("👋 Calling greet() method again...");
        let greet = core
            .call_view_method(contract_address, "greet()(string)")
            .await?;
        println!("🗨️ Updated greeting: {}", greet[0]);
    }

    println!("🎉 Demo completed successfully! 🎊");

    Ok(())
}
