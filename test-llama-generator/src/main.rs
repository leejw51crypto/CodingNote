//! This is a translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;
use std::ffi::CString;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;
use std::time::Duration;

/// Command line arguments
#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The path to the model
    #[command(subcommand)]
    model: Model,
    /// The prompt
    #[clap(default_value = "Hello my name is")]
    prompt: String,
    /// Set the length of the prompt + output in tokens
    #[arg(long, default_value_t = 32)]
    n_len: i32,
    /// Override some parameters of the model
    #[arg(short = 'o', value_parser = parse_key_val)]
    key_value_overrides: Vec<(String, ParamOverrideValue)>,
    /// Disable offloading layers to the gpu
    #[cfg(feature = "cublas")]
    #[clap(long)]
    disable_gpu: bool,
}

/// Parse a single key-value pair
fn parse_key_val(s: &str) -> Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{}`", s))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;

    Ok((key, value))
}

/// Model source options
#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the model
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// The repo containing the model
        repo: String,
        /// The model name
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

/// Llama text generator
pub struct LlamaGenerator {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl LlamaGenerator {
    /// Create a new Llama text generator
    pub fn new() -> Result<Self> {
        let backend = LlamaBackend::init()?;
        let model = Self::create_model(&backend)?;
        Ok(Self { backend, model })
    }

    /// Run the text generation
    pub async fn run(&mut self) -> Result<()> {
        loop {
            println!("Enter prompt:");
            let input_prompt: String = text_io::read!("{}\n");
            if input_prompt.is_empty() {
                break;
            }
            // Initialize the context
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(2048))
                .with_seed(1234);

            let mut ctx: LlamaContext = self
                .model
                .new_context(&self.backend, ctx_params)
                .with_context(|| "unable to create the llama_context")?;

            let mut batch = Self::create_batch(&self.model, &ctx, &input_prompt)?;
            println!("  ");
            println!("#################batch created#################");
            println!("batch {:?}", batch);

            Self::generate_text(&self.model, &mut ctx, &mut batch, &input_prompt)?;
        }
        Ok(())
    }

    /// Create the Llama model
    fn create_model(backend: &LlamaBackend) -> Result<LlamaModel> {
        let Args {
            n_len: _,
            model,
            prompt: _,
            #[cfg(feature = "cublas")]
            disable_gpu,
            key_value_overrides,
        } = Args::parse();

        // Initialize model parameters
        let model_params = {
            #[cfg(feature = "cublas")]
            if !disable_gpu {
                LlamaModelParams::default().with_n_gpu_layers(1000)
            } else {
                LlamaModelParams::default()
            }
            #[cfg(not(feature = "cublas"))]
            LlamaModelParams::default()
        };

        let mut model_params = pin!(model_params);

        for (k, v) in &key_value_overrides {
            let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
            model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        }

        let model_path = model
            .get_or_load()
            .with_context(|| "failed to get model from args")?;

        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .with_context(|| "unable to load model")?;
        Ok(model)
    }

    /// Create a Llama batch for text generation
    fn create_batch(
        mymodel: &LlamaModel,
        ctx: &LlamaContext,
        input_prompt: &str,
    ) -> Result<LlamaBatch> {
        let Args {
            n_len,
            model: _,
            prompt: _,
            #[cfg(feature = "cublas")]
            disable_gpu,
            key_value_overrides: _,
        } = Args::parse();
        // Tokenize the prompt
        let tokens_list: Vec<LlamaToken> = mymodel
            .str_to_token(input_prompt, AddBos::Always)
            .with_context(|| format!("failed to tokenize {input_prompt}"))?;
        println!("token list : {:?}", tokens_list);
        let n_cxt = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

        eprintln!("n_len = {n_len}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

        // Make sure the KV cache is big enough to hold all the prompt and generated tokens
        if n_kv_req > n_cxt {
            bail!(
                "n_kv_req > n_ctx, the required kv cache size is not big enough
    either reduce n_len or increase n_ctx"
            )
        }

        if tokens_list.len() >= usize::try_from(n_len)? {
            bail!("the prompt is too long, it has more tokens than n_len")
        }

        // Print the prompt token-by-token
        eprintln!();

        for token in &tokens_list {
            eprint!("{}", mymodel.token_to_str(*token)?);
        }

        std::io::stderr().flush()?;

        // Create a llama_batch with size 512
        // We use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(512, 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        Ok(batch)
    }

    /// Generate text based on the input prompt
    fn generate_text(
        mymodel: &LlamaModel,
        ctx: &mut LlamaContext,
        batch: &mut LlamaBatch,
        _input_prompt: &str,
    ) -> Result<()> {
        let Args {
            n_len,
            model: _,
            prompt: _,
            #[cfg(feature = "cublas")]
            disable_gpu,
            key_value_overrides: _,
        } = Args::parse();

        ctx.decode(batch).with_context(|| "llama_decode() failed")?;

        // Main loop
        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;

        let t_main_start = ggml_time_us();

        while n_cur <= n_len {
            // Sample the next token
            {
                let candidates = ctx.candidates_ith(batch.n_tokens() - 1);

                let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

                // Sample the most likely token
                let new_token_id = ctx.sample_token_greedy(candidates_p);

                // Is it an end of stream?
                if new_token_id == mymodel.token_eos() {
                    eprintln!();
                    break;
                }

                print!("{}", mymodel.token_to_str(new_token_id)?);
                std::io::stdout().flush()?;

                batch.clear();
                batch.add(new_token_id, n_cur, &[0], true)?;
            }

            n_cur += 1;

            ctx.decode(batch).with_context(|| "failed to eval")?;

            n_decode += 1;
        }

        eprintln!("\n");

        let t_main_end = ggml_time_us();

        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

        eprintln!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );

        println!("{}", ctx.timings());
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut generator = LlamaGenerator::new()?;

    generator.run().await?;
    Ok(())
}
