#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::data_array::LlamaTokenDataArray,
};
use std::{
    io::{self, Write},
    num::NonZeroU32,
    path::PathBuf,
};

/// Represents the core AI functionality
struct OndeviceAi {
    backend: LlamaBackend,
    model: LlamaModel,
    ctx_params: LlamaContextParams,
}

impl OndeviceAi {
    /// Creates a new OndeviceAi instance
    fn new() -> Result<Self> {
        let mut backend = LlamaBackend::init()?;
        backend.void_logs();

        let model_params = LlamaModelParams::default();
        let model_path = Self::get_model_path()?;
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
            .with_seed(1234);

        Ok(Self {
            backend,
            model,
            ctx_params,
        })
    }

    /// Asks for the model path or uses the default
    fn get_model_path() -> Result<PathBuf> {
        print!("Enter the path to the GGUF model file (or press Enter for default): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            Ok(PathBuf::from("my.gguf"))
        } else {
            Ok(PathBuf::from(input))
        }
    }

    /// Generates text based on the given prompt
    fn generate(&self, prompt: &str) -> Result<()> {
        let mut ctx = self
            .model
            .new_context(&self.backend, self.ctx_params.clone())?;
        let tokens_list = self.model.str_to_token(prompt, AddBos::Always)?;
        let mut batch = self.create_initial_batch(&tokens_list)?;

        ctx.decode(&mut batch)?;

        let mut n_cur = batch.n_tokens();
        let n_len = 512; // Maximum number of tokens to generate
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut newline_count = 0;

        while n_cur <= n_len {
            let new_token_id = self.sample_next_token(&mut ctx, &batch)?;

            if new_token_id == self.model.token_eos() {
                println!("\n🏁 End of generation");
                break;
            }

            if new_token_id == llama_cpp_2::token::LlamaToken(13) {
                newline_count += 1;
            }
            self.process_and_print_token(new_token_id, &mut decoder)?;

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch).context("failed to eval")?;

            if newline_count >= 10 {
                break;
            }
        }

        Ok(())
    }

    /// Creates the initial batch from the tokenized prompt
    fn create_initial_batch(
        &self,
        tokens_list: &[llama_cpp_2::token::LlamaToken],
    ) -> Result<LlamaBatch> {
        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens_list.len() - 1) as i32;

        for (i, &token) in tokens_list.iter().enumerate() {
            let is_last = i as i32 == last_index;
            batch.add(token, i as i32, &[0], is_last)?;
        }

        Ok(batch)
    }

    /// Samples the next token using greedy sampling
    fn sample_next_token(
        &self,
        ctx: &mut llama_cpp_2::context::LlamaContext,
        batch: &LlamaBatch,
    ) -> Result<llama_cpp_2::token::LlamaToken> {
        let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
        let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
        Ok(ctx.sample_token_greedy(candidates_p))
    }

    /// Processes and prints the generated token
    fn process_and_print_token(
        &self,
        token_id: llama_cpp_2::token::LlamaToken,
        decoder: &mut encoding_rs::Decoder,
    ) -> Result<()> {
        let output_bytes = self.model.token_to_bytes(token_id, Special::Tokenize)?;
        let mut output_string = String::with_capacity(32);
        let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
        print!("{}", output_string);
        io::stdout().flush()?;
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("🤖 Welcome to the AI Chatbot! 🌟");
    let ai_core = OndeviceAi::new()?;

    // Main interaction loop for the AI chatbot
    loop {
        print!("📨 Enter your prompt (or 'quit' to exit): ");
        io::stdout().flush()?;

        let mut prompt = String::new();
        io::stdin().read_line(&mut prompt)?;
        prompt = prompt.trim().to_string();

        if prompt.to_lowercase() == "quit" {
            println!("👋 Goodbye! Thanks for chatting!");
            break;
        }

        println!("🧠 Generating response...");
        ai_core.generate(&prompt)?;
        println!("✨ Generation complete!\n");
    }

    Ok(())
}
