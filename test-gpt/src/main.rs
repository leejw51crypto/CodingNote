use anyhow::{Context, Result};
use std::{
    convert::Infallible,
    io::{self, Write},
    path::PathBuf,
};

// Constants for model architecture and model path
const MODEL_ARCHITECTURE: &str = "llama";
const MODEL_PATH: &str = "orca-mini-7b.ggmlv3.q4_0.bin";

#[tokio::main]
async fn main() -> Result<()> {
    let home_dir = ".";

    // Construct model path from home directory and model path
    let model_path = PathBuf::from(format!("{}/{}", home_dir, MODEL_PATH));

    // Parse model architecture string into an enum
    let model_architecture = MODEL_ARCHITECTURE.parse::<llm::ModelArchitecture>()?;

    // Load the model from the provided path
    let model =
        load_model(Some(model_architecture), &model_path).context("Failed to load the model")?;

    // Infinite loop to get user input and run inference on the model
    loop {
        let prompt = get_user_input("Enter a prompt (or 'quit' to exit): ")?;

        if prompt.trim().to_lowercase() == "quit" {
            break;
        }

        run_inference(model.as_ref(), &prompt)?;
    }

    Ok(())
}

/// Load a model from a given path and architecture.
fn load_model(
    architecture: Option<llm::ModelArchitecture>,
    model_path: &std::path::Path,
) -> Result<Box<dyn llm::Model>> {
    let now = std::time::Instant::now();

    let model = llm::load_dynamic(
        architecture,
        model_path,
        llm::TokenizerSource::Embedded,
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .context("Failed to load the model")?;

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    Ok(model)
}

/// Run inference on the model with a given prompt.
fn run_inference(model: &dyn llm::Model, prompt: &str) -> Result<()> {
    let mut session = start_session(model)?;

    process_inference(&mut session, model, prompt)?;

    Ok(())
}

/// Start an inference session.
fn start_session(model: &dyn llm::Model) -> Result<Box<llm::InferenceSession>> {
    Ok(Box::new(model.start_session(Default::default())))
}

/// Process inference for the given session and prompt.
fn process_inference(
    session: &mut llm::InferenceSession,
    model: &dyn llm::Model,
    prompt: &str,
) -> Result<()> {
    let res = session.infer::<Infallible>(
        model,
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: None,
        },
        &mut Default::default(),
        handle_inference_response,
    );

    match res {
        Ok(result) => println!("\n\nInference stats:\n{}", result),
        Err(err) => println!("\n{}", err),
    }

    Ok(())
}

/// Handle inference response and provide feedback.
fn handle_inference_response(
    response: llm::InferenceResponse,
) -> Result<llm::InferenceFeedback, Infallible> {
    match response {
        llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
            print!("{}", t);
            std::io::stdout().flush().unwrap();

            Ok(llm::InferenceFeedback::Continue)
        }
        _ => Ok(llm::InferenceFeedback::Continue),
    }
}

/// Get user input from the console.
fn get_user_input(prompt: &str) -> Result<String> {
    print!("{}", prompt);
    io::stdout().flush().context("Failed to flush stdout")?;

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .context("Failed to read user input")?;

    Ok(input.trim().to_owned())
}
