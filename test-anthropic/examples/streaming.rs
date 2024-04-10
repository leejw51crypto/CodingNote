use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::io::Write;
#[derive(Serialize, Debug)]
struct MessageRequest {
    model: String,
    max_tokens: i32,
    messages: Vec<Message>,
    stream: bool,
}

#[derive(Serialize, Debug)]
struct Message {
    role: String,
    content: String,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageResponse },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: Value },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: ApiError },
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct MessageResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<Value>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: Usage,
}

// supress not used warning
#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct ContentBlockDelta {
    #[serde(rename = "type")]
    delta_type: String,
    text: String,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct Usage {
    input_tokens: i32,
    output_tokens: i32,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct ApiError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

async fn stream_messages(api_key: &str, request: &MessageRequest) -> Result<()> {
    let client = Client::new();
    let mut response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("Content-Type", "application/json")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "messages-2023-12-15")
        .json(request)
        .send()
        .await?;

    while let Some(chunk) = response.chunk().await? {
        let chunk_str = String::from_utf8_lossy(&chunk);

        for line in chunk_str.lines() {
            if line.starts_with("data:") {
                let json_str = line.trim_start_matches("data:").trim();
                if let Ok(event) = serde_json::from_str::<StreamEvent>(json_str) {
                    match event {
                        StreamEvent::MessageStart { message: _ } => {
                            println!("--------------------------");
                        }
                        StreamEvent::ContentBlockStart {
                            index: _,
                            content_block: _,
                        } => {}
                        StreamEvent::ContentBlockDelta { index: _, delta } => {
                            print!("{}", delta.text);
                            std::io::stdout().flush().unwrap();
                        }
                        StreamEvent::ContentBlockStop { index } => {
                            println!("Content block stopped at index {}", index);
                        }
                        StreamEvent::MessageDelta { delta: _ } => {}
                        StreamEvent::MessageStop => {
                            println!("--------------------------");
                            break;
                        }
                        StreamEvent::Ping => {}
                        StreamEvent::Error { error } => {
                            anyhow::bail!("Error: {:?}", error);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Get the API key from the environment variable
    let api_key = env::var("ANTHROPIC_API_KEY")?;

    // Create a MessageRequest
    let request = MessageRequest {
        model: "claude-3-opus-20240229".to_string(),
        max_tokens: 256,
        messages: vec![Message {
            role: "user".to_string(),
            content: "what is rust lang?".to_string(),
        }],
        stream: true,
    };

    // Stream the messages
    stream_messages(&api_key, &request).await?;

    Ok(())
}
