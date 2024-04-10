use serde::{Deserialize, Serialize};
use std::env;

#[allow(dead_code)]
#[derive(Serialize)]
struct MessageRequest {
    model: String,
    max_tokens: i32,
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct MessageResponse {
    id: String,
    content: Vec<Content>,
    model: String,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: Usage,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Content {
    text: Option<String>,
    id: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Usage {
    input_tokens: i32,
    output_tokens: i32,
}

#[tokio::main]
async fn main() {
    // Get the API key from the environment variable
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");

    // Create a MessageRequest
    let request = MessageRequest {
        // claude-3-opus-20240229
        // claude-3-sonnet-20240229
        // claude-3-haiku-20240307
        model: "claude-3-haiku-20240307".to_string(),
        max_tokens: 1024,
        messages: vec![Message {
            role: "user".to_string(),
            content: "what is rust lang?".to_string(),
        }],
    };

    // Send the request and get the response
    let client = reqwest::Client::new();
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("Content-Type", "application/json")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&request)
        .send()
        .await
        .unwrap();

    // Parse the response JSON into a MessageResponse struct
    let response_data: MessageResponse = response.json().await.unwrap();

    // Print the response content
    for content in response_data.content {
        if let Some(text) = content.text {
            println!("{}", text);
        }
    }
}
