use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    // Read OpenAI API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("🔑❌ OPENAI_API_KEY not set");

    // Read YAML file
    let yaml_content = fs::read_to_string("input.yaml")?;

    // Check all SQL statements in the YAML
    let result = check_sql_statements(&api_key, &yaml_content).await?;
    println!("🎉✨ Result:\n{}  {}", result.0, result.1);
    Ok(())
}

#[derive(Deserialize, Serialize, Debug)]
struct SqlAnalysisResponse {
    is_safe: bool,
    issues: Vec<String>,
}

async fn check_sql_statements(api_key: &str, yaml_content: &str) -> Result<(bool, String)> {
    let client = reqwest::Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a SQL security expert. 🕵️‍♀️🔒 Analyze the given YAML content containing SQL statements for potential issues such as SQL injection vulnerabilities or bad SQL code. Respond with an analysis for each SQL statement. Your response should be a JSON object with 'is_safe' (boolean) indicating if all statements are safe, and 'issues' (array of strings) listing any potential issues found."
                },
                {
                    "role": "user",
                    "content": format!("🔍🧐 Analyze the SQL statements in this YAML content:\n\n{}", yaml_content)
                }
            ],
            "response_format": { "type": "json_object" }
        }))
        .send()
        .await?;

    let response_body: Value = response.json().await?;

    let content = response_body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("🚫 Invalid response format"))?;

    let analysis: SqlAnalysisResponse = serde_json::from_str(content)?;

    let result_message = if analysis.is_safe {
        "✅🎊 The SQL statements appear to be safe and sound!".to_string()
    } else {
        let issues_str = analysis.issues.join(", ");
        format!("⚠️🚨 Potential issues detected: {}", issues_str)
    };

    Ok((analysis.is_safe, result_message))
}
