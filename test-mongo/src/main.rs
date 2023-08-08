use anyhow::Result;
use dotenv::dotenv;
use mongodb::{
    bson::{doc, Document},
    options::ClientOptions,
    Client,
};
use std::env;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables and establish a MongoDB client
    dotenv().ok();
    let client = establish_mongo_client().await?;

    // Fetch and display documents from the collection
    fetch_and_display_documents(&client).await?;

    Ok(())
}

/// Establishes a MongoDB client based on environment configuration
async fn establish_mongo_client() -> Result<Client> {
    let mongo_uri = env::var("MONGOURI").expect("MONGOURI not found");

    let mut client_options = ClientOptions::parse(&mongo_uri).await?;
    client_options.app_name = Some("MyAppName".to_string());

    Ok(Client::with_options(client_options)?)
}

/// Fetches documents from the specified collection and displays them
async fn fetch_and_display_documents(client: &Client) -> Result<()> {
    let db_name = "sampleDB";
    println!("db_name: {}", db_name);
    let db = client.database(db_name);
    let coll = db.collection::<Document>("sampleCollection");

    let mut cursor = coll.find(None, None).await?;

    while let Some(result) = cursor.next().await {
        match result {
            Ok(document) => display_document(&document),
            Err(e) => println!("Error: {}", e),
        }
    }

    Ok(())
}

/// Displays the 'name' field from a document if it exists
fn display_document(document: &Document) {
    if let Some(name) = document.get("name").and_then(|v| v.as_str()) {
        println!("Found: {}", name);
    } else {
        println!("No name found!");
    }
}
