use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Address to bind (e.g., "127.0.0.1:8080")
    #[clap(short, long, default_value = "0.0.0.0:8080")]
    address: String,
}

async fn store(bytes: web::Bytes) -> impl Responder {
    // If you need to convert bytes back to String, use the following line:
    let text = String::from_utf8_lossy(&bytes);
    println!("Received {} bytes", bytes.len());
    println!("text {}", text);

    HttpResponse::Ok().body(bytes)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    HttpServer::new(|| App::new().route("/store", web::post().to(store)))
        .bind(&args.address)?
        .run()
        .await
}
