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

// The store function now takes a String directly from the request body
async fn store(text: String) -> impl Responder {
    HttpResponse::Ok().body(text)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    HttpServer::new(|| App::new().route("/store", web::post().to(store)))
        .bind(&args.address)?
        .run()
        .await
}
