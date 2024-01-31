use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use clap::Parser;
use serde::Deserialize;
// Define a struct to deserialize the query parameters
#[derive(Deserialize)]
struct QueryInfo {
    text: String,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Address to bind (e.g., "127.0.0.1:8080")
    #[clap(short, long, default_value = "0.0.0.0:8080")]
    address: String,
}

async fn store(info: web::Query<QueryInfo>) -> impl Responder {
    HttpResponse::Ok().body(info.text.clone())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    HttpServer::new(|| App::new().route("/store", web::get().to(store)))
        .bind(&args.address)?
        .run()
        .await
}
