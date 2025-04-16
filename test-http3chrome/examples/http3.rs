use bytes::Bytes;
use h3::quic::BidiStream;
use h3::server::RequestStream;
use h3_quinn::quinn::crypto::rustls::QuicServerConfig;
use http::{Response, StatusCode};
use hyper::body::{Body, Incoming};
use hyper::server::conn::http1;
use hyper_util::rt::TokioIo;
use quinn::{Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::fs;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use tracing::error;

// ALPN protocols
static ALPN_H3: &[u8] = b"h3";
static ALPN_H1: &[u8] = b"http/1.1";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Load TLS certificates and private key
    let cert_bytes = fs::read("cert.pem").await?;
    let key_bytes = fs::read("key.pem").await?;

    // Parse certificates and private key
    let cert = rustls_pemfile::certs(&mut &cert_bytes[..])
        .next()
        .ok_or("No certificate found")?
        .map_err(|e| format!("Failed to parse certificate: {}", e))?;
    let key = rustls_pemfile::pkcs8_private_keys(&mut &key_bytes[..])
        .next()
        .ok_or("No private key found")?
        .map_err(|e| format!("Failed to parse private key: {}", e))?;

    let cert = CertificateDer::from(cert);
    let key = PrivateKeyDer::from(key);
    let key_clone = key.clone_key();

    // Create HTTP/3 TLS config
    let mut h3_tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert.clone()], key)?;
    h3_tls_config.alpn_protocols = vec![ALPN_H3.to_vec()];
    h3_tls_config.max_early_data_size = u32::MAX;

    // HTTP/1.1 TLS config
    let mut h1_tls_config = tokio_rustls::rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert], key_clone)?;
    h1_tls_config.alpn_protocols = vec![ALPN_H1.to_vec()];
    let h1_tls_acceptor = TlsAcceptor::from(Arc::new(h1_tls_config));

    // Create HTTP/3 server config
    let h3_server_config =
        ServerConfig::with_crypto(Arc::new(QuicServerConfig::try_from(h3_tls_config)?));

    // Create HTTP/3 endpoint
    let h3_addr = "[::]:4433".parse::<SocketAddr>()?;
    let h3_endpoint = Endpoint::server(h3_server_config, h3_addr)?;
    println!("HTTP/3 server listening on {}", h3_addr);

    // Create HTTP/1.1 endpoint
    let h1_addr = "[::]:4433".parse::<SocketAddr>()?;
    let h1_listener = TcpListener::bind(h1_addr).await?;
    println!("HTTP/1.1 server listening on {}", h1_addr);
    println!("Try visiting: https://localhost:4433/hello");

    // Run both servers concurrently
    tokio::select! {
        res = run_h3_server(h3_endpoint) => {
            if let Err(e) = res {
                error!("HTTP/3 server error: {}", e);
            }
        }
        res = run_h1_server(h1_listener, h1_tls_acceptor) => {
            if let Err(e) = res {
                error!("HTTP/1.1 server error: {}", e);
            }
        }
    }

    Ok(())
}

async fn run_h3_server(endpoint: Endpoint) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    while let Some(conn) = endpoint.accept().await {
        println!("New HTTP/3 connection attempt received!");
        tokio::spawn(async move {
            match conn.await {
                Ok(connection) => {
                    println!("HTTP/3 connection established successfully!");
                    if let Err(e) = handle_h3_connection(connection).await {
                        error!("Connection error: {}", e);
                    }
                }
                Err(e) => {
                    error!("Failed to establish connection: {}", e);
                }
            }
        });
    }
    Ok(())
}

async fn run_h1_server(
    listener: TcpListener,
    tls_acceptor: TlsAcceptor,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    loop {
        let (stream, _) = listener.accept().await?;
        let acceptor = tls_acceptor.clone();

        tokio::spawn(async move {
            match acceptor.accept(stream).await {
                Ok(tls_stream) => {
                    let io = TokioIo::new(tls_stream);

                    let service = hyper::service::service_fn(|req| async {
                        Ok::<_, Infallible>(handle_h1_request(req))
                    });

                    if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                        error!("Error serving connection: {:?}", err);
                    }
                }
                Err(e) => error!("TLS error: {:?}", e),
            }
        });
    }
}

async fn handle_h3_connection(
    connection: quinn::Connection,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Starting H3 connection setup...");
    let mut h3_conn = h3::server::Connection::new(h3_quinn::Connection::new(connection)).await?;
    println!("H3 connection setup complete!");

    while let Ok(Some((req, stream))) = h3_conn.accept().await {
        tokio::spawn(async move {
            if let Err(e) = handle_h3_request(req, stream).await {
                error!("Error handling request: {}", e);
            }
        });
    }

    Ok(())
}

async fn handle_h3_request<T>(
    req: http::Request<()>,
    mut stream: RequestStream<T, Bytes>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    T: BidiStream<Bytes>,
{
    println!("Received HTTP/3 request: {} {}", req.method(), req.uri());

    let response = if req.uri().path() == "/hello" {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/plain")
            .body(())?
    } else {
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "text/plain")
            .body(())?
    };

    stream.send_response(response).await?;

    if req.uri().path() == "/hello" {
        stream.send_data(Bytes::from("Hello from HTTP/3!")).await?;
    } else {
        stream.send_data(Bytes::from("404 Not Found")).await?;
    }

    stream.finish().await?;
    Ok(())
}

fn handle_h1_request(req: hyper::Request<Incoming>) -> hyper::Response<String> {
    println!("Received HTTP/1.1 request: {} {}", req.method(), req.uri());

    if req.uri().path() == "/hello" {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/plain")
            .header("alt-svc", "h3=\":4433\"; ma=3600")
            .body(String::from(concat!(
                "Hello from HTTP/1.1!\n\n",
                "To enable HTTP/3 with self-signed certificate:\n",
                "1. Visit chrome://flags/#allow-insecure-localhost\n",
                "2. Enable 'Allow invalid certificates for resources loaded from localhost'\n",
                "3. Visit chrome://flags/#enable-quic\n",
                "4. Enable 'Experimental QUIC protocol'\n",
                "5. Restart Chrome\n",
                "6. Try https://localhost:4433/hello directly\n"
            )))
            .unwrap_or_else(|_| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(String::from("Internal Server Error"))
                    .unwrap_or_default()
            })
    } else {
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "text/plain")
            .header("alt-svc", "h3=\":4433\"; ma=3600")
            .body(String::from("404 Not Found"))
            .unwrap_or_else(|_| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(String::from("Internal Server Error"))
                    .unwrap_or_default()
            })
    }
}
