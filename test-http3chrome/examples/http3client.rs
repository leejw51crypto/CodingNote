use bytes::{Buf, Bytes};
use futures::future;
use h3::client;
use h3_quinn::quinn::crypto::rustls::QuicClientConfig;
use http::Request;
use quinn::{ClientConfig, Endpoint};
use rustls::Error;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{
    RootCertStore,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
};
use std::error::Error as StdError;
use std::sync::Arc;
use tokio::fs;
use tracing::{Level, debug, error, info};

static ALPN: &[u8] = b"h3";

#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn StdError + Send + Sync>> {
    // Enable logging with debug level
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .pretty()
        .init();

    println!("🚀 Starting HTTP/3 client...");

    // Create TLS config with dangerous options for development
    let mut tls_config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    tls_config.enable_early_data = true;
    tls_config.alpn_protocols = vec![ALPN.to_vec()];
    println!("✅ TLS config created");

    // Create client config
    let client_config = ClientConfig::new(Arc::new(
        QuicClientConfig::try_from(tls_config)
            .map_err(|e| format!("Failed to create QUIC client config: {}", e))?,
    ));
    println!("✅ Client config created");

    // Create endpoint
    let mut endpoint = Endpoint::client(
        "0.0.0.0:0"
            .parse()
            .map_err(|e| format!("Failed to parse client address: {}", e))?,
    )?;
    endpoint.set_default_client_config(client_config);
    println!("✅ Endpoint created");

    // Connect to server
    println!("🔄 Connecting to server at 127.0.0.1:4433...");
    let connection = endpoint
        .connect(
            "127.0.0.1:4433"
                .parse()
                .map_err(|e| format!("Failed to parse server address: {}", e))?,
            "localhost",
        )?
        .await
        .map_err(|e| format!("Failed to establish connection: {}", e))?;
    println!("✅ Connected successfully!");

    // Create H3 connection
    println!("🔄 Building H3 connection...");
    let (mut driver, mut h3_conn) = client::builder()
        .build::<_, _, Bytes>(h3_quinn::Connection::new(connection))
        .await
        .map_err(|e| format!("Failed to build H3 connection: {}", e))?;
    println!("✅ H3 connection built");

    // Create a task to drive the connection
    let drive_task = tokio::spawn(async move {
        println!("🔄 Connection driver started");
        let result = future::poll_fn(|cx| driver.poll_close(cx)).await;
        println!("Connection driver finished");
        result
    });

    // Create request
    let req = Request::builder()
        .method("GET")
        .uri("https://localhost:4433/hello")
        .body(())
        .map_err(|e| format!("Failed to build request: {}", e))?;

    // Send request
    println!("📤 Sending request to /hello...");
    let mut stream = h3_conn
        .send_request(req)
        .await
        .map_err(|e| format!("Failed to send request: {}", e))?;
    println!("✅ Request sent successfully!");

    // Get response
    println!("🔄 Waiting for response...");
    let response = stream
        .recv_response()
        .await
        .map_err(|e| format!("Failed to receive response: {}", e))?;
    println!("📥 Response received!");
    println!(
        "Status: {} {}",
        response.status().as_u16(),
        response.status().canonical_reason().unwrap_or("")
    );

    // Print response headers
    println!("Headers:");
    for (name, value) in response.headers() {
        println!("  {}: {}", name, value.to_str().unwrap_or("<binary>"));
    }

    // Read and accumulate response data
    println!("🔄 Reading response body...");
    let mut body = Vec::new();
    while let Some(bytes) = stream
        .recv_data()
        .await
        .map_err(|e| format!("Failed to receive data: {}", e))?
    {
        println!("Received chunk of {} bytes", bytes.chunk().len());
        body.extend_from_slice(bytes.chunk());
    }

    // Display the complete response body
    println!("📝 Response body:");
    match String::from_utf8(body) {
        Ok(text) => println!("{}", text),
        Err(e) => println!("Failed to decode response as UTF-8: {}", e),
    }

    println!("🔄 Closing connection...");
    drop(stream);
    drop(h3_conn);

    // Wait for the driver task
    if let Err(e) = drive_task
        .await
        .map_err(|e| format!("Failed to join driver task: {}", e))?
    {
        println!("❌ Driver error: {}", e);
    }

    // Wait for the endpoint to be idle and then close it
    endpoint.wait_idle().await;
    endpoint.close(0u32.into(), b"Done");

    println!("✅ Connection closed successfully!");
    Ok(())
}
