use anyhow::Result;
use bytes::Bytes;
use clap::Parser;
use h3::{error::ErrorLevel, quic::BidiStream, server::RequestStream};
use h3_quinn::quinn;
use http::{Request, StatusCode};
use quinn::Endpoint;
use rustls::{Certificate, PrivateKey};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::io::AsyncWriteExt;
use tracing::{error, info, trace_span};

#[derive(Parser, Debug)]
#[clap(name = "server")]
pub struct Opt {
    #[clap(
        short,
        long,
        default_value = "[::1]:4433",
        help = "What address:port to listen for new connections"
    )]
    pub listen: SocketAddr,

    // Assuming you have a similar structure for `certs`
    #[clap(flatten)]
    pub certs: Certs,

    #[clap(short, long, default_value = "./data/mytalk.sqlite")]
    pub serverdb_location: String,
}

#[derive(Parser, Debug)]
pub struct Certs {
    #[clap(
        long,
        short,
        default_value = "examples/server.cert",
        help = "Certificate for TLS. If present, `--key` is mandatory."
    )]
    pub cert: PathBuf,

    #[clap(
        long,
        short,
        default_value = "examples/server.key",
        help = "Private key for the certificate."
    )]
    pub key: PathBuf,
}

static ALPN: &[u8] = b"h3";

#[derive(Debug, Default)]
pub struct H3Server {}

impl H3Server {
    pub async fn usercheckprocess() -> Result<()> {
        loop {
            // read string
            println!("q. exit");
            let a: String = text_io::read!("{}\n");
            match a.as_str() {
                "q" => {
                    println!("exit");
                    std::process::exit(0);
                }
                _ => {
                    println!("unknown command");
                }
            }
        }
        #[allow(unreachable_code)]
        Ok(())
    }
    pub async fn usercheck(&mut self) -> Result<()> {
        // spawn thread
        tokio::spawn(Self::usercheckprocess());
        Ok(())
    }
    pub async fn connectionwait(
        h3_conn: &mut h3::server::Connection<h3_quinn::Connection, Bytes>,
    ) -> Result<()> {
        loop {
            match h3_conn.accept().await {
                Ok(Some((req, stream))) => {
                    info!("new request: {:#?}", req);
                    tokio::spawn(Self::handle_request(req, stream));
                }
                Ok(None) => {
                    break;
                }

                Err(err) => {
                    error!("error on accept {}", err);
                    match err.get_error_level() {
                        ErrorLevel::ConnectionError => break,
                        ErrorLevel::StreamError => continue,
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn waitconnection(new_conn: quinn::Connecting) -> Result<()> {
        match new_conn.await {
            Ok(conn) => {
                info!("new connection established");

                let mut h3_conn =
                    h3::server::Connection::new(h3_quinn::Connection::new(conn)).await?;
                Self::connectionwait(&mut h3_conn).await?;
            }
            Err(err) => {
                error!("accepting connection failed: {:?}", err);
            }
        }
        Ok(())
    }
    pub async fn wait(&mut self, endpoint: Endpoint) -> Result<()> {
        while let Some(new_conn) = endpoint.accept().await {
            trace_span!("New connection being attempted");
            tokio::spawn(Self::waitconnection(new_conn));
        }
        endpoint.wait_idle().await;
        Ok(())
    }
    pub async fn process(&mut self) -> Result<()> {
        self.usercheck().await?;
        let opt = Opt::parse();

        let Certs { cert, key } = opt.certs;
        let cert = Certificate(std::fs::read(cert)?);
        let key = PrivateKey(std::fs::read(key)?);

        let mut tls_config = rustls::ServerConfig::builder()
            .with_safe_default_cipher_suites()
            .with_safe_default_kx_groups()
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)?;

        tls_config.max_early_data_size = u32::MAX;
        tls_config.alpn_protocols = vec![ALPN.into()];

        let server_config = quinn::ServerConfig::with_crypto(Arc::new(tls_config));
        let endpoint = quinn::Endpoint::server(server_config, opt.listen)?;

        info!("listening on {}", opt.listen);
        self.wait(endpoint).await?;

        Ok(())
    }

    pub async fn start_server() -> Result<()> {
        info!("starting server");
        let mut server = H3Server::default();
        server.process().await?;
        Ok(())
    }

    async fn handle_request<T>(req: Request<()>, mut stream: RequestStream<T, Bytes>) -> Result<()>
    where
        T: BidiStream<Bytes>,
    {
        info!("user request: {:#?}", req);
        println!("user request: {:?}", req.uri().path());

        let mut out = Vec::new();
        while let Some(mut chunk) = stream.recv_data().await? {
            out.write_all_buf(&mut chunk).await?;
            out.flush().await?;
        }

        let userpath = req.uri().path();
        let mut status = StatusCode::OK;
        let mut responsebytes = Vec::new();
        let dispatchresult = Self::dispatch_message(userpath, &out).await;
        match dispatchresult {
            Ok(value) => {
                responsebytes = value;
            }
            Err(_e) => {
                status = StatusCode::NOT_FOUND;
            }
        }

        let resp = http::Response::builder().status(status).body(()).unwrap();
        stream.send_response(resp).await?;
        stream.send_data(Bytes::from(responsebytes)).await?;
        stream.finish().await?;
        Ok(())
    }

    async fn dispatch_message(userpath: &str, requestdata: &[u8]) -> Result<Vec<u8>> {
        let responsebytes: Vec<u8> = requestdata.to_vec();
        if userpath == "/hello" {
            println!("hello");
        } else {
            anyhow::bail!("unknown path");
        }
        Ok(responsebytes)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut server = H3Server::default();
    server.process().await?;
    Ok(())
}
