use std::{path::PathBuf, sync::Arc};

use anyhow::{anyhow, Result};
use capnp::message::Builder;
use clap::Parser;
use fake::faker::address::raw::*;
use fake::faker::name::raw::*;
use fake::locales::*;
use fake::Fake;
use futures::future;
use h3_quinn::quinn;
use hello_capnp::{people, person};
use quinn::Endpoint;
use rand::Rng;
use rustls::RootCertStore;
use tokio::io::AsyncWriteExt;
use tracing::{error, info};

static ALPN: &[u8] = b"h3";
pub mod hello_capnp {
    include!(concat!(env!("OUT_DIR"), "/proto/hello_capnp.rs"));
}

#[derive(Parser, Debug)]
#[clap(name = "server")]
pub struct Opt {
    #[clap(
        long,
        short = 'c',
        default_value = "examples/ca.cert",
        help = "Certificate of CA who issues the server certificate"
    )]
    pub ca: PathBuf,

    #[clap(name = "keylogfile", long)]
    pub key_log_file: bool,

    #[clap()]
    pub uri: String,

    #[clap(long = "wallet", default_value = "./wallets/mywallet.txt")]
    pub wallet_location: String,

    #[clap(long = "clientdb", default_value = "./data/myclient.sqlite")]
    pub clientdb_location: String,
}

#[derive(Debug, Default)]
pub struct H3Client {}

impl H3Client {
    pub async fn perform_http_request(apipath: &str, datatosend: &[u8]) -> Result<Vec<u8>> {
        let retbytes = Self::perform_http_request_core(apipath, datatosend).await;
        match retbytes {
            Ok(retbytes) => Ok(retbytes),
            Err(e) => Err(anyhow!("perform_http_request error: {:?}", e)),
        }
    }
    pub fn get_cert_roots() -> Result<RootCertStore> {
        let opt = Opt::parse();
        let mut roots = rustls::RootCertStore::empty();
        match rustls_native_certs::load_native_certs() {
            Ok(certs) => {
                for cert in certs {
                    if let Err(e) = roots.add(&rustls::Certificate(cert.0)) {
                        error!("failed to parse trust anchor: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("couldn't load any default trust roots: {}", e);
            }
        };
        roots.add(&rustls::Certificate(std::fs::read(opt.ca)?))?;
        Ok(roots)
    }
    pub fn get_client_endpoint() -> Result<Endpoint> {
        let opt = Opt::parse();
        let roots = Self::get_cert_roots()?;

        let mut tls_config = rustls::ClientConfig::builder()
            .with_safe_default_cipher_suites()
            .with_safe_default_kx_groups()
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_root_certificates(roots)
            .with_no_client_auth();

        tls_config.enable_early_data = true;
        tls_config.alpn_protocols = vec![ALPN.into()];

        if opt.key_log_file {
            tls_config.key_log = Arc::new(rustls::KeyLogFile::new());
        }

        let mut client_endpoint = h3_quinn::quinn::Endpoint::client("[::]:0".parse()?)?;

        let client_config = quinn::ClientConfig::new(Arc::new(tls_config));
        client_endpoint.set_default_client_config(client_config);
        Ok(client_endpoint)
    }
    pub async fn perform_http_request_core(
        apipath: &str,
        datatosend: &[u8],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let opt = Opt::parse();
        let fulluri = format!("{}{}", opt.uri, apipath);
        let uri = fulluri.parse::<http::Uri>()?;
        if uri.scheme() != Some(&http::uri::Scheme::HTTPS) {
            Err("uri scheme must be 'https'")?;
        }
        let auth = uri.authority().ok_or("uri must have a host")?.clone();
        let port = auth.port_u16().unwrap_or(443);
        let addr = tokio::net::lookup_host((auth.host(), port))
            .await?
            .next()
            .ok_or("dns found no addresses")?;

        info!("DNS lookup for {:?}: {:?}", uri, addr);
        let client_endpoint = Self::get_client_endpoint()?;

        let conn = client_endpoint.connect(addr, auth.host())?.await?;
        info!("QUIC connection established");
        let quinn_conn = h3_quinn::Connection::new(conn);
        let (mut driver, mut send_request) = h3::client::new(quinn_conn).await?;
        let drive = async move {
            future::poll_fn(|cx| driver.poll_close(cx)).await?;
            Ok::<(), Box<dyn std::error::Error>>(())
        };

        let request = async move {
            info!("sending request ...");
            // print uri
            info!("uri: {}", uri);

            let req = http::Request::builder().uri(uri).body(())?;
            let mut stream = send_request.send_request(req).await?;
            stream
                .send_data(bytes::Bytes::from(datatosend.to_vec()))
                .await?;
            stream.finish().await?;

            let resp = stream.recv_response().await?;
            info!("received response: {:?}", resp);
            // read stream
            let mut outbytes = Vec::new();

            while let Some(mut chunk) = stream.recv_data().await? {
                outbytes.write_all_buf(&mut chunk).await?;
                outbytes.flush().await?;
            }

            Ok::<Vec<u8>, Box<dyn std::error::Error>>(outbytes)
        };

        let (req_res, drive_res) = tokio::join!(request, drive);
        let retbytes = req_res?;
        drive_res?;
        println!("retbytes: {}", retbytes.len());
        // wait for the connection to be closed before exiting
        client_endpoint.wait_idle().await;

        Ok(retbytes)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();
    let datatosend = writecapnp()?;
    let ret = H3Client::perform_http_request("/hello", &datatosend)
        .await?
        .to_vec();
    readcapnp(&ret)?;
    Ok(())
}

fn writecapnp() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut message = Builder::new_default();
    {
        let mut people = message.init_root::<people::Builder>();

        // Generate a random number of persons
        let num_persons = rand::thread_rng().gen_range(1900..20000);
        let mut persons = people.reborrow().init_persons(num_persons as u32);

        for i in 0..num_persons {
            let mut person = persons.reborrow().get(i as u32);

            let name: String = Name(EN).fake();
            let myaddress: String = SecondaryAddress(EN).fake();
            let mycity: String = CityName(EN).fake();
            let mypostcode: String = PostCode(EN).fake();
            let age = rand::thread_rng().gen_range(0..100) as i32;

            person.set_name(&name);
            person.set_age(age);
            // make empty byte array of 100,000 bytes
            let photobytes: Vec<u8> = vec![0; 1000];
            person.set_photo(&photobytes);

            let mut address = person.init_address();
            address.set_street(&myaddress);
            address.set_city(&mycity);
            address.set_zip(&mypostcode);
        }
    }

    let mut output = Vec::new();
    capnp::serialize::write_message(&mut output, &message)?;

    Ok(output)
}

fn show_person(person: person::Reader) -> Result<(), capnp::Error> {
    println!("Name: {}", person.get_name()?);
    println!("Age: {}", person.get_age());
    if let Ok(address) = person.get_address() {
        println!("Street: {}", address.get_street()?);
        println!("City: {}", address.get_city()?);
        println!("ZIP: {}", address.get_zip()?);
    }
    Ok(())
}

fn readcapnp(output: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let message_reader =
        capnp::serialize::read_message(&mut &output[..], capnp::message::ReaderOptions::new())?;
    let people_reader = message_reader.get_root::<people::Reader>()?;
    let persons = people_reader.get_persons()?;
    let num_persons = persons.len();

    show_person(persons.get(0))?;
    println!("-------------------");
    show_person(persons.get(num_persons - 1))?;
    println!("Number of persons: {}", people_reader.get_persons()?.len());
    println!("deserialize {} bytes", output.len());

    Ok(())
}
