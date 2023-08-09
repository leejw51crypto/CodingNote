use capnp::capability::Promise;
use capnp_rpc::{pry, rpc_twoparty_capnp, twoparty, RpcSystem};

mod hello_capnp {
    include!(concat!(env!("OUT_DIR"), "/proto/hello_capnp.rs"));
}

use crate::hello_capnp::hello_world::Client;
use crate::hello_capnp::hello_world::Server;
use crate::hello_capnp::hello_world::{SayHelloParams, SayHelloResults};
use anyhow::Result;
use futures::AsyncReadExt;
use std::net::ToSocketAddrs;
struct HelloWorldImpl;

impl Server for HelloWorldImpl {
    fn say_hello(
        &mut self,
        params: SayHelloParams,
        mut results: SayHelloResults,
    ) -> Promise<(), ::capnp::Error> {
        let request = pry!(pry!(params.get()).get_request());
        let name = pry!(request.get_name());
        let message = format!("Hello, {name}!");

        results.get().init_reply().set_message(&message);

        Promise::ok(())
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args: Vec<String> = ::std::env::args().collect();
    if args.len() != 2 {
        println!("usage: {} server ADDRESS[:PORT]", args[0]);
        return Ok(());
    }

    let addr = args[1]
        .to_socket_addrs()?
        .next()
        .expect("could not parse address");

    tokio::task::LocalSet::new()
        .run_until(async move {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            let hello_world_client: Client = capnp_rpc::new_client(HelloWorldImpl);

            loop {
                let (stream, _) = listener.accept().await?;
                stream.set_nodelay(true)?;
                let (reader, writer) =
                    tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
                let network = twoparty::VatNetwork::new(
                    reader,
                    writer,
                    rpc_twoparty_capnp::Side::Server,
                    Default::default(),
                );

                let rpc_system =
                    RpcSystem::new(Box::new(network), Some(hello_world_client.clone().client));

                tokio::task::spawn_local(rpc_system);
            }
        })
        .await
}
