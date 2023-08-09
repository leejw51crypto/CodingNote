mod hello_capnp {
    include!(concat!(env!("OUT_DIR"), "/proto/hello_capnp.rs"));
}

use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use std::net::ToSocketAddrs;

use anyhow::Result;
use futures::AsyncReadExt;

#[tokio::main]
pub async fn main() -> Result<()> {
    let args: Vec<String> = ::std::env::args().collect();
    if args.len() != 3 {
        println!("usage: {} client HOST:PORT MESSAGE", args[0]);
        return Ok(());
    }

    let addr = args[1]
        .to_socket_addrs()?
        .next()
        .expect("could not parse address");

    let msg = args[2].to_string();

    tokio::task::LocalSet::new()
        .run_until(async move {
            let stream = tokio::net::TcpStream::connect(&addr).await?;
            stream.set_nodelay(true)?;
            let (reader, writer) =
                tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
            let rpc_network = Box::new(twoparty::VatNetwork::new(
                reader,
                writer,
                rpc_twoparty_capnp::Side::Client,
                Default::default(),
            ));
            let mut rpc_system = RpcSystem::new(rpc_network, None);
            let hello_world: crate::hello_capnp::hello_world::Client =
                rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);

            tokio::task::spawn_local(rpc_system);

            let mut request = hello_world.say_hello2_request();
            request.get().init_request().set_name(&msg);

            let reply = request.send().promise.await?;

            println!("received: {}", reply.get()?.get_reply()?.get_message()?);
            Ok(())
        })
        .await
}
