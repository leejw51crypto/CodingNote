use anyhow::Result;

pub mod messagepack;
pub mod protobuf;
pub mod capnp;
pub mod definition;

fn main() -> Result<()> {
   crate::capnp::main()?;
    println!("----------------------");
    crate::messagepack::main()?;
    println!("----------------------");
    crate::protobuf::main()?;
    Ok(())
}
