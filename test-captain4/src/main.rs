use anyhow::Result;

pub mod capnp;
pub mod definition;
pub mod messagepack;
pub mod mycode;
pub mod protobuf;

fn main() -> Result<()> {
    crate::capnp::main()?;
    println!("----------------------");
    crate::messagepack::main()?;
    println!("----------------------");
    crate::protobuf::main()?;
    Ok(())
}
