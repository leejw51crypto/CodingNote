use anyhow::Result;
use capnp::message::ReaderOptions;
use capnp::serialize::{read_message, write_message};
use capnp::{message, serialize};
use sha2::Digest;
use std::io::{Read, Write};
pub mod blockchain_capnp {
    include!(concat!(env!("OUT_DIR"), "/proto/blockchain_capnp.rs"));
}

fn create_block() -> message::Builder<capnp::message::HeapAllocator> {
    let mut messageblock = capnp::message::Builder::new_default();
    let mut block = messageblock.init_root::<blockchain_capnp::block::Builder>();

    block.set_timestamp(1627932234);
    block.set_prev_hash(&[1, 2, 3, 4]);

    let mut transactions = block.reborrow().init_transactions(2);

    let mut transaction1 = transactions.reborrow().get(0);
    transaction1.set_sender("Alice");
    transaction1.set_recipient("Bob");
    transaction1.set_amount(100);

    let mut transaction2 = transactions.get(1);
    transaction2.set_sender("Bob");
    transaction2.set_recipient("Charlie");
    transaction2.set_amount(50);

    messageblock
}

fn serialize_block(
    mut messageblock: message::Builder<capnp::message::HeapAllocator>,
) -> Result<Vec<u8>> {
    let mut block_out = Vec::new();
    write_message(&mut block_out, &messageblock)?;
    Ok(block_out)
}
fn write() -> Result<Vec<u8>> {
    let mut message = message::Builder::new_default();
    let mut blockchain = message.init_root::<blockchain_capnp::blockchain::Builder>();
    let mut finalblocks = blockchain.init_blocks(3);

    // first block
    let mut finalblock = finalblocks.reborrow().get(0);

    let messageblock = create_block();
    let block_out = serialize_block(messageblock)?;

    let hash = sha2::Sha256::digest(&block_out);
    finalblock.set_hash(&hash);
    finalblock.set_block(&block_out);

    // second block
    let mut finalblock = finalblocks.reborrow().get(1);
    let hash = sha2::Sha256::digest(&block_out);
    finalblock.set_hash(&hash);
    finalblock.set_block(&block_out);

    let mut finalblock = finalblocks.reborrow().get(2);
    let hash = sha2::Sha256::digest(&block_out);
    finalblock.set_hash(&hash);
    finalblock.set_block(&block_out);

    let mut out = Vec::new();
    write_message(&mut out, &message)?;
    Ok(out)
}

fn read(input: &[u8]) -> Result<()> {
    let mut reader = read_message(&mut std::io::Cursor::new(input), ReaderOptions::new())?;
    let blockchain = reader.get_root::<blockchain_capnp::blockchain::Reader>()?;

    // show block count
    println!("Block Count: {}", blockchain.get_blocks()?.len());
    for block2 in blockchain.get_blocks()?.iter() {
        let hash = block2.get_hash()?;
        println!("Hash: {:?}", hash);

        let block_data = block2.get_block()?;
        let mut block_reader =
            read_message(&mut std::io::Cursor::new(block_data), ReaderOptions::new())?;
        let block = block_reader.get_root::<blockchain_capnp::block::Reader>()?;

        println!("Timestamp: {}", block.get_timestamp());
        println!("Prev Hash: {:?}", block.get_prev_hash()?);
        println!("Transactions: Count {}", block.get_transactions()?.len());
        for transaction in block.get_transactions()?.iter() {
            println!(
                "Transaction - Sender: {}, Recipient: {}, Amount: {}",
                transaction.get_sender()?,
                transaction.get_recipient()?,
                transaction.get_amount()
            );
        }
    }

    Ok(())
}

fn main2() -> Result<()> {
    let out = write()?;
    println!("Serialized data: {}", hex::encode(&out));

    read(&out)?;
    Ok(())
}

fn main() -> Result<()> {
    let mut out = Vec::new();
    {
        let mut message = message::Builder::new_default();
        let mut tx = message.init_root::<blockchain_capnp::transaction::Builder>();
        tx.set_sender("Alice");
        tx.set_recipient("Bob");
        tx.set_amount(100);

        write_message(&mut out, &message)?;
        println!("TX Serialized data: {}", hex::encode(&out));
    }

    {
        let reader = read_message(&mut std::io::Cursor::new(&out), ReaderOptions::new())?;
        let tx = reader.get_root::<blockchain_capnp::transaction::Reader>()?;
        println!(
            "TX: Sender: {}, Recipient: {}, Amount: {}",
            tx.get_sender()?,
            tx.get_recipient()?,
            tx.get_amount()
        );
    }

    {
        let reader = read_message(&mut std::io::Cursor::new(&out), ReaderOptions::new())?;
        let tx = reader.get_root::<blockchain_capnp::transaction::Reader>()?;
        let amount = tx.get_amount();
        println!("tx amount: {}", amount);
    }
    Ok(())
}
