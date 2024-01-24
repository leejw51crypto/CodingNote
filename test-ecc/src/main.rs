use anyhow::Result;
use crypto::buffer::{BufferResult, ReadBuffer, WriteBuffer};
use crypto::{aes, blockmodes, buffer, symmetriccipher};
use ecies::{decrypt, encrypt};
use ethers::{
    prelude::*,
    signers::{coins_bip39::English, MnemonicBuilder},
};
use k256::{
    ecdsa::SigningKey,
    elliptic_curve::generic_array::{typenum::U32, GenericArray},
};

use rand::rngs::OsRng;
use rand::RngCore;
#[tokio::main]
async fn main() -> Result<()> {
    // Retrieve mnemonic from environment variable
    let mymnemonics = std::env::var("MY_MNEMONICS").expect("MY_MNEMONICS must be set");

    // Create a wallet from the mnemonic
    let wallet = MnemonicBuilder::<English>::default()
        .phrase(mymnemonics.as_str())
        .index(0u32)?
        .build()
        .expect("Failed to build wallet");

    // Generate Ethereum address from wallet
    let eth_address = wallet.address();
    println!("Ethereum Address: {:?}", eth_address);

    // Message to encrypt
    let message = b"Hello, ECC encryption!";
    println!(
        "Original Message: {}",
        std::str::from_utf8(message).unwrap()
    );
    println!("Original Message: {}", hex::encode(message));
    println!("--------------------------");
    // Convert Ethereum private key to a compatible format for ECIES
    let secret_key_bytes: Vec<u8> = wallet.signer().to_bytes().to_vec();
    let secret_key_bytes_array: GenericArray<u8, U32> =
        GenericArray::clone_from_slice(&secret_key_bytes);
    let secret_key = SigningKey::from_bytes(&secret_key_bytes_array).expect("Failed to create key");

    // Derive public key for encryption
    let public_key = secret_key.verifying_key();
    let public_key_bytes = public_key.to_encoded_point(false).to_bytes();

    let mut symmetrickey: [u8; 32] = [0; 32];
    OsRng.fill_bytes(&mut symmetrickey);
    println!("SymmetricKey: {}", hex::encode(&symmetrickey));

    let myencrypteddata = mysymmetricencrypt(message, &symmetrickey.to_vec())?;
    println!("Encrypted Message: {}", hex::encode(&myencrypteddata));

    // Encrypt the message
    let encrypted_message = encrypt(&public_key_bytes, &symmetrickey).expect("Encryption failed");
    println!(
        "Encrypted SymmetricKey: {}",
        hex::encode(&encrypted_message)
    );

    println!("--------------------------");

    // Decrypt the message
    let decrypted_key =
        decrypt(secret_key.to_bytes().as_ref(), &encrypted_message).expect("Decryption failed");
    println!("Decrypted SymmetricKey: {}", hex::encode(&decrypted_key));

    let mydecrypteddata = mysymmetricdecrypt(&myencrypteddata, &decrypted_key)?;
    println!("Decrypted Message: {}", hex::encode(&mydecrypteddata));
    // print the decrypted message as string
    println!(
        "Decrypted message: {}",
        std::str::from_utf8(&mydecrypteddata).unwrap()
    );

    Ok(())
}

// Encrypt a buffer with the given key and iv using
// AES-256/CBC/Pkcs encryption.
fn symmetricencrypt(
    data: &[u8],
    key: &[u8],
    iv: &[u8],
) -> Result<Vec<u8>, symmetriccipher::SymmetricCipherError> {
    // Create an encryptor instance of the best performing
    // type available for the platform.
    let mut encryptor =
        aes::cbc_encryptor(aes::KeySize::KeySize256, key, iv, blockmodes::PkcsPadding);

    // Each encryption operation encrypts some data from
    // an input buffer into an output buffer. Those buffers
    // must be instances of RefReaderBuffer and RefWriteBuffer
    // (respectively) which keep track of how much data has been
    // read from or written to them.
    let mut final_result = Vec::<u8>::new();
    let mut read_buffer = buffer::RefReadBuffer::new(data);
    let mut buffer = [0; 4096];
    let mut write_buffer = buffer::RefWriteBuffer::new(&mut buffer);

    // Each encryption operation will "make progress". "Making progress"
    // is a bit loosely defined, but basically, at the end of each operation
    // either BufferUnderflow or BufferOverflow will be returned (unless
    // there was an error). If the return value is BufferUnderflow, it means
    // that the operation ended while wanting more input data. If the return
    // value is BufferOverflow, it means that the operation ended because it
    // needed more space to output data. As long as the next call to the encryption
    // operation provides the space that was requested (either more input data
    // or more output space), the operation is guaranteed to get closer to
    // completing the full operation - ie: "make progress".
    //
    // Here, we pass the data to encrypt to the enryptor along with a fixed-size
    // output buffer. The 'true' flag indicates that the end of the data that
    // is to be encrypted is included in the input buffer (which is true, since
    // the input data includes all the data to encrypt). After each call, we copy
    // any output data to our result Vec. If we get a BufferOverflow, we keep
    // going in the loop since it means that there is more work to do. We can
    // complete as soon as we get a BufferUnderflow since the encryptor is telling
    // us that it stopped processing data due to not having any more data in the
    // input buffer.
    loop {
        let result = encryptor.encrypt(&mut read_buffer, &mut write_buffer, true)?;

        // "write_buffer.take_read_buffer().take_remaining()" means:
        // from the writable buffer, create a new readable buffer which
        // contains all data that has been written, and then access all
        // of that data as a slice.
        final_result.extend(
            write_buffer
                .take_read_buffer()
                .take_remaining()
                .iter()
                .map(|&i| i),
        );

        match result {
            BufferResult::BufferUnderflow => break,
            BufferResult::BufferOverflow => {}
        }
    }

    Ok(final_result)
}

fn symmetricdecrypt(
    encrypted_data: &[u8],
    key: &[u8],
    iv: &[u8],
) -> Result<Vec<u8>, symmetriccipher::SymmetricCipherError> {
    let mut decryptor =
        aes::cbc_decryptor(aes::KeySize::KeySize256, key, iv, blockmodes::PkcsPadding);

    let mut final_result = Vec::<u8>::new();
    let mut read_buffer = buffer::RefReadBuffer::new(encrypted_data);
    let mut buffer = [0; 4096];
    let mut write_buffer = buffer::RefWriteBuffer::new(&mut buffer);

    loop {
        let result = decryptor.decrypt(&mut read_buffer, &mut write_buffer, true)?;
        final_result.extend(
            write_buffer
                .take_read_buffer()
                .take_remaining()
                .iter()
                .map(|&i| i),
        );
        match result {
            BufferResult::BufferUnderflow => break,
            BufferResult::BufferOverflow => {}
        }
    }

    Ok(final_result)
}

fn mysymmetricencrypt(message: &[u8], key: &[u8]) -> Result<Vec<u8>> {
    let iv: [u8; 16] = [0; 16];
    let encrypted_data = symmetricencrypt(message, &key, &iv)
        .map_err(|e| anyhow::anyhow!("encrypt error: {:?}", e))?;
    Ok(encrypted_data)
}

fn mysymmetricdecrypt(encrypted_data: &[u8], key: &[u8]) -> Result<Vec<u8>> {
    let iv: [u8; 16] = [0; 16];
    let decrypted_data = symmetricdecrypt(&encrypted_data[..], &key, &iv)
        .map_err(|e| anyhow::anyhow!("decrypt error: {:?}", e))?;
    Ok(decrypted_data)
}
