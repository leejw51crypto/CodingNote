use anyhow::Result;
use bip39::{Language, Mnemonic};
use colored::*;
use rand::Rng;

/// Generates a random BIP39 mnemonic phrase with 24 words.
fn generate_mnemonic() -> Result<String> {
    // Generate a random entropy of 32 bytes (256 bits)
    let entropy_bytes: [u8; 32] = rand::thread_rng().gen();

    // Create a mnemonic phrase from the entropy using the English language
    let mnemonic = Mnemonic::from_entropy_in(Language::English, &entropy_bytes)?;

    // Ensure the mnemonic has the expected word count of 24
    assert_eq!(mnemonic.word_count(), 24);

    // Convert the mnemonic to a string
    let phrase = mnemonic.to_string();

    Ok(phrase)
}

fn main() -> Result<()> {
    // Generate a random mnemonic with 24 words
    let mnemonic = generate_mnemonic()?;

    // Display the mnemonic phrase with a cool gradient effect
    println!("🔑 Your Cool Mnemonic Phrase 🔑");
    for (index, word) in mnemonic.split_whitespace().enumerate() {
        let color = match index % 3 {
            0 => "red",
            1 => "green",
            2 => "blue",
            _ => unreachable!(),
        };
        print!("{} ", word.color(color).bold());
    }
    println!("\n");

    Ok(())
}
