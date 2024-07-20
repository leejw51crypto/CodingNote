use arboard::Clipboard;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new Clipboard instance
    let mut clipboard = Clipboard::new()?;

    // Copy (set) text to clipboard
    clipboard.set_text("Hello, Arboard!")?;
    println!("Text copied to clipboard");

    // Paste (get) text from clipboard
    let pasted_text = clipboard.get_text()?;
    println!("Text pasted from clipboard: {}", pasted_text);

    Ok(())
}
