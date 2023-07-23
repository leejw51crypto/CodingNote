use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    // Read the file path from the user
    println!("Enter the file path:");
    let mut file_path = String::new();
    io::stdin().read_line(&mut file_path)?;

    // Trim any leading or trailing whitespace or newline characters
    file_path = file_path.trim().to_string();

    
    // Open the file
    let mut file = File::open(&file_path)?;

    // Read the file content into a string
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    
    let mut key = String::new();
    println!("Enter the key:");
    io::stdin().read_line(&mut key)?;
    key=key.trim().to_string();    
    // if key is empty
    if key.is_empty() {
        println!("Key is empty");
        return Ok(());
    }
    let totaljson:serde_json::Value=serde_json::from_str(&content)?;
    let keyjson= totaljson[key].clone();
    content= keyjson.to_string();    

    // Replace " with \"
    let replaced_content = content.replace("\"", "\\\"");

    // Add double quotes at the beginning and end of the replaced content
    let final_content = format!("\"{}\"", replaced_content);

    // Create a new file with the original filename + ".new"
    let new_file_path = format!("{}.new", file_path);
    let mut new_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&new_file_path)?;

    // Write the final content to the new file
    new_file.write_all(final_content.as_bytes())?;

    println!("Replaced content saved to: {}", new_file_path);

    Ok(())
}
