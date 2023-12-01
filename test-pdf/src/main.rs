use anyhow::Result;
use walkdir::{WalkDir, DirEntry};
use regex::Regex;
use std::collections::HashSet;
use genpdf::Element;
use std::fs;
use std::io;
use std::io::Read;
use std::path::Path;
fn main() -> Result<()> {
    // Load a font from the file system
    let font_family =
        genpdf::fonts::from_files("./data", "Ubuntu", None).expect("Failed to load font family");

    // Create a document and set the default font family
    let mut doc = genpdf::Document::new(font_family);

    // Change the default settings
    doc.set_title("Demo document");

    // Customize the pages
    let mut decorator = genpdf::SimplePageDecorator::new();
    decorator.set_margins(10);
    doc.set_page_decorator(decorator);

    // Fetch text files from the data folder "mydocument"
    let text_files = read_data_folder("./mydocument")?;
    println!("Found {} text files", text_files.len());
    //display all text_files
    for file in &text_files {
        println!("File: {}", file.path().display());
    }

    // Add contents of each text file to the PDF
    for file in text_files {
        let filepath = file.path().display().to_string();   
        let filename = file.file_name().to_string_lossy().to_string();
        let mut file_content = String::new();

        let mut style = genpdf::style::Style::new();
        style.set_font_size(18);
        style.set_bold();

        
        fs::File::open(file.path())
            .expect("Failed to open file")
            .read_to_string(&mut file_content)
            .expect("Failed to read file content");

            doc.push(genpdf::elements::Paragraph::new(&filepath)
            .styled(style)); // Example style

        //println!("File: {}\n{}", filename, file_content);
        // Split the text into sentences
        let sentences = split_into_sentences(&file_content);

        // Add each sentence as a paragraph to the PDF
        for sentence in sentences {
          //doc.push(genpdf::elements::Paragraph::new(sentence));
          doc.push(genpdf::elements::Paragraph::new(format!("@{}#", sentence)));
        }

        // Add filename and content to the document
        //doc.push(genpdf::elements::Paragraph::new(format!("File: {}\n{}", filename, file_content)));
    }
    // Render the document and write it to a file
    doc.render_to_file("output.pdf")?;
    Ok(())
}

// Function to repeat a text multiple times with a newline
fn repeat_text(text: &str, count: usize) -> String {
    (0..count).map(|_| text).collect()
}

fn split_into_sentences(text: &str) -> Vec<String> {
    text.split(|c| c == '\n' || c == '\r')
        .map(|sentence| sentence.replace("\n", "").replace("\r", "").trim().to_string())
        .filter(|sentence| !sentence.is_empty())
        .collect()
}



fn read_data_folder(path: &str) -> Result<Vec<DirEntry>> {
    let allowed_extensions: HashSet<_> = vec!["txt", "proto", "rs", "go"]
        .into_iter()
        .collect();

    println!("Reading data folder: {}", path);

    let entries = WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            if entry.file_type().is_file() {
                entry.path().extension()
                    .and_then(std::ffi::OsStr::to_str)
                    .map(|ext| allowed_extensions.contains(ext))
                    .and_then(|is_allowed| if is_allowed { Some(entry) } else { None })
            } else {
                None
            }
        })
        .collect::<Vec<DirEntry>>();

    Ok(entries)
}