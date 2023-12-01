fn main() {
    // Load a font from the file system
    let font_family = genpdf::fonts::from_files("./data", "Ubuntu", None)
        .expect("Failed to load font family");

    // Create a document and set the default font family
    let mut doc = genpdf::Document::new(font_family);

    // Change the default settings
    doc.set_title("Demo document");

    // Customize the pages
    let mut decorator = genpdf::SimplePageDecorator::new();
    decorator.set_margins(10);
    doc.set_page_decorator(decorator);

    // Add one or more elements
    for _ in 0..20 {
        let long_text = repeat_text("This is a demo document.\n", 10); // Repeat the sentence 50 times with new lines
        doc.push(genpdf::elements::Paragraph::new(long_text));

        //let image_path = "./data/myimage.png"; // Specify the path to your image file
        //let image = genpdf::elements::Image::from_path(image_path).expect("Failed to load image");

        // Add the image to the document
       // doc.push(image);
    }

    // Render the document and write it to a file
    doc.render_to_file("output.pdf")
        .expect("Failed to write PDF file");
}

// Function to repeat a text multiple times with a newline
fn repeat_text(text: &str, count: usize) -> String {
    (0..count).map(|_| text).collect()
}
