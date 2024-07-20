use anyhow::{Context, Result};
use arboard::{Clipboard, ImageData};
use image::{DynamicImage, ImageFormat};
use std::path::Path;

fn get_image_and_save(clipboard: &mut Clipboard, file_path: &str) -> Result<()> {
    let image_data = clipboard
        .get_image()
        .context("Failed to get image from clipboard")?;
    println!(
        "Got image from clipboard. Dimensions: {}x{}",
        image_data.width, image_data.height
    );

    let image = DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(
            image_data.width as u32,
            image_data.height as u32,
            image_data.bytes.into(),
        )
        .context("Failed to create RgbaImage")?,
    );

    image
        .save_with_format(file_path, ImageFormat::Png)
        .context("Failed to save image")?;
    println!("Saved image to {}", file_path);
    Ok(())
}

fn read_image_and_set(clipboard: &mut Clipboard, file_path: &str) -> Result<()> {
    let image = image::open(file_path).context("Failed to open image file")?;
    let rgba_image = image.to_rgba8();
    let image_data = ImageData {
        width: rgba_image.width() as usize,
        height: rgba_image.height() as usize,
        bytes: rgba_image.into_raw().into(),
    };
    clipboard
        .set_image(image_data)
        .context("Failed to set image to clipboard")?;
    println!("Set image from {} to clipboard", file_path);
    Ok(())
}

fn main() -> Result<()> {
    let mut clipboard = Clipboard::new().context("Failed to create clipboard")?;

    // Get image from clipboard and save to file
    let save_path = "clipboard_image.png";
    if let Err(e) = get_image_and_save(&mut clipboard, save_path) {
        println!("Error saving image from clipboard: {:#}", e);
    }

    // Read image from file and set to clipboard
    let file_path = "image_to_set.png";
    if Path::new(file_path).exists() {
        if let Err(e) = read_image_and_set(&mut clipboard, file_path) {
            println!("Error setting image to clipboard: {:#}", e);
        }
    } else {
        println!("File {} does not exist", file_path);
    }

    Ok(())
}
