use anyhow::{Context, Result};
use image::io::Reader as ImageReader;
use image::Pixel;
use image::{ImageBuffer, Rgba};
use rusttype::{Font, Scale};
use std::io::{self, Write};
use viuer::{print_from_file, Config};

fn main() -> Result<()> {
    // Read input image name from the user
    let input_image = get_user_input(
        "Enter the input image name (default: myimage.jpg):",
        "myimage.jpg",
    )?;

    // Read meme text from the user
    let text = get_user_input_multiple(
        "Enter the meme text (default: Your meme text here):",
        "Your meme text here",
    )?;

    // Read output image name from the user
    let output_image = get_user_input(
        "Enter the output image name (default: meme.png):",
        "meme.png",
    )?;

    // Load the image file
    let mut image = ImageReader::open(&input_image)
        .with_context(|| format!("Failed to open image file: {}", input_image))?
        .decode()?
        .to_rgba8();

    // Load the font file
    let font_data = include_bytes!("myfont.ttf");
    let font = Font::try_from_bytes(font_data).with_context(|| "Failed to load font")?;

    // Calculate the initial font size based on the image height
    let initial_font_size = image.height() as f32 * 0.2;
    let initial_scale = Scale::uniform(initial_font_size);

    // Split the text into multiple lines based on the maximum width
    let max_width = image.width() as f32 * 0.9;
    let mut lines = split_text(&font, initial_scale, &text, max_width)?;

    // Adjust font size based on the number of lines
    let font_size = if lines.len() > 5 {
        let scale_factor = (5.0 / lines.len() as f32).max(0.5);
        image.height() as f32 * 0.2 * scale_factor
    } else {
        initial_font_size
    };

    let scale = Scale::uniform(font_size);
    let mut thickness = (font_size * 0.05) as i32;
    if thickness < 2 {
        thickness = 2;
    }

    // Recalculate lines with the new font size
    lines = split_text(&font, scale, &text, max_width)?;

    // Calculate the total height of the text
    let line_height = font.v_metrics(scale).ascent - font.v_metrics(scale).descent;
    let total_height = line_height * lines.len() as f32;

    // Calculate the starting position for the text
    let text_y = if lines.len() > 10 {
        image.height() as f32 * 0.1 // Start higher for very long text
    } else {
        image.height() as f32 - total_height - 20.0 // Original position
    };

    // Handle overflow
    let max_lines = (image.height() as f32 / line_height) as usize - 2; // Leave some margin
    if lines.len() > max_lines {
        lines.truncate(max_lines);
        if let Some(last) = lines.last_mut() {
            let mut chars: Vec<char> = last.chars().collect();
            if chars.len() > 3 {
                chars.truncate(chars.len() - 3);
                chars.extend(['.', '.', '.']);
                *last = chars.into_iter().collect();
            }
        }
    }

    // Draw the text with an outline and fill
    for (i, line) in lines.iter().enumerate() {
        let (line_width, _) = measure_text(&font, scale, line)?;
        let text_x = (image.width() as f32 - line_width) / 2.0;
        let line_y = text_y + i as f32 * line_height;

        draw_text_outline(
            &mut image,
            text_x as i32,
            line_y as i32,
            scale,
            &font,
            line,
            thickness,
        )?;
        draw_text(&mut image, text_x as i32, line_y as i32, scale, &font, line)?;
    }

    // Save the modified image
    image.save(&output_image)?;

    // Print the generated meme image
    print_generated_meme(&output_image)?;

    Ok(())
}

fn get_user_input(prompt: &str, default: &str) -> Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    Ok(if input.is_empty() {
        default.to_string()
    } else {
        input.to_string()
    })
}

fn get_user_input_multiple(prompt: &str, default: &str) -> Result<String> {
    println!("{}", prompt);
    println!("Enter an empty line to finish.");

    let mut lines = Vec::new();
    loop {
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        let line = line.trim();

        if line.is_empty() {
            break;
        }
        lines.push(line.to_string());
    }

    let input = lines.join("\n");

    Ok(if input.is_empty() {
        default.to_string()
    } else {
        input
    })
}

fn measure_text(font: &Font, scale: Scale, text: &str) -> Result<(f32, f32)> {
    let v_metrics = font.v_metrics(scale);
    let glyphs: Vec<_> = font
        .layout(text, scale, rusttype::point(0.0, v_metrics.ascent))
        .collect();

    let width = glyphs.last().map_or(Ok(0.0), |glyph| {
        glyph
            .pixel_bounding_box()
            .map(|bbox| bbox.max.x as f32)
            .context("Failed to get glyph bounding box")
    })?;
    let height = v_metrics.ascent - v_metrics.descent;

    Ok((width, height))
}

fn draw_text(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: i32,
    y: i32,
    scale: Scale,
    font: &Font,
    text: &str,
) -> Result<()> {
    draw_text_with_color(image, x, y, scale, font, text, Rgba([255, 255, 255, 255]))
}

fn draw_text_outline(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: i32,
    y: i32,
    scale: Scale,
    font: &Font,
    text: &str,
    thickness: i32,
) -> Result<()> {
    for sy in -thickness..=thickness {
        for sx in -thickness..=thickness {
            if sx.abs() == thickness || sy.abs() == thickness {
                draw_text_with_color(
                    image,
                    x + sx,
                    y + sy,
                    scale,
                    font,
                    text,
                    Rgba([0, 0, 0, 255]),
                )?;
            }
        }
    }
    Ok(())
}

fn draw_text_with_color(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: i32,
    y: i32,
    scale: Scale,
    font: &Font,
    text: &str,
    color: Rgba<u8>,
) -> Result<()> {
    let v_metrics = font.v_metrics(scale);
    let glyphs: Vec<_> = font
        .layout(text, scale, rusttype::point(0.0, v_metrics.ascent))
        .collect();

    for glyph in glyphs {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|gx, gy, gv| {
                let gx = gx as i32 + x + bounding_box.min.x;
                let gy = gy as i32 + y + bounding_box.min.y;
                let image_x = gx as u32;
                let image_y = gy as u32;

                if image_x < image.width() && image_y < image.height() {
                    let background_pixel = image.get_pixel(image_x, image_y);
                    let alpha = (gv * color.channels()[3] as f32) as u8;
                    let blended_pixel = alpha_blend(background_pixel, color, alpha);
                    image.put_pixel(image_x, image_y, blended_pixel);
                }
            });
        }
    }
    Ok(())
}

fn alpha_blend(background: &Rgba<u8>, foreground: Rgba<u8>, alpha: u8) -> Rgba<u8> {
    let alpha_f = alpha as f32 / 255.0;
    let inv_alpha_f = 1.0 - alpha_f;

    let mut blended_pixel = Rgba([0, 0, 0, 0]);
    for i in 0..3 {
        blended_pixel.channels_mut()[i] = (background.channels()[i] as f32 * inv_alpha_f
            + foreground.channels()[i] as f32 * alpha_f)
            as u8;
    }
    blended_pixel.channels_mut()[3] = 255;

    blended_pixel
}

fn split_text(font: &Font, scale: Scale, text: &str, max_width: f32) -> Result<Vec<String>> {
    let mut lines = Vec::new();

    // Calculate maximum width for a single line
    let max_line_width = max_width * 0.9; // Using 90% of the image width

    for paragraph in text.split("\n\n") {
        for line in paragraph.split('\n') {
            let mut current_line = String::new();
            for word in line.split_whitespace() {
                let (word_width, _) = measure_text(font, scale, word)?;

                if word_width > max_line_width {
                    // If the word itself is longer than max_line_width, split it
                    if !current_line.is_empty() {
                        lines.push(current_line);
                        current_line = String::new();
                    }

                    let mut char_line = String::new();
                    for ch in word.chars() {
                        let potential_line = char_line.clone() + &ch.to_string();
                        let (potential_width, _) = measure_text(font, scale, &potential_line)?;

                        if potential_width > max_line_width {
                            lines.push(char_line);
                            char_line = ch.to_string();
                        } else {
                            char_line.push(ch);
                        }
                    }

                    if !char_line.is_empty() {
                        lines.push(char_line);
                    }
                } else {
                    let mut potential_line = current_line.clone();
                    if !potential_line.is_empty() {
                        potential_line.push(' ');
                    }
                    potential_line.push_str(word);

                    let (line_width, _) = measure_text(font, scale, &potential_line)?;
                    if line_width > max_line_width {
                        if !current_line.is_empty() {
                            lines.push(current_line);
                        }
                        current_line = word.to_string();
                    } else {
                        current_line = potential_line;
                    }
                }
            }
            if !current_line.is_empty() {
                lines.push(current_line);
            }
        }
        if paragraph.ends_with("\n\n") {
            lines.push(String::new());
        }
    }
    Ok(lines)
}
// Add this new function to print the generated meme
fn print_generated_meme(image_path: &str) -> Result<()> {
    let conf = Config {
        width: Some(80),
        height: Some(15),
        absolute_offset: false,
        ..Default::default()
    };

    print_from_file(image_path, &conf).context("Image printing failed")?;
    Ok(())
}
