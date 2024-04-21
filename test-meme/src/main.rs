use anyhow::{Context, Result};
use image::io::Reader as ImageReader;
use image::Pixel;
use image::{ImageBuffer, Rgba};
use rusttype::{Font, Scale};
use std::io::{self, Write};

fn main() -> Result<()> {
    // Read input image name from the user
    let input_image = get_user_input(
        "Enter the input image name (default: myimage.jpg):",
        "myimage.jpg",
    )?;

    // Read meme text from the user
    let text = get_user_input(
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

    // Calculate the font size based on the image height
    let font_size = image.height() as f32 * 0.06;
    let scale = Scale::uniform(font_size);
    let mut thickness = (font_size * 0.05) as i32;
    if thickness < 2 {
        thickness = 2;
    }

    // Split the text into multiple lines based on the maximum width
    let max_width = image.width() as f32 * 0.8;
    let lines = split_text(&font, scale, &text, max_width)?;

    // Calculate the total height of the text
    let line_height = font.v_metrics(scale).ascent - font.v_metrics(scale).descent;
    let total_height = line_height * lines.len() as f32;

    // Calculate the starting position for the text
    let text_y = image.height() as f32 - total_height - 20.0;

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
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        let mut potential_line = current_line.clone();
        if !potential_line.is_empty() {
            potential_line.push(' ');
        }
        potential_line.push_str(word);

        let (line_width, _) = measure_text(font, scale, &potential_line)?;
        if line_width > max_width {
            if !current_line.is_empty() {
                lines.push(current_line);
            }
            current_line = word.to_string();
        } else {
            current_line = potential_line;
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    Ok(lines)
}
