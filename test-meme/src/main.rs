use image::Pixel;
use image::{ImageBuffer, Rgba};
use rusttype::{Font, Scale};
use text_io::read;

fn main() {
    // Read input image name from the user
    println!("Enter the input image name (default: myimage.jpg):");
    let input_image: String = read!("{}\n");
    let input_image = if input_image.is_empty() {
        "myimage.jpg".to_string()
    } else {
        input_image
    };

    // Read meme text from the user
    println!("Enter the meme text (default: Your meme text here):");
    let text: String = read!("{}\n");
    let text = if text.is_empty() {
        "Your meme text here".to_string()
    } else {
        text
    };

    // Read output image name from the user
    println!("Enter the output image name (default: meme.png):");
    let output_image: String = read!("{}\n");
    let output_image = if output_image.is_empty() {
        "meme.png".to_string()
    } else {
        output_image
    };

    // Load the image file
    let mut image = image::open(&input_image).unwrap().to_rgba8();

    // Load the font file
    let font_data = include_bytes!("myfont.ttf");
    let font = Font::try_from_bytes(font_data).unwrap();

    // Calculate the font size based on the image height
    let font_size = image.height() as f32 * 0.1;

    // Create a scale for the font
    let scale = Scale::uniform(font_size);

    // Split the text into multiple lines based on the maximum width
    let max_width = image.width() as f32 * 0.8; // Adjust the maximum width as needed
    let lines = split_text(&font, scale, &text, max_width);

    // Calculate the total height of the text
    let line_height = font.v_metrics(scale).ascent - font.v_metrics(scale).descent;
    let total_height = line_height * lines.len() as f32;

    // Calculate the starting position for the text
    let text_y = image.height() as f32 - total_height - 20.0; // Adjust the vertical position as needed

    let thickness = (font_size * 0.2) as i32;

    for (i, line) in lines.iter().enumerate() {
        let (line_width, _) = measure_text(&font, scale, line);
        let text_x = (image.width() as f32 - line_width) / 2.0;
        let line_y = text_y + i as f32 * line_height;

        for sy in -thickness..thickness {
            for sx in -thickness..thickness {
                draw_text_mut(
                    &mut image,
                    Rgba([0, 0, 0, 255]),
                    text_x as i32 + sx,
                    line_y as i32 + sy,
                    scale,
                    &font,
                    line,
                );
            }
        }

        // Draw the white text
        draw_text_mut(
            &mut image,
            Rgba([255, 255, 255, 255]),
            text_x as i32,
            line_y as i32,
            scale,
            &font,
            line,
        );
    }

    // Save the modified image
    image.save(&output_image).unwrap();
}

fn measure_text(font: &Font, scale: Scale, text: &str) -> (f32, f32) {
    let v_metrics = font.v_metrics(scale);
    let glyphs: Vec<_> = font
        .layout(text, scale, rusttype::point(0.0, v_metrics.ascent))
        .collect();

    let width = glyphs.last().unwrap().pixel_bounding_box().unwrap().max.x as f32;
    let height = v_metrics.ascent - v_metrics.descent;

    (width, height)
}

fn draw_text_mut(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    color: Rgba<u8>,
    x: i32,
    y: i32,
    scale: Scale,
    font: &Font,
    text: &str,
) {
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
                    let pixel = image.get_pixel_mut(image_x, image_y);
                    let alpha = (gv * 255.0) as u8;
                    let text_color = Rgba([color[0], color[1], color[2], alpha]);
                    pixel.blend(&text_color);
                }
            });
        }
    }
}

fn split_text(font: &Font, scale: Scale, text: &str, max_width: f32) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        let mut potential_line = current_line.clone();
        if !potential_line.is_empty() {
            potential_line.push(' ');
        }
        potential_line.push_str(word);

        let (line_width, _) = measure_text(font, scale, &potential_line);
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

    lines
}
