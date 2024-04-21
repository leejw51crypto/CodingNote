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

    // Calculate the text position
    let (text_width, text_height) = measure_text(&font, scale, &text);
    let text_x = (image.width() as f32 - text_width) / 2.0;
    let text_y = image.height() as f32 - text_height - 20.0; // Adjust the vertical position as needed

    let thickness = (font_size * 0.2) as i32;
    for sy in -thickness..thickness {
        for sx in -thickness..thickness {
            draw_text_mut(
                &mut image,
                Rgba([0, 0, 0, 255]),
                text_x as i32 + sx,
                text_y as i32 + sy,
                scale,
                &font,
                &text,
            );
        }
    }

    // Draw the white text
    draw_text_mut(
        &mut image,
        Rgba([255, 255, 255, 255]),
        text_x as i32,
        text_y as i32,
        scale,
        &font,
        &text,
    );

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
