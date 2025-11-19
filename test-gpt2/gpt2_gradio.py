import os

import gradio as gr
import markdown
import torch
import torch.nn.functional as F
from playwright.sync_api import sync_playwright
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()


def generate_with_details(prompt, max_new_tokens=10, top_k=10, temperature=1.0):
    """Generate text and show detailed inference steps."""

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output_text = ""
    all_steps = []

    # Initial input info
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    output_text += f"## Initial Input\n"
    output_text += f"**Prompt:** {prompt}\n"
    output_text += f"**Input IDs:** {input_ids[0].tolist()}\n"
    output_text += f"**Tokens:** {input_tokens}\n"
    output_text += f"**Input shape:** {list(input_ids.shape)}\n\n"
    output_text += "---\n\n"

    generated_ids = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

            # Apply temperature
            scaled_logits = next_token_logits / temperature

            # Get probabilities
            probs = F.softmax(scaled_logits, dim=-1)

            # Get top-k tokens
            topk_probs, topk_indices = torch.topk(probs, top_k)
            topk_logits = next_token_logits[topk_indices]

            # Choose the top token (greedy decoding)
            chosen_id = topk_indices[0].item()
            chosen_token = tokenizer.decode([chosen_id])
            chosen_prob = topk_probs[0].item()

            # Format step output
            output_text += f"## Step {step + 1}\n\n"
            output_text += f"### Input\n"
            output_text += f"- **Sequence length:** {generated_ids.shape[1]}\n"
            output_text += (
                f"- **Current text:** `{tokenizer.decode(generated_ids[0])}`\n\n"
            )

            output_text += "| Position | Token ID | Token | \n"
            output_text += "|----------|----------|-------|\n"
            for pos in range(generated_ids.shape[1]):
                token_id = generated_ids[0, pos].item()
                token_str = tokenizer.decode([token_id]).replace("\n", "\\n")
                output_text += f"| {pos} | {token_id} | `{token_str}` |\n"
            output_text += "\n"

            output_text += f"### Output\n"
            output_text += f"- **Logits shape:** {list(logits.shape)}\n"
            output_text += f"- **Last position logits range:** [{next_token_logits.min().item():.2f}, {next_token_logits.max().item():.2f}]\n\n"

            # Show logits for each input token position as table
            output_text += "**Logits per position:**\n\n"
            output_text += "| Position | Input Token | Top Prediction | Logit |\n"
            output_text += "|----------|-------------|----------------|-------|\n"
            for pos in range(generated_ids.shape[1]):
                pos_logits = logits[0, pos, :]
                pos_token = tokenizer.decode([generated_ids[0, pos].item()]).replace(
                    "\n", "\\n"
                )
                pos_top_val, pos_top_idx = torch.topk(pos_logits, 1)
                pos_top_token = tokenizer.decode([pos_top_idx[0].item()]).replace(
                    "\n", "\\n"
                )
                output_text += f"| {pos} | `{pos_token}` | `{pos_top_token}` | {pos_top_val[0].item():.2f} |\n"
            output_text += "\n"

            output_text += f"### Top-{top_k} Candidates\n"
            output_text += "| Rank | Token | ID | Logit | Probability |\n"
            output_text += "|------|-------|-----|-------|-------------|\n"

            for i in range(top_k):
                token_id = topk_indices[i].item()
                token_str = tokenizer.decode([token_id]).replace("\n", "\\n")
                logit_val = topk_logits[i].item()
                prob_val = topk_probs[i].item()
                marker = "→" if i == 0 else ""
                output_text += f"| {i+1} {marker} | `{token_str}` | {token_id} | {logit_val:.2f} | {prob_val:.4f} |\n"

            output_text += f"\n### Chosen Token\n"
            output_text += f"- **Token:** `{chosen_token}`\n"
            output_text += f"- **ID:** {chosen_id}\n"
            output_text += f"- **Probability:** {chosen_prob:.4f}\n\n"
            output_text += "---\n\n"

            # Append chosen token
            generated_ids = torch.cat(
                [generated_ids, torch.tensor([[chosen_id]])], dim=1
            )

            # Stop if EOS token
            if chosen_id == tokenizer.eos_token_id:
                break

    # Final result
    final_text = tokenizer.decode(generated_ids[0])
    output_text += f"## Final Result\n"
    output_text += f"**Generated text:** {final_text}\n"
    output_text += f"**Total tokens:** {generated_ids.shape[1]}\n"

    return output_text, final_text


def save_as_jpg(markdown_text):
    """Convert markdown output to JPG image."""
    if not markdown_text:
        return None

    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_text, extensions=["tables"])

    # Add styling
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 20px;
                background: white;
                font-size: 12px;
                line-height: 1.4;
                width: 800px;
            }}
            h2 {{
                color: #2563eb;
                border-bottom: 2px solid #2563eb;
                padding-bottom: 5px;
                margin-top: 15px;
            }}
            h3 {{
                color: #1e40af;
                margin-top: 10px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 6px;
                text-align: left;
            }}
            th {{
                background-color: #f3f4f6;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9fafb;
            }}
            code {{
                background-color: #e5e7eb;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }}
            hr {{
                border: none;
                border-top: 1px solid #e5e7eb;
                margin: 15px 0;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Save to file using Playwright
    output_path = "gpt2_inference_output.jpg"

    # Write HTML to temp file
    html_path = "/tmp/gpt2_output.html"
    with open(html_path, "w") as f:
        f.write(styled_html)

    # Use Playwright to screenshot
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 850, "height": 600})
        page.goto(f"file://{html_path}")

        # Get full page height
        height = page.evaluate("document.body.scrollHeight")
        page.set_viewport_size({"width": 850, "height": height})

        page.screenshot(path=output_path, type="jpeg", quality=95, full_page=True)
        browser.close()

    return output_path


# Create Gradio interface
with gr.Blocks(title="GPT-2 Inference Visualizer") as demo:
    gr.Markdown("# GPT-2 Inference Step-by-Step Visualizer")
    gr.Markdown(
        "Understand how GPT-2 converts logits to tokens during text generation."
    )

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                value="The quick brown fox",
            )
            max_tokens = gr.Slider(
                minimum=1, maximum=50, value=10, step=1, label="Max New Tokens"
            )
            top_k_slider = gr.Slider(
                minimum=5, maximum=50, value=10, step=1, label="Top-K to Display"
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature"
            )
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                save_btn = gr.Button("Save as JPG", variant="secondary")
            saved_file = gr.File(label="Download JPG")

    with gr.Row():
        with gr.Column():
            generated_output = gr.Textbox(label="Generated Text", lines=2)

    with gr.Row():
        details_output = gr.Markdown(label="Inference Details")

    generate_btn.click(
        fn=generate_with_details,
        inputs=[prompt_input, max_tokens, top_k_slider, temperature],
        outputs=[details_output, generated_output],
    )

    save_btn.click(fn=save_as_jpg, inputs=[details_output], outputs=[saved_file])

if __name__ == "__main__":
    demo.launch()
