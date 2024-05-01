import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess the input image
image_path = "image.jpg"
image = Image.open(image_path)
image_input = processor(images=image, return_tensors="pt")

# Encode the image
with torch.no_grad():
    image_features = model.get_image_features(**image_input)

print("Image Embedding Shape:", image_features.shape)
