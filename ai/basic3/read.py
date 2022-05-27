#!/usr/bin/python3
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# Load pre-trained ViT model and feature extractor
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# model = ViTForImageClassification.from_pretrained('./beans_outputs')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Load input image and preprocess
image = Image.open("a.jpg")
image = image.resize((224, 224))  # resize image to 224x224 pixels
inputs = feature_extractor(images=image, return_tensors="pt")

# Make prediction and print predicted label
outputs = model(**inputs)
predicted_label = outputs.logits.argmax(dim=-1).item()
print(f"Predicted label: {predicted_label}")
