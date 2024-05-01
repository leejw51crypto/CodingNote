import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the candidate text labels
text_labels = ["cat", "dog", "bird", "car", "plane"]

# Prepare the text labels for CLIP
text_inputs = processor(text=text_labels, return_tensors="pt", padding=True)

# Encode the text labels
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

# Load and preprocess the input image
image_path = "image.jpg"
image = Image.open(image_path)
image_input = processor(images=image, return_tensors="pt")

# Encode the image
with torch.no_grad():
    image_features = model.get_image_features(**image_input)

# Compute the similarity scores between the image and text labels
similarity_scores = torch.cosine_similarity(image_features, text_features)

# Get the index of the most similar text label
predicted_label_index = similarity_scores.argmax().item()

# Print the predicted label
predicted_label = text_labels[predicted_label_index]
print(f"The predicted label for the image is: {predicted_label}")
