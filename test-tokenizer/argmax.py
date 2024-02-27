import torch
import torch.nn.functional as F

# Generate three arrays of length 10 with random values
logits_examples = torch.rand((3, 10))  # 3 examples, each with 10 random logits

# Iterate through each array of logits
for i, logits in enumerate(logits_examples, start=1):
    #print  i+1, logits
    print(f"Example {i}: {logits}")
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=0)

    # Sum of probabilities (should be 1.0 for a correct softmax)
    sum_probabilities = torch.sum(probabilities)

    # Determine the predicted class using argmax
    predicted_class = torch.argmax(probabilities)

    # Print results
    print(f"Set {i}:")
    print(f"Probabilities: {probabilities}")
    print(f"Sum of probabilities: {sum_probabilities}")
    print(f"Predicted class: {predicted_class}")
    print("-" * 50)
