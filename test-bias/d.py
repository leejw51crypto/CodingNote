import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def cross_entropy_loss(probs, targets):
    return -np.sum(targets * np.log(probs), axis=-1)

def backpropagation(logits, targets, learning_rate):
    # Shape of logits: (batch_size, sequence_length, vocab_size)
    # Shape of targets: (batch_size, sequence_length, vocab_size)
    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)

    # Apply softmax to obtain predicted probabilities
    probs = softmax(logits)
    print("Probabilities shape:", probs.shape)

    # Compute cross-entropy loss
    loss = cross_entropy_loss(probs, targets)
    print("Loss shape:", loss.shape)

    # Compute gradients
    dlogits = probs - targets
    print("Gradients shape:", dlogits.shape)

    # Update the logits using gradient descent
    logits -= learning_rate * dlogits
    print("Updated logits shape:", logits.shape)

    return logits, loss

# Example usage
batch_size = 2
sequence_length = 3
vocab_size = 4

# Random logits and targets
logits = np.random.randn(batch_size, sequence_length, vocab_size)
targets = np.random.randint(0, vocab_size, size=(batch_size, sequence_length))
targets_one_hot = np.eye(vocab_size)[targets]

learning_rate = 0.1
num_iterations = 5

for i in range(num_iterations):
    print(f"\nIteration {i+1}:")
    logits, loss = backpropagation(logits, targets_one_hot, learning_rate)
    print(f"Average Loss: {np.mean(loss):.4f}")
