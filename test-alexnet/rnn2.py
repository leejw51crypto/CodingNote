import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.005):
        # Initialize parameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Add gradient clipping threshold
        self.clip_threshold = 5.0
        
        # Initialize weights with better scaling
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0/input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(2.0/hidden_size)
        
        # Initialize biases
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias

    def forward(self, inputs, hidden_state):
        """Forward pass through the network"""
        self.hidden_states = [hidden_state]
        self.inputs = inputs
        self.outputs = []

        # For each input in sequence
        for i in range(len(inputs)):
            # Add small epsilon to prevent explosion
            hidden_state = np.tanh(
                np.dot(self.Wxh, inputs[i]) + 
                np.dot(self.Whh, self.hidden_states[-1]) + 
                self.bh
            ) + 1e-10
            
            # Calculate output with temperature
            output = np.dot(self.Why, hidden_state) + self.by
            
            # Apply softmax with temperature
            output = output / 1.0  # Temperature parameter (higher = more random)
            output_prob = np.exp(output) / (np.sum(np.exp(output)) + 1e-10)
            
            # Store states for backpropagation
            self.hidden_states.append(hidden_state)
            self.outputs.append(output_prob)

        return self.outputs, self.hidden_states

    def backward(self, targets):
        """Backward pass for parameter learning"""
        # Initialize gradient accumulators
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Initialize hidden state gradient
        dhnext = np.zeros_like(self.hidden_states[0])

        # For each timestep, backward
        for t in reversed(range(len(self.inputs))):
            # Gradient of output
            dy = self.outputs[t].copy()
            dy[targets[t]] -= 1
            
            # Hidden to output weights
            dWhy += np.dot(dy, self.hidden_states[t+1].T)
            dby += dy
            
            # Into hidden state
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - self.hidden_states[t+1] * self.hidden_states[t+1]) * dh
            
            # Calculate gradients
            dbh += dhraw
            dWxh += np.dot(dhraw, self.inputs[t].T)
            dWhh += np.dot(dhraw, self.hidden_states[t].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        # Clip gradients to prevent explosion
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        # Update parameters
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

def train_rnn():
    """Example: Train RNN on a simple sequence"""
    # Add more complete sentences with proper endings
    data = """hello world hello how are you doing today. 
              """.replace('\n', ' ')
    
    chars = list(set(data))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Slightly increase model capacity
    hidden_size = 96
    learning_rate = 0.001
    
    # Initialize RNN
    rnn = SimpleRNN(
        input_size=len(chars),
        hidden_size=hidden_size,
        output_size=len(chars),
        learning_rate=learning_rate
    )
    
    # Modified training loop
    best_loss = float('inf')
    patience = 20  # Increased patience
    patience_counter = 0
    
    n_iterations = 1000
    for i in range(n_iterations):
        # Prepare inputs and targets
        inputs = []
        targets = []
        for t in range(len(data)-1):
            input_vec = np.zeros((len(chars), 1))
            input_vec[char_to_idx[data[t]]] = 1
            inputs.append(input_vec)
            targets.append(char_to_idx[data[t+1]])
        
        # Forward pass
        hidden_state = np.zeros((hidden_size, 1))
        outputs, hidden_states = rnn.forward(inputs, hidden_state)
        
        # Backward pass
        rnn.backward(targets)
        
        # Calculate loss (cross-entropy) - Fixed scalar conversion
        loss = 0
        for t in range(len(targets)):
            prob = outputs[t][targets[t]]
            loss += -np.log(prob + 1e-10).item()  # Proper scalar conversion
        
        # Early stopping with more patience
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and i > 100:  # Don't stop too early
            print(f'Early stopping at iteration {i}')
            break
        
        # Print progress
        if i % 10 == 0:
            print(f'Iteration {i}, Loss: {loss:.4f}')

    # Return the trained model and character mappings
    return rnn, char_to_idx, idx_to_char

def generate_text(rnn, seed_char, char_to_idx, idx_to_char, length=50):
    """Generate text using trained RNN"""
    current_char = seed_char
    generated_text = seed_char
    hidden_state = np.zeros((rnn.hidden_size, 1))
    
    # Track if we're at the end of a sentence
    sentence_ended = False
    
    for i in range(length):
        x = np.zeros((len(char_to_idx), 1))
        x[char_to_idx[current_char]] = 1
        
        outputs, hidden_states = rnn.forward([x], hidden_state)
        hidden_state = hidden_states[-1]
        
        # Adjust temperature based on position in sequence
        if i > length * 0.8:  # Near the end
            temperature = 0.01  # Very focused
        else:
            temperature = 0.05  # Normal generation
            
        probs = outputs[-1]
        probs = np.power(probs, 1/temperature)
        probs = probs / np.sum(probs)
        
        # Force ending with punctuation if near the end
        if i > length - 5 and not sentence_ended:
            # Filter for punctuation marks
            for idx in [char_to_idx[c] for c in '.?!']:
                probs[idx] *= 10.0  # Increase probability of endings
            probs = probs / np.sum(probs)
        
        # Very conservative sampling
        if np.random.random() < 0.98:  # 98% of the time take most likely
            next_char_idx = np.argmax(probs)
        else:
            next_char_idx = np.random.choice(len(probs), p=probs.ravel())
        
        current_char = idx_to_char[next_char_idx]
        
        # Check if we've ended the sentence
        if current_char in '.?!':
            sentence_ended = True
            
        generated_text += current_char
        
        # Stop if we've completed a sentence
        if sentence_ended and len(generated_text.strip()) > length * 0.8:
            break
    
    return generated_text

# Run the example
if __name__ == "__main__":
    print("Training RNN...")
    rnn, char_to_idx, idx_to_char = train_rnn()
    
    print("\nGenerating text...")
    generated = generate_text(rnn, 'h', char_to_idx, idx_to_char, length=50)
    print("Generated text:", generated)