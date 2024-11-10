import numpy as np

class SimpleRNNCell:
    """
    A basic RNN cell showing internal calculations step by step
    """
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases
        self.W_xh = np.random.randn(input_size, hidden_size)  # Input -> Hidden weights
        self.W_hh = np.random.randn(hidden_size, hidden_size) # Hidden -> Hidden weights
        self.W_hy = np.random.randn(hidden_size, hidden_size) # Hidden -> Output weights
        self.b_h = np.zeros((1, hidden_size))                 # Hidden bias
        self.b_y = np.zeros((1, hidden_size))                 # Output bias

    def step(self, x, h_prev):
        """
        Single step of RNN computation
        x: input at current time step
        h_prev: hidden state from previous time step
        """
        # Step 1: Combine input with previous hidden state
        # h_new = tanh(W_xh * x + W_hh * h_prev + b_h)
        input_transform = np.dot(x, self.W_xh)          # Transform input
        hidden_transform = np.dot(h_prev, self.W_hh)    # Transform previous hidden state
        total_input = input_transform + hidden_transform + self.b_h
        h_new = np.tanh(total_input)                    # New hidden state

        # Step 2: Calculate output
        # y = W_hy * h_new + b_y
        y = np.dot(h_new, self.W_hy) + self.b_y        # Output

        return h_new, y

# Example usage showing the flow of information
def run_rnn_example():
    # Initialize RNN cell
    input_size = 5
    hidden_size = 3
    rnn = SimpleRNNCell(input_size, hidden_size)
    
    # Initial hidden state (all zeros)
    h = np.zeros((1, hidden_size))
    
    # Example sequence of 3 time steps
    sequence = [
        np.random.randn(1, input_size),  # Input at time step 1
        np.random.randn(1, input_size),  # Input at time step 2
        np.random.randn(1, input_size)   # Input at time step 3
    ]
    
    print("Processing sequence step by step:")
    for t, x in enumerate(sequence):
        print(f"\nTime Step {t+1}")
        print("Input shape:", x.shape)
        print("Previous hidden state shape:", h.shape)
        
        # Process one time step
        h, y = rnn.step(x, h)
        
        print("New hidden state shape:", h.shape)
        print("Output shape:", y.shape)
        
        # Print actual values for understanding
        print(f"Input values:\n{x}")
        print(f"Hidden state:\n{h}")
        print(f"Output:\n{y}")

# Run the example
if __name__ == "__main__":
    run_rnn_example()

"""
Key Equations in RNN:

1. Hidden State Update:
   h_t = tanh(W_xh * x_t + W_hh * h_(t-1) + b_h)
   where:
   - h_t is the new hidden state
   - x_t is the current input
   - h_(t-1) is the previous hidden state
   - W_xh, W_hh are weight matrices
   - b_h is the hidden bias

2. Output Calculation:
   y_t = W_hy * h_t + b_y
   where:
   - y_t is the output at current time step
   - W_hy is the output weight matrix
   - b_y is the output bias

The magic of RNN is in how it maintains and updates the hidden state,
which acts as the network's "memory" of previous inputs.
"""