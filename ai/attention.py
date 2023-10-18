import numpy as np

def scaled_dot_product_attention(query, key, value):
    """
    Compute the scaled dot product attention for self-attention.
    
    Args:
    - query: Query matrix (shape: [batch_size, num_positions, depth])
    - key: Key matrix (shape: [batch_size, num_positions, depth])
    - value: Value matrix (shape: [batch_size, num_positions, value_depth])
    
    Returns:
    - output: Output after applying attention weights on the value matrix (shape: [batch_size, num_positions, value_depth])
    - attention_weights: Attention weights (shape: [batch_size, num_positions, num_positions])
    """
    
    # Compute the dot product between query and key
    matmul_qk = np.matmul(query, key.transpose(0, 2, 1))
    
    # Scale the dot product by the square root of the depth
    depth = query.shape[-1]
    logits = matmul_qk / np.sqrt(depth)
    
    # Compute the attention weights using softmax
    attention_weights = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    
    # Compute the output by multiplying the attention weights with the value matrix
    output = np.matmul(attention_weights, value)
    
    # Print formatted outputs
    print("Shapes:")
    print("- Query (Q):", query.shape)  # Expected: (batch_size, num_positions, depth)
    print("- Key (K):", key.shape)      # Expected: (batch_size, num_positions, depth)
    print("- Key Transposed:", key.transpose(0, 2, 1).shape)  # Expected: (batch_size, depth, num_positions)
    print("- Dot Product (Q.K^T):", matmul_qk.shape)  # Expected: (batch_size, num_positions, num_positions)
    print("- Attention Weights:", attention_weights.shape)  # Expected: (batch_size, num_positions, num_positions)
    print("- Value (V):", value.shape)  # Expected: (batch_size, num_positions, value_depth)
    print("- Output:", output.shape)    # Expected: (batch_size, num_positions, value_depth)
    
    print("\nValues:")
    # The following matrices will have random values based on the input, so no specific expected values can be provided.
    print("- Dot Product (Q.K^T):\n", matmul_qk)
    print("- Scaled Dot Product:\n", logits)
    print("- Attention Weights:\n", attention_weights)
    print("- Output:\n", output)
    
    return output, attention_weights

# Example usage
batch_size = 2
num_positions = 4
depth = 5
value_depth = 6

query = np.random.rand(batch_size, num_positions, depth)
key = np.random.rand(batch_size, num_positions, depth)
value = np.random.rand(batch_size, num_positions, value_depth)

print("Parameters:")
print("- Batch Size:", batch_size)  # Expected: 2
print("- Number of Positions:", num_positions)  # Expected: 4
print("- Depth:", depth)  # Expected: 5
print("- Value Depth:", value_depth)  # Expected: 6

print("\nInput Matrices:")
# The following matrices will have random values, so no specific expected values can be provided.
print("- Query matrix:\n", query)
print("- Key matrix:\n", key)
print("- Value matrix:\n", value)

print("\nResults:")
output, attention_weights = scaled_dot_product_attention(query, key, value)
