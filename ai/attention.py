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
    # The transpose(0, 2, 1) operation is applied to a 3-dimensional numpy array, 
    # and it rearranges the dimensions of the array according to the specified order.
    # Given a 3-dimensional array with shape (a, b, c), the transpose(0, 2, 1) operation
    # will rearrange its dimensions to have a shape of (a, c, b).
    matmul_qk = np.matmul(query, key.transpose(0, 2, 1))
    print("Shape of Q:\n ->", query.shape)
    print("Shape of K:\n ->", key.shape)
    print("Shape of K.transpose(0, 2, 1) ->", key.transpose(0, 2, 1).shape)
    print("Shape of np.matmul(query, key.transpose(0, 2, 1)) ->", matmul_qk.shape)
    print("Dot product of Q and K:\n", matmul_qk)
    
    # Scale the dot product by the square root of the depth
    depth = query.shape[-1]
    logits = matmul_qk / np.sqrt(depth)
    print("Scaled dot product:\n", logits)
    
    # Compute the attention weights using softmax
    attention_weights = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    print("Attention weights:\n", attention_weights)
    
    # Compute the output by multiplying the attention weights with the value matrix
    # print shape of attention_weights: (2, 4, 4)
    # print shape of value: (2, 4, 6)
    # print shape of output: (2, 4, 6)
    output = np.matmul(attention_weights, value)
    print("Shape of attention_weights:\n ->", attention_weights.shape)
    print("Shape of value:\n ->", value.shape)
    print("Shape of np.matmul(attention_weights, value):\n ->", output.shape)
    print("Output:\n", output)
    print("Output shape:", output.shape)
    
    return output, attention_weights

# Example usage
batch_size = 2
num_positions = 4
depth = 5
value_depth = 6

query = np.random.rand(batch_size, num_positions, depth)
key = np.random.rand(batch_size, num_positions, depth)
value = np.random.rand(batch_size, num_positions, value_depth)

print("batch_size:", batch_size)
print("num_positions:", num_positions)
print("depth:", depth)
print("value_depth:", value_depth)
print("-----------------------------")
print("Query matrix:\n", query)
print("\nKey matrix:\n", key)
print("\nValue matrix:\n", value)

output, attention_weights = scaled_dot_product_attention(query, key, value)
