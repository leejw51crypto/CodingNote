import numpy as np

# Create sample input (like a feature map after convolution)
# Shape: (4, 4) - a 4x4 matrix
input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

print("Input shape:", input_matrix.shape)
print("Input matrix:\n", input_matrix)

def max_pool_2x2(input):
    """
    Max pooling with 2x2 window and stride 2
    """
    h, w = input.shape
    # Output will be half the size in both dimensions
    output = np.zeros((h//2, w//2))
    
    # Step by 2 (stride = 2)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            # Take maximum of 2x2 window
            window = input[i:i+2, j:j+2]
            output[i//2, j//2] = np.max(window)
    
    return output

# Apply max pooling
output = max_pool_2x2(input_matrix)
print("\nOutput shape:", output.shape)
print("Output matrix:\n", output)

# AlexNet first pooling (3x3 window, stride 2)
def alexnet_pool(input):
    """
    AlexNet pooling: 3x3 window, stride 2
    """
    h, w = input.shape
    output = np.zeros((h//2, w//2))  # Approximate output size
    
    for i in range(0, h-2, 2):  # stride 2
        for j in range(0, w-2, 2):
            # Take maximum of 3x3 window
            window = input[i:i+3, j:j+3]
            output[i//2, j//2] = np.max(window)
    
    return output

# Create larger input for AlexNet pooling
alexnet_input = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

print("\nAlexNet pooling example:")
print("Input shape:", alexnet_input.shape)
alexnet_output = alexnet_pool(alexnet_input)
print("Output shape:", alexnet_output.shape)
print("Output matrix:\n", alexnet_output)