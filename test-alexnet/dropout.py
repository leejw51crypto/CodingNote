import numpy as np

def dropout_forward(X, dropout_rate=0.5, is_training=True):
    """
    Applies dropout to input X
    
    Args:
        X: Input array of any shape
        dropout_rate: Probability of dropping a unit (0 to 1)
        is_training: Whether in training mode or test mode
    
    Returns:
        out: Output array, same shape as X
        mask: Dropout mask for backpropagation
    """
    if not is_training:
        return X, None
    
    # Generate dropout mask
    mask = (np.random.rand(*X.shape) > dropout_rate)
    
    # Scale the mask to maintain expected values
    mask = mask / (1 - dropout_rate)
    
    # Apply dropout
    out = X * mask
    
    return out, mask

# Example usage
if __name__ == "__main__":
    # Create sample input
    X = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
    
    print("Original input:")
    print(X)
    
    # Apply dropout during training
    print("\nWith dropout (training):")
    output_train, mask = dropout_forward(X, dropout_rate=0.5, is_training=True)
    print("Output:")
    print(output_train)
    print("Mask:")
    print(mask)
    
    # During testing/inference
    print("\nDuring testing (no dropout):")
    output_test, _ = dropout_forward(X, dropout_rate=0.5, is_training=False)
    print(output_test)