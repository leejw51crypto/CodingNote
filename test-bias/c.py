import numpy as np

def softmax(z):
    """
    Computes the softmax output given an input tensor.

    Args:
        z (numpy.ndarray): Input tensor of shape (batch_size, num_classes).

    Returns:
        numpy.ndarray: Softmax output tensor of shape (batch_size, num_classes).
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_jacobian(z):
    """
    Computes the Jacobian matrix of the softmax function for a given input tensor.

    Args:
        z (numpy.ndarray): Input tensor of shape (batch_size, num_classes).

    Returns:
        numpy.ndarray: Jacobian matrix of shape (num_classes, num_classes).
    """
    sm = softmax(z)
    num_classes = z.shape[1]
    jacobian = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                jacobian[i, j] = sm[0, i] * (1 - sm[0, i])
            else:
                jacobian[i, j] = -sm[0, i] * sm[0, j]
    return jacobian

def softmax_gradient(z, grad_output):
    """
    Computes the gradient of the loss function with respect to the input of the softmax layer.

    Args:
        z (numpy.ndarray): Input tensor of shape (batch_size, num_classes).
        grad_output (numpy.ndarray): Gradient of the loss with respect to the softmax output,
                                     of shape (batch_size, num_classes).

    Returns:
        numpy.ndarray: Gradient of the loss with respect to the input, of shape (batch_size, num_classes).
    """
    jacobian = softmax_jacobian(z)
    grad_input = np.dot(grad_output, jacobian)
    return grad_input

# Example usage
z = np.array([[1.0, 2.0, 3.0]])  # Input tensor of shape (1, 3)
grad_output = np.array([[0.1, 0.2, 0.7]])  # Gradient of the loss w.r.t. softmax output, of shape (1, 3)

print("Input tensor shape:", z.shape)
print("Gradient output shape:", grad_output.shape)

softmax_output = softmax(z)
print("\nSoftmax output:")
print(softmax_output)
print("Softmax output shape:", softmax_output.shape)

jacobian = softmax_jacobian(z)
print("\nSoftmax Jacobian:")
print(jacobian)
print("Jacobian shape:", jacobian.shape)

print(f"grad_output shape {grad_output.shape}")
print(f"z shape {z.shape}")
grad_input = softmax_gradient(z, grad_output)
print("\nGradient of the loss with respect to the input:")
print(grad_input)
print("Gradient input shape:", grad_input.shape)
