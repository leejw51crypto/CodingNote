# Neural Network Implementation Notes

## 1. Forward Propagation Steps
- Input → Hidden Layer: Z1 = X·W1 + b1
  [(10×3) · (3×4) + (1×4) = (10×4)]
- ReLU Activation: A1 = max(0, Z1)
  [(10×4) → (10×4)]
- Hidden → Output Layer: Z2 = A1·W2 + b2
  [(10×4) · (4×2) + (1×2) = (10×2)]
- Softmax Activation: A2 = softmax(Z2)
  [(10×2) → (10×2)]

## 2. Backward Propagation Steps

### Output Layer Gradient (dZ2)
```python
# Combined derivative of softmax and cross-entropy loss
dZ2 = A2 - Y    # Shape: (10×2)

# Full chain rule representation:
∂L/∂Z2 = ∂L/∂A2 · ∂A2/∂Z2    # All intermediate shapes: (10×2)

# Component breakdown:
1. Cross-entropy loss derivative: ∂L/∂A2 = -Y/A2    # Shape: (10×2)
2. Softmax derivative: ∂A2/∂Z2 = A2(1-A2)          # Shape: (10×2)
3. Combined result: A2 - Y
```

### Parameter Updates
```python
# Output Layer Parameters
dL/dW2 = A1^T · dZ2    # Shape: (4×2) = (4×10) · (10×2)
dL/db2 = sum(dZ2)      # Shape: (1×2) = sum((10×2), axis=0)

# Hidden Layer Gradient
dL/dA1 = dZ2 · W2^T    # Shape: (10×4) = (10×2) · (2×4)
dL/dZ1 = dA1 * (Z1 > 0)  # Shape: (10×4) = (10×4) ⊙ (10×4)

# Hidden Layer Parameters
dL/dW1 = X^T · dZ1     # Shape: (3×4) = (3×10) · (10×4)
dL/db1 = sum(dZ1)      # Shape: (1×4) = sum((10×4), axis=0)
```

## 3. Network Architecture Diagram

```
Forward Propagation:
Input         Hidden Layer        Output Layer
(X)           (ReLU)             (Softmax)
                                                        
[x₁]          [ReLU]             [Softmax]    [y₁]     
[x₂] --W1→    [ReLU] ---W2→      [      ] →   [y₂]     
[x₃]          [ReLU]             [Softmax]    
              [ReLU]                          
```

## 4. Matrix Dimensions Summary

### Forward Pass
```
[Input X]        [Hidden Layer]         [Output Layer]      [Target Y]
(10×3)              (10×4)                 (10×2)            (10×2)
   |                   |                      |                |
   |     W1(3×4)       |       W2(4×2)        |                |
   |     b1(1×4)       |       b2(1×2)        |                |
   |                   |                      |                |
[x₁,x₂,x₃] → Z1→[ReLU]→A1 → Z2→[Softmax]→A2 → CrossEntropy → Loss
```

### Parameter Shapes
```
Parameters:                 Activations:           Gradients:
W1: (3×4)                  Z1: (10×4)             dZ1: (10×4)
b1: (1×4)                  A1: (10×4)             dW1: (3×4)
W2: (4×2)                  Z2: (10×2)             db1: (1×4)
b2: (1×2)                  A2: (10×2)             dZ2: (10×2)
                                                  dW2: (4×2)
                                                  db2: (1×2)
```

## 5. Implementation Notes

### Axis Operations in NumPy
- axis=0: Sum along rows (vertically)
- axis=1: Sum along columns (horizontally)

Example:
```
axis=0 (sum down columns):
[1  2  3  4]
[5  6  7  8]
[9  8  7  6]      sum       [20  20  20  20]
[5  4  3  2]    ======>
```

### Chain Rule Application
The complete chain rule for W1:
```
∂L/∂W1 = ∂L/∂A2 · ∂A2/∂Z2 · ∂Z2/∂A1 · ∂A1/∂Z1 · ∂Z1/∂W1
```