
# Forward Propagation

# Forward Propagation Steps

## 1. Input Layer
- X shape: `(10, 3)` (batch_size × input_features)
- Forward equation: `X → Hidden Layer`

## 2. Hidden Layer Linear Transformation
- Formula: `Z1 = X·W1 + b1`
- Shapes: `((10, 3)) · ((3, 4)) + ((1, 4)) = ((10, 4))`
- Each row of X multiplied by W1 gives features for hidden layer

## 3. Hidden Layer Activation
- Formula: `A1 = ReLU(Z1) = max(0, Z1)`
- Shapes: `ReLU((10, 4)) = (10, 4)`
- ReLU sets negative values to 0, keeps positive values

## 4. Output Layer Linear Transformation
- Formula: `Z2 = A1·W2 + b2`
- Shapes: `((10, 4)) · ((4, 2)) + ((1, 2)) = ((10, 2))`
- Hidden layer outputs transformed to output layer

## 5. Output Layer Activation (Softmax)
- Formula: `A2 = softmax(Z2) = exp(Z2)/sum(exp(Z2))`
- Shapes: `softmax((10, 2)) = (10, 2)`
- Softmax normalizes outputs to probability distribution

## Forward Relationships

### 1. Layer Connections
```
Input → [W1,b1] → Z1 → ReLU → A1 → [W2,b2] → Z2 → Softmax → A2
```

### 2. Activation Functions
- Hidden Layer: `ReLU(Z1) = max(0, Z1)`
- Output Layer: `Softmax(Z2) = exp(Z2)/sum(exp(Z2))`

### 3. Shape Transformations
```
Input:        (10, 3)
Hidden Layer: (10, 4) → (10, 4)
Output Layer: (10, 2) → (10, 2)
```

# Backward Propagation

# Backpropagation Steps

## 1. Output Layer Gradient (dZ2)
- Formula: `dZ2 = A2 - Y`  (Combined softmax and cross-entropy derivative)
- Shapes: `((10, 2)) - ((10, 2)) = ((10, 2))`
- This represents the error at the output layer

## 2. Output Layer Weight Gradients
- Formula: `dW2 = A1^T · dZ2`
- Shapes: `((4, 10)) · ((10, 2)) = ((4, 2))`
- Formula: `db2 = sum(dZ2, axis=0)`
- Shapes: `sum((10, 2), axis=0) = ((1, 2))`
- These gradients show how to update W2 and b2

## 3. Hidden Layer Gradient
- Formula: `dA1 = dZ2 · W2^T`  (Backpropagate error)
- Shapes: `((10, 2)) · ((2, 4)) = ((10, 4))`
- Formula: `dZ1 = dA1 ⊙ relu'(Z1)`  (Apply ReLU derivative)
- Shapes: `((10, 4)) ⊙ ((10, 4)) = ((10, 4))`
- Error propagated back through ReLU activation

## 4. Hidden Layer Weight Gradients
- Formula: `dW1 = X^T · dZ1`
- Shapes: `((3, 10)) · ((10, 4)) = ((3, 4))`
- Formula: `db1 = sum(dZ1, axis=0)`
- Shapes: `sum((10, 4), axis=0) = ((1, 4))`
- These gradients show how to update W1 and b1

## Complete Chain Rule
```
For W1: ∂L/∂W1 = ∂L/∂A2 · ∂A2/∂Z2 · ∂Z2/∂A1 · ∂A1/∂Z1 · ∂Z1/∂W1
For W2: ∂L/∂W2 = ∂L/∂A2 · ∂A2/∂Z2 · ∂Z2/∂W2
```

## Forward-Backward Relationships

### 1. Forward Path
```
X → Z1 → A1 → Z2 → A2 → Loss
```

### 2. Backward Path
```
Loss → dZ2 → [dW2, dA1] → dZ1 → dW1
```

### 3. Parameter Update vs Backprop Paths
```
Parameter Updates: dZ2 → dW2, dZ1 → dW1
Error Backprop:   dZ2 → dZ1 (using W2, not dW2)
```

### 4. Key Relationships
```
Forward: Z1 = X·W1,      Backward: dW1 = X^T·dZ1
Forward: Z2 = A1·W2,     Backward: dW2 = A1^T·dZ2
Forward: A1 = ReLU(Z1),  Backward: dZ1 = dA1 ⊙ (Z1 > 0)
```
