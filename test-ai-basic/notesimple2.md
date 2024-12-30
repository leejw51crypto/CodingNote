# Simple Neural Network Backpropagation

## Network Architecture
```
Input (X) → Hidden Layer (A1) → Output Layer (A2)
X.shape=(10,3)  A1.shape=(10,4)  A2.shape=(10,2)
W1.shape=(3,4)  W2.shape=(4,2)
b1.shape=(1,4)  b2.shape=(1,2)
```

## Forward Propagation
1. Hidden Layer:
   ```
   Z1 = X·W1 + b1    # X(10,3)·W1(3,4) + b1(1,4) → Z1(10,4)
   A1 = ReLU(Z1)     # A1(10,4) = max(0, Z1)
   ```

2. Output Layer:
   ```
   Z2 = A1·W2 + b2   # A1(10,4)·W2(4,2) + b2(1,2) → Z2(10,2)
   A2 = Softmax(Z2)  # A2(10,2) = probability distribution
   ```

## Backward Propagation

1. Output Layer Gradient:
   ```
   dZ2 = A2 - Y      # dZ2(10,2) = A2(10,2) - Y(10,2)
   dW2 = A1ᵀ·dZ2     # A1ᵀ(4,10)·dZ2(10,2) → dW2(4,2)
   db2 = sum(dZ2)    # dZ2(10,2) → db2(1,2)
   ```

2. Hidden Layer Gradient:
   ```
   dZ1 = (dZ2·W2ᵀ)⊙(Z1>0)  # [dZ2(10,2)·W2ᵀ(2,4)]⊙mask(10,4) → dZ1(10,4)
   dW1 = Xᵀ·dZ1            # Xᵀ(3,10)·dZ1(10,4) → dW1(3,4)
   db1 = sum(dZ1)          # dZ1(10,4) → db1(1,4)
   ```

## Key Points
1. Gradient Flow:
   ```
   Loss → dZ2(10,2)
      ↙            ↘
   dW2(4,2)        dZ1(10,4)
                       ↓
                    dW1(3,4)
   ```

2. Important Notes:
   - All shapes preserve batch size (10) in first dimension
   - ReLU derivative: 1 where Z1>0, 0 where Z1≤0
   - Broadcasting: b1(1,4) and b2(1,2) broadcast to batch dimension
   - Weight matrices: W1(3,4) transforms from 3→4 features
                     W2(4,2) transforms from 4→2 features

3. Initialization:
   - W1, W2: Initialize with small random values (e.g., He initialization)
   - b1, b2: Initialize to zeros