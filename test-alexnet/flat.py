import numpy as np
import torch

def show_flattening_simple():
    # Create a simple 2x2 feature map with 2 channels
    feature_map = np.array([
        # Channel 1
        [[1, 2],
         [3, 4]],
        # Channel 2
        [[5, 6],
         [7, 8]]
    ])
    
    print("Original feature map shape (channels, height, width):")
    print(feature_map.shape)  # (2, 2, 2)
    print("\nFeature map content:")
    print("Channel 1:")
    print(feature_map[0])
    print("Channel 2:")
    print(feature_map[1])
    
    # Flatten
    flattened = feature_map.flatten()
    print("\nFlattened array shape:")
    print(flattened.shape)  # (8,)
    print("\nFlattened array values:")
    print(flattened)
    
def show_flattening_pytorch():
    # Create a simple feature map: (batch_size, channels, height, width)
    feature_map = torch.tensor([
        # Batch item 1
        [
            # Channel 1
            [[1, 2],
             [3, 4]],
            # Channel 2
            [[5, 6],
             [7, 8]]
        ]
    ], dtype=torch.float32)
    
    print("\nPyTorch feature map shape (batch, channels, height, width):")
    print(feature_map.shape)  # torch.Size([1, 2, 2, 2])
    
    # Show content
    print("\nFeature map content for first batch:")
    print("Channel 1:")
    print(feature_map[0,0])
    print("Channel 2:")
    print(feature_map[0,1])
    
    # Flatten while keeping batch dimension
    flattened = feature_map.view(feature_map.size(0), -1)
    print("\nFlattened tensor (keeping batch dimension):")
    print(flattened)
    print("Flattened shape:", flattened.shape)  # torch.Size([1, 8])

def show_spatial_relationship():
    # Create a more visual example with a simple pattern
    feature_map = torch.tensor([
        # Batch item 1
        [
            # Channel 1 - vertical line pattern
            [[1, 0, 1],
             [1, 0, 1],
             [1, 0, 1]],
            # Channel 2 - horizontal line pattern
            [[1, 1, 1],
             [0, 0, 0],
             [1, 1, 1]]
        ]
    ], dtype=torch.float32)
    
    print("\nVisual pattern example:")
    print("\nChannel 1 (vertical lines):")
    print(feature_map[0,0])
    print("\nChannel 2 (horizontal lines):")
    print(feature_map[0,1])
    
    # Flatten
    flattened = feature_map.view(feature_map.size(0), -1)
    print("\nFlattened pattern:")
    print(flattened)
    print("Flattened shape:", flattened.shape)  # torch.Size([1, 18])

if __name__ == "__main__":
    print("=== Simple NumPy Example ===")
    show_flattening_simple()
    
    print("\n=== PyTorch Example ===")
    show_flattening_pytorch()
    
    print("\n=== Spatial Relationship Example ===")
    show_spatial_relationship()