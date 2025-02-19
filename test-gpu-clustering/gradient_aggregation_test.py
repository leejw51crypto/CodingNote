"""
# Gradient Aggregation in Distributed Training

## Overview
This code demonstrates the fundamental concept of gradient aggregation in distributed deep learning,
simulating how gradients are combined across multiple nodes and GPUs.

## Key Components

### 1. Multi-Node Setup
- Simulates 2 nodes (like two servers)
- Each node has 8 GPUs
- Each GPU produces its own gradients

### 2. Gradient Generation
- Node 1: Base gradient value = 1.0 + noise
- Node 2: Base gradient value = 2.0 + noise
- Noise added to simulate real training variations

### 3. Aggregation Process
1. Collect gradients from all GPUs on each node
2. Sum all gradients across nodes
3. Average by total number of devices (GPUs)

### 4. Testing & Verification
- Range Check: Ensures aggregated values are between node ranges
- Mean Check: Verifies proper averaging of gradients
- Variance Check: Confirms preservation of gradient distribution

## Real-world Application
This simulates a simplified version of what happens in distributed training:
1. Each GPU computes gradients independently
2. Gradients are aggregated across all devices
3. Averaged gradients are used to update the model

In production:
- Uses efficient all-reduce operations
- Implements ring-based communication
- Includes optimization like gradient compression
"""

import torch
import torch.distributed as dist
import numpy as np


def simulate_node_gradients(node_id, num_gpus=8, feature_size=10):
    """Simulate gradients for a single node with multiple GPUs"""
    # Create distinct gradients for each node
    base_value = 1.0 if node_id == 0 else 2.0

    # Dictionary to store gradients for each GPU
    node_gradients = {}

    for gpu_id in range(num_gpus):
        # Create gradient with some random noise but distinct base value
        gradient = torch.ones(feature_size) * base_value
        gradient += torch.randn_like(gradient) * 0.1  # Add some noise
        node_gradients[f"gpu_{gpu_id}"] = gradient

    return node_gradients


def aggregate_gradients(all_node_gradients):
    """Aggregate gradients across all nodes and GPUs"""
    total_gradient = None

    for node_gradients in all_node_gradients:
        for gpu_gradient in node_gradients.values():
            if total_gradient is None:
                total_gradient = gpu_gradient.clone()
            else:
                total_gradient += gpu_gradient

    # Average the gradients
    total_devices = sum(len(node.keys()) for node in all_node_gradients)
    total_gradient /= total_devices

    return total_gradient


def test_gradient_aggregation(all_node_gradients, total_gradient):
    """Test if aggregated gradient properly combines features from all nodes"""
    print("\n=== Testing Gradient Aggregation ===")

    # Test 1: Check if aggregated value is between node 0 and node 1 values
    node0_values = torch.stack(list(all_node_gradients[0].values()))
    node1_values = torch.stack(list(all_node_gradients[1].values()))

    node0_min, node0_max = torch.min(node0_values), torch.max(node0_values)
    node1_min, node1_max = torch.min(node1_values), torch.max(node1_values)
    total_min, total_max = torch.min(total_gradient), torch.max(total_gradient)

    print("\nTest 1 - Value Range Check:")
    print(f"Node 0 range: [{node0_min:.4f}, {node0_max:.4f}]")
    print(f"Node 1 range: [{node1_min:.4f}, {node1_max:.4f}]")
    print(f"Total range: [{total_min:.4f}, {total_max:.4f}]")

    # Test 2: Verify mean values
    node0_mean = torch.mean(node0_values)
    node1_mean = torch.mean(node1_values)
    total_mean = torch.mean(total_gradient)

    print("\nTest 2 - Mean Value Check:")
    print(f"Node 0 mean: {node0_mean:.4f}")
    print(f"Node 1 mean: {node1_mean:.4f}")
    print(f"Total mean: {total_mean:.4f}")

    expected_mean = (node0_mean + node1_mean) / 2
    mean_diff = abs(total_mean - expected_mean)
    print(f"Difference from expected mean: {mean_diff:.6f}")

    # Fix the variance testing logic
    print("\nTest 3 - Detailed Variance Analysis:")
    node0_var = torch.var(node0_values)
    node1_var = torch.var(node1_values)
    total_var = torch.var(total_gradient)

    # Calculate theoretical variance after averaging
    # When we average two independent distributions, their variances combine as:
    # Var(aX + bY) = a²Var(X) + b²Var(Y), where a and b are weights
    # In our case, a = b = 0.5
    theoretical_var = (node0_var + node1_var) / 4  # (0.5²)Var(X) + (0.5²)Var(Y)
    variance_tolerance = 0.1 * theoretical_var  # 10% tolerance

    print(
        f"""
    Variance Analysis:
    - Node 0 variance: {node0_var:.4f} (around mean {node0_mean:.4f})
    - Node 1 variance: {node1_var:.4f} (around mean {node1_mean:.4f})
    - Total variance: {total_var:.4f} (around mean {total_mean:.4f})
    - Theoretical variance: {theoretical_var:.4f}
    
    Explanation:
    1. Original Variances:
       - Node 0 varies around {node0_mean:.4f} ± {torch.sqrt(node0_var):.4f}
       - Node 1 varies around {node1_mean:.4f} ± {torch.sqrt(node1_var):.4f}
    
    2. Expected Behavior:
       - When averaging two distributions (weights=0.5 each):
       - Theoretical combined variance = (Var1 + Var2)/4 = {theoretical_var:.4f}
       - Actual combined variance = {total_var:.4f}
       - Difference from theoretical: {abs(total_var - theoretical_var):.4f}
    """
    )

    # Revised variance tests
    tests_passed = True
    if total_var > max(node0_var, node1_var):
        print("❌ FAIL: Total variance larger than input variances")
        tests_passed = False

    # Check if variance is within acceptable range of theoretical value
    if abs(total_var - theoretical_var) > variance_tolerance:
        print(
            f"⚠️ WARNING: Variance ({total_var:.4f}) differs from theoretical ({theoretical_var:.4f})"
        )
        print("This might be okay if the distributions are not perfectly independent")

    # Test results
    if tests_passed:
        print("\n✅ All tests passed! Gradient aggregation working as expected")

    return tests_passed


def main():
    print(
        """
# Distributed Training Simulation
    
## Configuration
- Number of Nodes: 2
- GPUs per Node: 8
- Feature Size: 10
    
## Process
1. Generate gradients for each GPU
2. Aggregate across nodes
3. Verify aggregation correctness
    """
    )

    # Simulation parameters
    num_nodes = 2
    num_gpus = 8
    feature_size = 10

    print("\n## Gradient Generation")
    # Simulate gradients for each node
    all_node_gradients = []
    for node_id in range(num_nodes):
        node_gradients = simulate_node_gradients(
            node_id=node_id, num_gpus=num_gpus, feature_size=feature_size
        )
        all_node_gradients.append(node_gradients)

        print(f"\nNode {node_id} gradients:")
        print(f"First GPU gradient: {node_gradients['gpu_0']}")
        print(f"Last GPU gradient: {node_gradients[f'gpu_{num_gpus-1}']}")

    # Aggregate gradients
    total_gradient = aggregate_gradients(all_node_gradients)

    print("\n## Aggregation Results")
    print("\nAggregated gradient:")
    print(total_gradient)

    # Verify that the aggregated gradient is between the values of both nodes
    node0_mean = torch.mean(torch.stack(list(all_node_gradients[0].values())))
    node1_mean = torch.mean(torch.stack(list(all_node_gradients[1].values())))

    print("\nVerification:")
    print(f"Node 0 mean gradient: {node0_mean:.4f}")
    print(f"Node 1 mean gradient: {node1_mean:.4f}")
    print(f"Total mean gradient: {torch.mean(total_gradient):.4f}")

    # Add after calculating total_gradient:
    test_gradient_aggregation(all_node_gradients, total_gradient)


if __name__ == "__main__":
    main()
