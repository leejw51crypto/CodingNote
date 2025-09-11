#!/usr/bin/env python3
"""
PyTorch Interactive Learning TUI
================================
Beautiful terminal UI for learning PyTorch functions interactively.
Shows examples and executes user code with real-time results.

Run with: python pytorch_tui.py
"""

import math
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


# Device setup
def setup_device():
    """Setup the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_info = "NVIDIA GPU (CUDA)"
    else:
        device = torch.device("cpu")
        device_info = "CPU"
    return device, device_info


DEVICE, DEVICE_INFO = setup_device()
console = Console()

# Function catalog with device-aware examples
FUNCTION_CATALOG = {
    "tensor": {
        "signature": "torch.tensor(data, dtype=None, device=None, requires_grad=False)",
        "args": "data: array-like, dtype: data type, device: cpu/cuda/mps, requires_grad: bool",
        "purpose": "Create tensor from Python data",
        "example": f"torch.tensor([1, 2, 3, 4], device='{DEVICE}')",
        "category": "Creation",
    },
    "zeros": {
        "signature": "torch.zeros(*size, dtype=None, device=None)",
        "args": "*size: dimensions, dtype: data type, device: cpu/cuda/mps",
        "purpose": "Create tensor filled with zeros",
        "example": f"torch.zeros(3, 4, device='{DEVICE}')",
        "category": "Creation",
    },
    "ones": {
        "signature": "torch.ones(*size, dtype=None, device=None)",
        "args": "*size: dimensions, dtype: data type, device: cpu/cuda/mps",
        "purpose": "Create tensor filled with ones",
        "example": f"torch.ones(2, 3, device='{DEVICE}')",
        "category": "Creation",
    },
    "randn": {
        "signature": "torch.randn(*size, dtype=None, device=None)",
        "args": "*size: dimensions, dtype: data type, device: cpu/cuda/mps",
        "purpose": "Random normal distribution N(0,1)",
        "example": f"torch.randn(2, 3, device='{DEVICE}')",
        "category": "Creation",
    },
    "add": {
        "signature": "torch.add(input, other, alpha=1)",
        "args": "input: tensor1, other: tensor2, alpha: multiplier for other",
        "purpose": "Element-wise addition",
        "example": "torch.add(a, b)",
        "category": "Operations",
    },
    "matmul": {
        "signature": "torch.matmul(input, other)",
        "args": "input: tensor1, other: tensor2",
        "purpose": "Matrix multiplication",
        "example": "torch.matmul(a, b)",
        "category": "Operations",
    },
    "sum": {
        "signature": "torch.sum(input, dim=None, keepdim=False)",
        "args": "input: tensor, dim: dimension to sum (None=all), keepdim: keep dimensions",
        "purpose": "Sum elements along dimension",
        "example": "torch.sum(x)",
        "category": "Reductions",
    },
    "relu": {
        "signature": "F.relu(input, inplace=False)",
        "args": "input: tensor, inplace: modify input tensor",
        "purpose": "Rectified Linear Unit: max(0, x)",
        "example": "F.relu(x)",
        "category": "Neural Network",
    },
    "softmax": {
        "signature": "F.softmax(input, dim, dtype=None)",
        "args": "input: tensor, dim: dimension to apply softmax",
        "purpose": "Softmax: exp(x_i) / sum(exp(x_j))",
        "example": "F.softmax(x, dim=1)",
        "category": "Neural Network",
    },
    # Additional Tensor Creation Functions
    "eye": {
        "signature": "torch.eye(n, m=None, dtype=None)",
        "args": "n: rows, m: cols (default n), dtype: data type",
        "purpose": "Create identity matrix",
        "example": f"torch.eye(4, device='{DEVICE}')",
        "category": "Creation",
    },
    "arange": {
        "signature": "torch.arange(start, end, step=1)",
        "args": "start: begin, end: end (exclusive), step: increment",
        "purpose": "Create tensor with evenly spaced values",
        "example": "torch.arange(0, 10, 2)",
        "category": "Creation",
    },
    "linspace": {
        "signature": "torch.linspace(start, end, steps)",
        "args": "start: begin, end: end (inclusive), steps: number of points",
        "purpose": "Create tensor with linearly spaced values",
        "example": "torch.linspace(0, 1, 5)",
        "category": "Creation",
    },
    "rand": {
        "signature": "torch.rand(*size, dtype=None, device=None)",
        "args": "*size: dimensions, dtype: data type, device: cpu/cuda/mps",
        "purpose": "Random uniform distribution [0, 1)",
        "example": f"torch.rand(2, 3, device='{DEVICE}')",
        "category": "Creation",
    },
    "randint": {
        "signature": "torch.randint(low, high, size, dtype=None)",
        "args": "low: minimum, high: maximum (exclusive), size: dimensions",
        "purpose": "Random integers in range [low, high)",
        "example": "torch.randint(0, 10, (3, 4))",
        "category": "Creation",
    },
    # Basic Operations
    "sub": {
        "signature": "torch.sub(input, other, alpha=1)",
        "args": "input: tensor1, other: tensor2, alpha: multiplier for other",
        "purpose": "Element-wise subtraction",
        "example": "torch.sub(a, b)",
        "category": "Operations",
    },
    "mul": {
        "signature": "torch.mul(input, other)",
        "args": "input: tensor1, other: tensor2",
        "purpose": "Element-wise multiplication",
        "example": "torch.mul(a, b)",
        "category": "Operations",
    },
    "div": {
        "signature": "torch.div(input, other)",
        "args": "input: tensor1, other: tensor2",
        "purpose": "Element-wise division",
        "example": "torch.div(a, b)",
        "category": "Operations",
    },
    "pow": {
        "signature": "torch.pow(input, exponent)",
        "args": "input: base tensor, exponent: power (scalar or tensor)",
        "purpose": "Element-wise power",
        "example": "torch.pow(a, 2)",
        "category": "Operations",
    },
    # Tensor Manipulation
    "reshape": {
        "signature": "tensor.reshape(*shape)",
        "args": "*shape: new dimensions (must have same number of elements)",
        "purpose": "Change tensor shape (returns view if possible)",
        "example": "x.reshape(3, 2)",
        "category": "Manipulation",
    },
    "view": {
        "signature": "tensor.view(*shape)",
        "args": "*shape: new dimensions (tensor must be contiguous)",
        "purpose": "Change tensor shape (always returns view)",
        "example": "x.view(6, 1)",
        "category": "Manipulation",
    },
    "transpose": {
        "signature": "torch.transpose(input, dim0, dim1)",
        "args": "input: tensor, dim0: first dimension, dim1: second dimension",
        "purpose": "Swap two dimensions",
        "example": "x.transpose(0, 1)",
        "category": "Manipulation",
    },
    "squeeze": {
        "signature": "torch.squeeze(input, dim=None)",
        "args": "input: tensor, dim: dimension to squeeze (None for all size-1 dims)",
        "purpose": "Remove dimensions of size 1",
        "example": "torch.randn(1, 3, 1, 4).squeeze()",
        "category": "Manipulation",
    },
    "unsqueeze": {
        "signature": "torch.unsqueeze(input, dim)",
        "args": "input: tensor, dim: position to insert new dimension",
        "purpose": "Add dimension of size 1",
        "example": "x.unsqueeze(0)",
        "category": "Manipulation",
    },
    "flatten": {
        "signature": "torch.flatten(input, start_dim=0, end_dim=-1)",
        "args": "input: tensor, start_dim: first dim to flatten, end_dim: last dim to flatten",
        "purpose": "Flatten tensor dimensions",
        "example": "x.flatten()",
        "category": "Manipulation",
    },
    # Indexing and Slicing
    "index_select": {
        "signature": "torch.index_select(input, dim, index)",
        "args": "input: tensor, dim: dimension to select from, index: indices tensor",
        "purpose": "Select elements along dimension",
        "example": "torch.index_select(x, 0, torch.tensor([0, 2]))",
        "category": "Indexing",
    },
    "masked_select": {
        "signature": "torch.masked_select(input, mask)",
        "args": "input: tensor, mask: boolean tensor same shape as input",
        "purpose": "Select elements where mask is True",
        "example": "torch.masked_select(x, x > 0.5)",
        "category": "Indexing",
    },
    # Reduction Operations
    "mean": {
        "signature": "torch.mean(input, dim=None, keepdim=False)",
        "args": "input: tensor, dim: dimension to average, keepdim: keep dimensions",
        "purpose": "Average elements along dimension",
        "example": "torch.mean(x)",
        "category": "Reductions",
    },
    "max": {
        "signature": "torch.max(input, dim=None, keepdim=False)",
        "args": "input: tensor, dim: dimension to find max, keepdim: keep dimensions",
        "purpose": "Find maximum values",
        "example": "torch.max(x)",
        "category": "Reductions",
    },
    "min": {
        "signature": "torch.min(input, dim=None, keepdim=False)",
        "args": "input: tensor, dim: dimension to find min, keepdim: keep dimensions",
        "purpose": "Find minimum values",
        "example": "torch.min(x)",
        "category": "Reductions",
    },
    "std": {
        "signature": "torch.std(input, dim=None, unbiased=True, keepdim=False)",
        "args": "input: tensor, dim: dimension, unbiased: Bessel's correction, keepdim: keep dims",
        "purpose": "Standard deviation",
        "example": "torch.std(x)",
        "category": "Reductions",
    },
    "var": {
        "signature": "torch.var(input, dim=None, unbiased=True, keepdim=False)",
        "args": "input: tensor, dim: dimension, unbiased: Bessel's correction, keepdim: keep dims",
        "purpose": "Variance",
        "example": "torch.var(x)",
        "category": "Reductions",
    },
    # Mathematical Functions
    "sin": {
        "signature": "torch.sin(input)",
        "args": "input: tensor in radians",
        "purpose": "Sine function",
        "example": "torch.sin(x)",
        "category": "Math",
    },
    "cos": {
        "signature": "torch.cos(input)",
        "args": "input: tensor in radians",
        "purpose": "Cosine function",
        "example": "torch.cos(x)",
        "category": "Math",
    },
    "tan": {
        "signature": "torch.tan(input)",
        "args": "input: tensor in radians",
        "purpose": "Tangent function",
        "example": "torch.tan(x)",
        "category": "Math",
    },
    "exp": {
        "signature": "torch.exp(input)",
        "args": "input: tensor",
        "purpose": "Exponential function",
        "example": "torch.exp(x)",
        "category": "Math",
    },
    "log": {
        "signature": "torch.log(input)",
        "args": "input: tensor (positive values)",
        "purpose": "Natural logarithm",
        "example": "torch.log(x)",
        "category": "Math",
    },
    "log10": {
        "signature": "torch.log10(input)",
        "args": "input: tensor (positive values)",
        "purpose": "Base-10 logarithm",
        "example": "torch.log10(x)",
        "category": "Math",
    },
    "sqrt": {
        "signature": "torch.sqrt(input)",
        "args": "input: tensor (non-negative values)",
        "purpose": "Element-wise square root",
        "example": "torch.sqrt(x)",
        "category": "Math",
    },
    "round": {
        "signature": "torch.round(input)",
        "args": "input: tensor",
        "purpose": "Round to nearest integer",
        "example": "torch.round(x)",
        "category": "Math",
    },
    "floor": {
        "signature": "torch.floor(input)",
        "args": "input: tensor",
        "purpose": "Round down to integer",
        "example": "torch.floor(x)",
        "category": "Math",
    },
    "ceil": {
        "signature": "torch.ceil(input)",
        "args": "input: tensor",
        "purpose": "Round up to integer",
        "example": "torch.ceil(x)",
        "category": "Math",
    },
    "clamp": {
        "signature": "torch.clamp(input, min=None, max=None)",
        "args": "input: tensor, min: minimum value, max: maximum value",
        "purpose": "Clamp values to range [min, max]",
        "example": "torch.clamp(x, min=0, max=1)",
        "category": "Math",
    },
    # Linear Algebra
    "mm": {
        "signature": "torch.mm(input, mat2)",
        "args": "input: 2D tensor, mat2: 2D tensor",
        "purpose": "Matrix multiplication for 2D tensors",
        "example": "torch.mm(a, b)",
        "category": "Linear Algebra",
    },
    "bmm": {
        "signature": "torch.bmm(input, mat2)",
        "args": "input: 3D tensor (b×n×m), mat2: 3D tensor (b×m×p)",
        "purpose": "Batch matrix multiplication",
        "example": "torch.bmm(batch_a, batch_b)",
        "category": "Linear Algebra",
    },
    "mv": {
        "signature": "torch.mv(input, vec)",
        "args": "input: 2D tensor, vec: 1D tensor",
        "purpose": "Matrix-vector multiplication",
        "example": "torch.mv(matrix, vector)",
        "category": "Linear Algebra",
    },
    "dot": {
        "signature": "torch.dot(input, other)",
        "args": "input: 1D tensor, other: 1D tensor (same size)",
        "purpose": "Dot product of two 1D tensors",
        "example": "torch.dot(v1, v2)",
        "category": "Linear Algebra",
    },
    "det": {
        "signature": "torch.det(input)",
        "args": "input: square 2D tensor",
        "purpose": "Determinant of matrix",
        "example": "torch.det(matrix)",
        "category": "Linear Algebra",
    },
    "inverse": {
        "signature": "torch.inverse(input)",
        "args": "input: square 2D tensor",
        "purpose": "Matrix inverse",
        "example": "torch.inverse(matrix)",
        "category": "Linear Algebra",
    },
    # Neural Network Functions
    "sigmoid": {
        "signature": "F.sigmoid(input)",
        "args": "input: tensor",
        "purpose": "Sigmoid function: 1/(1+exp(-x))",
        "example": "F.sigmoid(x)",
        "category": "Neural Network",
    },
    "cross_entropy": {
        "signature": "F.cross_entropy(input, target, weight=None, reduction='mean')",
        "args": "input: logits (N,C), target: class indices (N,), weight: class weights",
        "purpose": "Cross entropy loss (combines softmax + NLL)",
        "example": "F.cross_entropy(logits, labels)",
        "category": "Neural Network",
    },
    "mse_loss": {
        "signature": "F.mse_loss(input, target, reduction='mean')",
        "args": "input: predictions, target: ground truth, reduction: 'mean'/'sum'/'none'",
        "purpose": "Mean Squared Error: (pred - target)²",
        "example": "F.mse_loss(pred, target)",
        "category": "Neural Network",
    },
    "max_pool2d": {
        "signature": "F.max_pool2d(input, kernel_size, stride=None, padding=0)",
        "args": "input: 4D tensor (N,C,H,W), kernel_size: pool size, stride: step size",
        "purpose": "2D max pooling",
        "example": "F.max_pool2d(torch.randn(1, 1, 4, 4), 2)",
        "category": "Neural Network",
    },
    "avg_pool2d": {
        "signature": "F.avg_pool2d(input, kernel_size, stride=None, padding=0)",
        "args": "input: 4D tensor (N,C,H,W), kernel_size: pool size, stride: step size",
        "purpose": "2D average pooling",
        "example": "F.avg_pool2d(torch.randn(1, 1, 4, 4), 2)",
        "category": "Neural Network",
    },
    "dropout": {
        "signature": "F.dropout(input, p=0.5, training=False, inplace=False)",
        "args": "input: tensor, p: dropout probability, training: training mode",
        "purpose": "Randomly zero elements during training",
        "example": "F.dropout(x, p=0.5, training=True)",
        "category": "Neural Network",
    },
    # Advanced Operations
    "einsum": {
        "signature": "torch.einsum(equation, *operands)",
        "args": "equation: Einstein notation string, operands: input tensors",
        "purpose": "Flexible tensor operations using Einstein notation",
        "example": "torch.einsum('ij,jk->ik', a, b)",
        "category": "Advanced",
    },
    "topk": {
        "signature": "torch.topk(input, k, dim=None, largest=True, sorted=True)",
        "args": "input: tensor, k: number of elements, dim: dimension, largest: top/bottom",
        "purpose": "Get k largest/smallest elements",
        "example": "torch.topk(x, k=2, dim=1)",
        "category": "Advanced",
    },
    "gather": {
        "signature": "torch.gather(input, dim, index)",
        "args": "input: source tensor, dim: dimension, index: indices to gather",
        "purpose": "Gather values along dimension using indices",
        "example": "torch.gather(x, 1, gather_idx)",
        "category": "Advanced",
    },
    "cat": {
        "signature": "torch.cat(tensors, dim=0)",
        "args": "tensors: list of tensors, dim: dimension to concatenate",
        "purpose": "Concatenate tensors along existing dimension",
        "example": "torch.cat([t1, t2], dim=0)",
        "category": "Advanced",
    },
    "stack": {
        "signature": "torch.stack(tensors, dim=0)",
        "args": "tensors: list of tensors, dim: dimension to stack",
        "purpose": "Stack tensors along new dimension",
        "example": "torch.stack([t1, t2])",
        "category": "Advanced",
    },
    # Utilities
    "clone": {
        "signature": "tensor.clone()",
        "args": "Creates copy with gradient tracking",
        "purpose": "Copy tensor with gradient tracking",
        "example": "x.clone()",
        "category": "Utilities",
    },
    "detach": {
        "signature": "tensor.detach()",
        "args": "Remove tensor from computation graph",
        "purpose": "Remove from computation graph",
        "example": "x.detach()",
        "category": "Utilities",
    },
    "to": {
        "signature": "tensor.to(device)",
        "args": "device: target device (cpu/cuda/mps)",
        "purpose": "Move tensor to device",
        "example": f"x.to('{DEVICE}')",
        "category": "Utilities",
    },
    "float": {
        "signature": "tensor.float()",
        "args": "Convert to float32 dtype",
        "purpose": "Convert to float32",
        "example": "x.float()",
        "category": "Utilities",
    },
    "double": {
        "signature": "tensor.double()",
        "args": "Convert to float64 dtype",
        "purpose": "Convert to float64",
        "example": "x.double()",
        "category": "Utilities",
    },
    "int": {
        "signature": "tensor.int()",
        "args": "Convert to int32 dtype",
        "purpose": "Convert to int32",
        "example": "x.int()",
        "category": "Utilities",
    },
    # Additional important functions
    "logspace": {
        "signature": "torch.logspace(start, end, steps, base=10)",
        "args": "start: log_start, end: log_end, steps: number, base: logarithm base",
        "purpose": "Create logarithmically spaced tensor",
        "example": "torch.logspace(0, 2, 5)",
        "category": "Creation",
    },
    "eigvals": {
        "signature": "torch.linalg.eigvals(input)",
        "args": "input: square tensor",
        "purpose": "Compute eigenvalues of matrix",
        "example": "torch.linalg.eigvals(matrix)",
        "category": "Linear Algebra",
    },
    "svd": {
        "signature": "torch.svd(input, some=True, compute_uv=True)",
        "args": "input: tensor, some: reduced SVD, compute_uv: compute U,V",
        "purpose": "Singular Value Decomposition",
        "example": "U, S, V = torch.svd(matrix)",
        "category": "Linear Algebra",
    },
    "from_numpy": {
        "signature": "torch.from_numpy(ndarray)",
        "args": "ndarray: NumPy array",
        "purpose": "Convert NumPy array to tensor",
        "example": "torch.from_numpy(np_array)",
        "category": "Utilities",
    },
    "numpy": {
        "signature": "tensor.numpy()",
        "args": "Convert tensor to NumPy array (CPU only)",
        "purpose": "Convert to NumPy array",
        "example": "x.cpu().numpy()",
        "category": "Utilities",
    },
    "no_grad": {
        "signature": "with torch.no_grad():",
        "args": "Context manager to disable gradient computation",
        "purpose": "Disable gradient tracking",
        "example": "with torch.no_grad(): y = x * 2",
        "category": "Gradients",
    },
    "requires_grad": {
        "signature": "tensor.requires_grad_(requires_grad=True)",
        "args": "requires_grad: bool to enable/disable gradient tracking",
        "purpose": "Enable gradient computation",
        "example": "x.clone().detach().requires_grad_(True)",
        "category": "Gradients",
    },
    "backward": {
        "signature": "tensor.backward(gradient=None, retain_graph=None)",
        "args": "gradient: gradient w.r.t. tensor, retain_graph: keep computation graph",
        "purpose": "Compute gradients via backpropagation",
        "example": "torch.tensor([1.0, 2.0, 3.0], requires_grad=True).sum().backward()",
        "category": "Gradients",
    },
    "clip_grad_norm_": {
        "signature": "torch.nn.utils.clip_grad_norm_(parameters, max_norm)",
        "args": "parameters: model parameters, max_norm: maximum gradient norm",
        "purpose": "Clip gradients to prevent exploding",
        "example": "torch.nn.utils.clip_grad_norm_(nn.Linear(3, 2).parameters(), 1.0)",
        "category": "Gradients",
    },
    "is_available": {
        "signature": "torch.cuda.is_available() / torch.backends.mps.is_available()",
        "args": "Check if GPU is available",
        "purpose": "Check GPU availability",
        "example": "torch.cuda.is_available()",
        "category": "Device",
    },
    "device": {
        "signature": "torch.device(device_name)",
        "args": "device_name: 'cpu', 'cuda', 'mps', or 'cuda:0' etc",
        "purpose": "Create device object",
        "example": "torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "category": "Device",
    },
    "broadcasting": {
        "signature": "Broadcasting rules",
        "args": "Automatic shape alignment for element-wise ops",
        "purpose": "Element-wise ops on different shapes",
        "example": "torch.randn(3, 1) + torch.randn(1, 4) # -> (3, 4)",
        "category": "Advanced",
    },
    "permute": {
        "signature": "tensor.permute(*dims)",
        "args": "*dims: new dimension order",
        "purpose": "Reorder tensor dimensions",
        "example": "x.permute(2, 0, 1)",
        "category": "Manipulation",
    },
    "contiguous": {
        "signature": "tensor.contiguous()",
        "args": "Make tensor contiguous in memory",
        "purpose": "Ensure tensor is contiguous",
        "example": "x.contiguous()",
        "category": "Utilities",
    },
    "norm": {
        "signature": "torch.norm(input, p='fro', dim=None, keepdim=False)",
        "args": "input: tensor, p: norm type (1, 2, 'fro'), dim: dimension",
        "purpose": "Compute vector/matrix norm",
        "example": "torch.norm(x, p=2)",
        "category": "Linear Algebra",
    },
    "where": {
        "signature": "torch.where(condition, x, y)",
        "args": "condition: bool tensor, x: true values, y: false values",
        "purpose": "Conditional selection",
        "example": "torch.where(x > 0, x, torch.zeros_like(x))",
        "category": "Advanced",
    },
    "argmax": {
        "signature": "torch.argmax(input, dim=None, keepdim=False)",
        "args": "input: tensor, dim: dimension to find argmax",
        "purpose": "Find indices of maximum values",
        "example": "torch.argmax(x, dim=1)",
        "category": "Reductions",
    },
    "argmin": {
        "signature": "torch.argmin(input, dim=None, keepdim=False)",
        "args": "input: tensor, dim: dimension to find argmin",
        "purpose": "Find indices of minimum values",
        "example": "torch.argmin(x, dim=1)",
        "category": "Reductions",
    },
}


def create_sample_tensors():
    """Create sample tensors for user to work with"""
    samples = {
        "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE),
        "b": torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=DEVICE),
        "x": torch.randn(2, 3, device=DEVICE),
        "y": torch.randn(3, 4, device=DEVICE),
        "matrix": torch.randn(3, 3, device=DEVICE),
        "vector": torch.randn(3, device=DEVICE),
        "v1": torch.randn(5, device=DEVICE),
        "v2": torch.randn(5, device=DEVICE),
        "batch_a": torch.randn(2, 3, 4, device=DEVICE),
        "batch_b": torch.randn(2, 4, 5, device=DEVICE),
        "t1": torch.randn(2, 3, device=DEVICE),
        "t2": torch.randn(2, 3, device=DEVICE),
        "logits": torch.randn(3, 5, device=DEVICE),
        "labels": torch.tensor([1, 0, 4], device=DEVICE),
        "pred": torch.randn(3, 2, device=DEVICE),
        "target": torch.randn(3, 2, device=DEVICE),
        "gather_idx": torch.tensor([[0, 2, 1], [1, 0, 2]], device=DEVICE),
        "np_array": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    }
    return samples


def set_default_device():
    """Set PyTorch default device"""
    if DEVICE.type != "cpu":
        try:
            torch.set_default_device(DEVICE)
        except:
            # Fallback for older PyTorch versions
            pass


def format_tensor_result(result):
    """Format tensor result for display"""
    if isinstance(result, torch.Tensor):
        lines = []
        lines.append(f"[bold cyan]Shape:[/bold cyan] {result.shape}")
        lines.append(f"[bold cyan]Dtype:[/bold cyan] {result.dtype}")
        lines.append(f"[bold cyan]Device:[/bold cyan] {result.device}")

        if result.numel() <= 20:
            lines.append(f"[bold cyan]Values:[/bold cyan]\n{result}")
        else:
            lines.append(f"[bold cyan]Values:[/bold cyan] (showing first 10)")
            lines.append(f"{result.flatten()[:10]}...")

        return "\n".join(lines)
    else:
        return str(result)


def show_tensor_info(tensor, name):
    """Show tensor information in a compact format"""
    if isinstance(tensor, torch.Tensor):
        # Format shape nicely
        shape_str = "×".join(map(str, tensor.shape))

        # Show tensor in its actual structure, not flattened
        if tensor.numel() <= 12:
            # Small tensors: show full structure
            tensor_str = (
                str(tensor)
                .replace("tensor(", "")
                .replace(")", "")
                .replace(f", device='{tensor.device}'", "")
            )
            if "\n" in tensor_str:
                # Multi-line tensor - format nicely
                lines = tensor_str.strip().split("\n")
                formatted_lines = []
                for line in lines:
                    line = line.strip()
                    if (
                        line.startswith("[")
                        or line.startswith("-")
                        or any(c.isdigit() for c in line)
                    ):
                        formatted_lines.append("    " + line)
                tensor_str = "\n".join(formatted_lines)
            values_str = f"\n{tensor_str}"
        elif tensor.numel() <= 50:
            # Medium tensors: show first few rows/elements
            if len(tensor.shape) == 1:
                values = tensor[:8].tolist()
                values_str = f"[{', '.join(f'{x:.2f}' for x in values)}...]"
            elif len(tensor.shape) == 2:
                first_rows = min(3, tensor.shape[0])
                sample_tensor = tensor[:first_rows]
                tensor_str = (
                    str(sample_tensor)
                    .replace("tensor(", "")
                    .replace(")", "")
                    .replace(f", device='{tensor.device}'", "")
                )
                values_str = f"\n    {tensor_str}\n    ..."
            else:
                # Multi-dimensional: show shape info
                values_str = f"(showing shape only for large tensor)"
        else:
            # Large tensors: just show dimensions
            values_str = f"(large tensor - {tensor.numel()} elements)"

        return f"[cyan]{name}[/cyan]: shape({shape_str}) {tensor.dtype} on {tensor.device}{values_str}"
    else:
        return f"[cyan]{name}[/cyan]: {tensor} ({type(tensor).__name__})"


def extract_variables_from_code(code, env):
    """Extract variable names that are tensors from the code"""
    import re

    # Find variable names in the code (simple pattern matching)
    var_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
    variables = re.findall(var_pattern, code)

    tensor_vars = []
    for var in variables:
        if (
            var in env
            and isinstance(env[var], torch.Tensor)
            and var not in ["torch", "F", "nn"]
        ):
            tensor_vars.append(var)

    # Remove duplicates while preserving order
    seen = set()
    unique_vars = []
    for var in tensor_vars:
        if var not in seen:
            seen.add(var)
            unique_vars.append(var)

    return unique_vars


def execute_code_safely(code, env):
    """Execute code and return result safely"""
    try:
        result = eval(code, env)
        return True, result, None
    except SyntaxError:
        try:
            exec(code, env)
            return True, None, None
        except Exception as e:
            return False, None, str(e)
    except Exception as e:
        return False, None, str(e)


def find_similar_functions(query, func_names, top_n=5):
    """Find functions most similar to the query string"""
    from difflib import SequenceMatcher

    query_lower = query.lower()
    scores = []

    for name in func_names:
        name_lower = name.lower()

        # Exact match
        if query_lower == name_lower:
            scores.append((name, 1.0))
            continue

        # Substring match (higher score if query is in name)
        if query_lower in name_lower:
            scores.append((name, 0.9))
            continue

        # Starts with query
        if name_lower.startswith(query_lower):
            scores.append((name, 0.85))
            continue

        # Calculate similarity score
        similarity = SequenceMatcher(None, query_lower, name_lower).ratio()
        scores.append((name, similarity))

    # Sort by score (descending) and return top matches
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


class PyTorchTUI:
    def __init__(self):
        # Set default device for PyTorch operations
        set_default_device()

        self.samples = create_sample_tensors()
        self.env = self.samples.copy()
        self.env.update(
            {
                "torch": torch,
                "F": F,
                "nn": nn,
                "np": np,
                "math": math,
                "device": DEVICE,
            }
        )

        # Add device-aware helper functions
        self.env["to_device"] = lambda x: x.to(DEVICE) if hasattr(x, "to") else x

    def show_welcome(self):
        """Display welcome screen"""
        welcome_text = Text()
        welcome_text.append(
            "🚀 PyTorch Interactive Learning TUI\n", style="bold magenta"
        )
        welcome_text.append(f"PyTorch Version: {torch.__version__}\n", style="cyan")
        welcome_text.append(f"Device: {DEVICE} ({DEVICE_INFO})\n", style="green")

        # Show device capabilities
        if DEVICE.type == "mps":
            welcome_text.append("✅ MPS (Apple Silicon GPU) enabled\n", style="green")
        elif DEVICE.type == "cuda":
            welcome_text.append("✅ CUDA enabled\n", style="green")
        else:
            welcome_text.append("ℹ️  Using CPU (GPU not available)\n", style="yellow")

        welcome_panel = Panel(
            welcome_text, title="Welcome", border_style="blue", box=box.ROUNDED
        )
        console.print(welcome_panel)

        # Show menu
        menu = Table(title="Learning Modes", box=box.ROUNDED)
        menu.add_column("Option", style="cyan", no_wrap=True)
        menu.add_column("Description", style="white")

        menu.add_row("1", "🎯 Tutorial - Step-by-step guided learning")
        menu.add_row("2", "🎓 Practice - Learn specific functions")
        menu.add_row("3", "🏃 Category Practice - Practice by category")
        menu.add_row("4", "🔍 Free Exploration - Type any PyTorch code")
        menu.add_row("Q", "❌ Quit")

        console.print(menu)
        console.print(
            "\n[dim]You can select by number (1-4) or by typing mode name (tutorial, practice, category, free, explore)[/dim]"
        )

    def show_available_variables(self):
        """Display available variables and their info"""
        var_table = Table(title="📊 Available Variables", box=box.ROUNDED)
        var_table.add_column("Variable", style="cyan", no_wrap=True)
        var_table.add_column("Shape", style="yellow")
        var_table.add_column("Device", style="green")
        var_table.add_column("Description", style="white")

        # Add sample tensors info
        descriptions = {
            "a": "2x2 matrix [[1,2],[3,4]]",
            "b": "2x2 matrix [[5,6],[7,8]]",
            "x": "2x3 random tensor",
            "y": "3x4 random tensor",
            "matrix": "3x3 random matrix",
            "vector": "3-element vector",
            "v1": "5-element vector",
            "v2": "5-element vector",
            "batch_a": "2x3x4 batch tensor",
            "batch_b": "2x4x5 batch tensor",
            "t1": "2x3 random tensor",
            "t2": "2x3 random tensor",
            "logits": "3x5 logits for 5 classes",
            "labels": "Class labels [1,0,4]",
            "pred": "3x2 predictions",
            "target": "3x2 targets",
            "gather_idx": "2x3 gather indices for x",
            "np_array": "NumPy array [[1,2,3],[4,5,6]]",
        }

        for var_name, tensor in self.samples.items():
            if isinstance(tensor, torch.Tensor):
                var_table.add_row(
                    var_name,
                    str(tuple(tensor.shape)),
                    str(tensor.device),
                    descriptions.get(var_name, "Tensor"),
                )
            elif isinstance(tensor, np.ndarray):
                var_table.add_row(
                    var_name,
                    str(tensor.shape),
                    "NumPy/CPU",
                    descriptions.get(var_name, "NumPy array"),
                )

        # Add other useful variables
        var_table.add_row(
            "device", "N/A", str(DEVICE), f"Current device: {DEVICE_INFO}"
        )
        var_table.add_row("torch", "N/A", "N/A", "PyTorch module")
        var_table.add_row("F", "N/A", "N/A", "torch.nn.functional")
        var_table.add_row("nn", "N/A", "N/A", "torch.nn module")

        console.print(var_table)

    def tutorial_mode(self):
        """Interactive tutorial mode"""
        console.print(Panel("🎯 Tutorial Mode", style="bold blue"))

        lessons = [
            (
                "Creating Tensors",
                [
                    ("torch.tensor([1, 2, 3])", "Create a tensor from a list"),
                    ("torch.zeros(2, 3)", "Create a tensor of zeros"),
                    ("torch.randn(2, 3)", "Create random tensor"),
                ],
            ),
            (
                "Basic Operations",
                [
                    ("a + b", "Add two tensors"),
                    ("a * b", "Element-wise multiplication"),
                    ("torch.matmul(a, b)", "Matrix multiplication"),
                    ("a @ b", "Matrix multiplication (@ operator)"),
                ],
            ),
            (
                "Neural Network Functions",
                [
                    ("F.relu(x)", "Apply ReLU activation"),
                    ("F.softmax(x, dim=1)", "Apply softmax"),
                ],
            ),
        ]

        for lesson_num, (topic, examples) in enumerate(lessons, 1):
            console.print(
                f"\n[bold yellow]📖 Lesson {lesson_num}: {topic}[/bold yellow]"
            )
            console.print("─" * 50)

            for example, description in examples:
                # Show the example
                example_panel = Panel(
                    f"[bold cyan]💡 {description}[/bold cyan]\n"
                    f"[bold white]Try this:[/bold white] [yellow]{example}[/yellow]",
                    title="Example",
                    border_style="green",
                )
                console.print(example_panel)

                # Show input tensors from the example
                example_vars = extract_variables_from_code(example, self.env)
                if example_vars:
                    input_info = "\n".join(
                        [show_tensor_info(self.env[var], var) for var in example_vars]
                    )
                    input_panel = Panel(
                        input_info,
                        title="📥 Available Input Tensors",
                        border_style="cyan",
                    )
                    console.print(input_panel)

                # Get user input
                while True:
                    user_code = Prompt.ask(
                        "[bold cyan]Type the code[/bold cyan]", default=example
                    )

                    # Show input tensors info
                    tensor_vars = extract_variables_from_code(user_code, self.env)
                    if tensor_vars:
                        input_info = "\n".join(
                            [
                                show_tensor_info(self.env[var], var)
                                for var in tensor_vars
                            ]
                        )
                        input_panel = Panel(
                            input_info, title="📥 Input Tensors", border_style="blue"
                        )
                        console.print(input_panel)

                    # Execute and show result
                    success, result, error = execute_code_safely(user_code, self.env)

                    if success:
                        if result is not None:
                            result_text = format_tensor_result(result)
                            result_panel = Panel(
                                result_text, title="✅ Result", border_style="green"
                            )
                            console.print(result_panel)
                        else:
                            console.print(
                                "[green]✅ Code executed successfully![/green]"
                            )
                        break
                    else:
                        error_panel = Panel(
                            f"[red]{error}[/red]", title="❌ Error", border_style="red"
                        )
                        console.print(error_panel)
                        if not Confirm.ask("Try again?"):
                            break

                if not Confirm.ask("Continue to next example?", default=True):
                    return

            console.print(f"[bold green]✅ Lesson {lesson_num} complete![/bold green]")
            if lesson_num < len(lessons) and not Confirm.ask(
                "Continue to next lesson?", default=True
            ):
                break

    def practice_mode(self):
        """Practice specific functions"""
        console.print(Panel("🎓 Practice Mode", style="bold blue"))
        console.print(
            "[dim]You can select functions by number or by typing their name (fuzzy matching supported)[/dim]\n"
        )

        # Show available variables
        self.show_available_variables()

        # Show function list
        func_table = Table(title="Available Functions", box=box.ROUNDED)
        func_table.add_column("#", style="cyan", no_wrap=True)
        func_table.add_column("Function", style="yellow", no_wrap=True)
        func_table.add_column("Purpose", style="white")
        func_table.add_column("Category", style="magenta")

        func_list = list(FUNCTION_CATALOG.items())
        for i, (name, info) in enumerate(func_list, 1):
            func_table.add_row(str(i), name, info["purpose"], info["category"])

        console.print(func_table)

        while True:
            choice = Prompt.ask("\nEnter function number or name (or 'q' to quit)")
            if choice.lower() == "q":
                break

            # Try to parse as number first
            try:
                func_idx = int(choice) - 1
                if 0 <= func_idx < len(func_list):
                    func_name, func_info = func_list[func_idx]
                    self.practice_function(func_name, func_info)
                else:
                    console.print(
                        "[red]Invalid number! Please choose a valid function number.[/red]"
                    )
            except ValueError:
                # Not a number, try to match by name
                func_names = [name for name, _ in func_list]

                # Check for exact match first
                if choice in func_names:
                    func_info = FUNCTION_CATALOG[choice]
                    self.practice_function(choice, func_info)
                else:
                    # Find similar functions
                    similar = find_similar_functions(choice, func_names, top_n=5)

                    if (
                        similar and similar[0][1] > 0.5
                    ):  # If best match has >50% similarity
                        best_match = similar[0][0]
                        best_score = similar[0][1]

                        # Auto-select if confidence is high (>= 80%) or significantly better than second choice
                        if best_score >= 0.8 or (
                            len(similar) > 1 and best_score - similar[1][1] > 0.3
                        ):
                            func_info = FUNCTION_CATALOG[best_match]
                            console.print(f"[dim]→ Selected: {best_match}[/dim]")
                            self.practice_function(best_match, func_info)
                        else:
                            # Only show suggestions for ambiguous matches
                            console.print(
                                "\n[yellow]Did you mean one of these?[/yellow]"
                            )
                            match_table = Table(box=box.SIMPLE)
                            match_table.add_column("#", style="cyan")
                            match_table.add_column("Function", style="yellow")
                            match_table.add_column("Match", style="green")

                            for i, (name, score) in enumerate(similar, 1):
                                if score > 0.3:  # Only show reasonably good matches
                                    match_table.add_row(
                                        str(i), name, f"{score*100:.0f}%"
                                    )

                            console.print(match_table)

                            # Ask user to select from matches
                            match_choice = Prompt.ask(
                                "Select a match (1-5) or press Enter to search again",
                                default="",
                            )

                            if match_choice.isdigit():
                                match_idx = int(match_choice) - 1
                                if 0 <= match_idx < len(similar):
                                    matched_name = similar[match_idx][0]
                                    func_info = FUNCTION_CATALOG[matched_name]
                                    self.practice_function(matched_name, func_info)
                    else:
                        console.print(
                            f"[red]No function found matching '{choice}'[/red]"
                        )
                        console.print(
                            "[dim]Try typing more of the function name or use a number from the list[/dim]"
                        )

    def practice_function(self, func_name, func_info):
        """Practice a specific function"""
        # Show function details
        details = f"[bold cyan]Function:[/bold cyan] {func_name}\n"
        details += f"[bold cyan]Purpose:[/bold cyan] {func_info['purpose']}\n"
        details += f"[bold cyan]Signature:[/bold cyan] {func_info['signature']}\n"
        details += f"[bold cyan]Arguments:[/bold cyan] {func_info['args']}"

        details_panel = Panel(
            details, title=f"📚 Learning: {func_name}", border_style="blue"
        )
        console.print(details_panel)

        # Show example
        example_panel = Panel(
            f"[yellow]{func_info['example']}[/yellow]",
            title="💡 Example to try",
            border_style="green",
        )
        console.print(example_panel)

        # Show input tensors from the default example
        default_vars = extract_variables_from_code(func_info["example"], self.env)
        if default_vars:
            input_info = "\n".join(
                [show_tensor_info(self.env[var], var) for var in default_vars]
            )
            input_panel = Panel(
                input_info, title="📥 Available Input Tensors", border_style="cyan"
            )
            console.print(input_panel)

        # Practice loop
        while True:
            user_code = Prompt.ask(
                "[bold cyan]Type the code[/bold cyan]", default=func_info["example"]
            )

            if user_code.lower() in ["q", "quit", "back"]:
                break

            # Show input tensors info
            tensor_vars = extract_variables_from_code(user_code, self.env)
            if tensor_vars:
                input_info = "\n".join(
                    [show_tensor_info(self.env[var], var) for var in tensor_vars]
                )
                input_panel = Panel(
                    input_info, title="📥 Input Tensors", border_style="blue"
                )
                console.print(input_panel)

            # Execute and show result
            success, result, error = execute_code_safely(user_code, self.env)

            if success:
                if result is not None:
                    result_text = format_tensor_result(result)
                    result_panel = Panel(
                        result_text, title="✅ Result", border_style="green"
                    )
                    console.print(result_panel)
                else:
                    console.print("[green]✅ Code executed successfully![/green]")
            else:
                error_panel = Panel(
                    f"[red]{error}[/red]", title="❌ Error", border_style="red"
                )
                console.print(error_panel)

            if not Confirm.ask("Try another variation?", default=True):
                break

    def category_practice_mode(self):
        """Practice by category"""
        console.print(Panel("🏃 Category Practice", style="bold blue"))

        # Show available variables
        self.show_available_variables()

        # Get categories
        categories = sorted(set(info["category"] for info in FUNCTION_CATALOG.values()))

        cat_table = Table(title="Categories", box=box.ROUNDED)
        cat_table.add_column("#", style="cyan", no_wrap=True)
        cat_table.add_column("Category", style="yellow")
        cat_table.add_column("Functions", style="white")

        for i, cat in enumerate(categories, 1):
            count = sum(
                1 for info in FUNCTION_CATALOG.values() if info["category"] == cat
            )
            cat_table.add_row(str(i), cat, str(count))

        console.print(cat_table)

        try:
            choice = Prompt.ask("Choose category number")
            cat_idx = int(choice) - 1

            if 0 <= cat_idx < len(categories):
                selected_cat = categories[cat_idx]
                self.practice_category(selected_cat)
            else:
                console.print("[red]Invalid choice![/red]")
        except ValueError:
            console.print("[red]Please enter a valid number![/red]")

    def practice_category(self, category):
        """Practice all functions in a category"""
        funcs_in_cat = [
            (name, info)
            for name, info in FUNCTION_CATALOG.items()
            if info["category"] == category
        ]

        console.print(f"\n[bold yellow]🏃 Practicing: {category}[/bold yellow]")
        console.print(f"This category has {len(funcs_in_cat)} functions")

        for i, (name, info) in enumerate(funcs_in_cat, 1):
            console.print(
                f"\n[bold cyan]📚 Exercise {i}/{len(funcs_in_cat)}: {name}[/bold cyan]"
            )

            # Show example
            example_panel = Panel(
                f"[bold white]Purpose:[/bold white] {info['purpose']}\n"
                f"[bold white]Try:[/bold white] [yellow]{info['example']}[/yellow]",
                title=f"Function: {name}",
                border_style="blue",
            )
            console.print(example_panel)

            # Show input tensors from the default example
            default_vars = extract_variables_from_code(info["example"], self.env)
            if default_vars:
                input_info = "\n".join(
                    [show_tensor_info(self.env[var], var) for var in default_vars]
                )
                input_panel = Panel(
                    input_info, title="📥 Available Input Tensors", border_style="cyan"
                )
                console.print(input_panel)

            # Get user input and execute
            user_code = Prompt.ask(
                "[bold cyan]Type the code[/bold cyan]", default=info["example"]
            )

            # Show input tensors info
            tensor_vars = extract_variables_from_code(user_code, self.env)
            if tensor_vars:
                input_info = "\n".join(
                    [show_tensor_info(self.env[var], var) for var in tensor_vars]
                )
                input_panel = Panel(
                    input_info, title="📥 Input Tensors", border_style="blue"
                )
                console.print(input_panel)

            success, result, error = execute_code_safely(user_code, self.env)

            if success:
                if result is not None:
                    result_text = format_tensor_result(result)
                    result_panel = Panel(
                        result_text, title="✅ Result", border_style="green"
                    )
                    console.print(result_panel)
                else:
                    console.print("[green]✅ Code executed successfully![/green]")
            else:
                error_panel = Panel(
                    f"[red]{error}[/red]", title="❌ Error", border_style="red"
                )
                console.print(error_panel)

            if i < len(funcs_in_cat) and not Confirm.ask(
                "Continue to next function?", default=True
            ):
                break

    def free_exploration_mode(self):
        """Free exploration mode"""
        console.print(Panel("🔍 Free Exploration Mode", style="bold blue"))
        console.print("Type any PyTorch code. Type 'quit' to exit.")

        # Show available variables with device info
        var_text = "Available tensors: " + ", ".join(self.samples.keys())
        var_text += f" (all on {DEVICE})"
        console.print(f"[dim]{var_text}[/dim]")
        console.print(f"[dim]Use to_device(tensor) to move tensors to {DEVICE}[/dim]")

        while True:
            user_code = Prompt.ask("[bold cyan]PyTorch>[/bold cyan]")

            if user_code.lower() in ["quit", "q", "exit"]:
                break

            # Show input tensors info
            tensor_vars = extract_variables_from_code(user_code, self.env)
            if tensor_vars:
                input_info = "\n".join(
                    [show_tensor_info(self.env[var], var) for var in tensor_vars]
                )
                input_panel = Panel(
                    input_info, title="📥 Input Tensors", border_style="blue"
                )
                console.print(input_panel)

            success, result, error = execute_code_safely(user_code, self.env)

            if success:
                if result is not None:
                    result_text = format_tensor_result(result)
                    result_panel = Panel(
                        result_text, title="Result", border_style="green"
                    )
                    console.print(result_panel)
                else:
                    console.print("[green]✅ Executed[/green]")
            else:
                error_panel = Panel(
                    f"[red]{error}[/red]", title="Error", border_style="red"
                )
                console.print(error_panel)

    def run(self):
        """Main TUI loop"""
        # Define menu options mapping
        menu_options = {
            "1": "tutorial",
            "2": "practice",
            "3": "category",
            "4": "free",
            "tutorial": "tutorial",
            "practice": "practice",
            "category": "category",
            "cat": "category",
            "free": "free",
            "explore": "free",
            "exploration": "free",
            "q": "quit",
            "quit": "quit",
            "exit": "quit",
        }

        while True:
            console.clear()
            self.show_welcome()

            choice = Prompt.ask("\nChoose an option").lower().strip()

            # Check for direct mapping
            if choice in menu_options:
                mode = menu_options[choice]
            else:
                # Try fuzzy matching for mode names
                mode_names = [
                    "tutorial",
                    "practice",
                    "category",
                    "free",
                    "explore",
                    "quit",
                ]
                similar = find_similar_functions(choice, mode_names, top_n=3)

                if similar and similar[0][1] > 0.6:  # If best match is > 60%
                    best_match = similar[0][0]
                    best_score = similar[0][1]

                    # Auto-select if confidence is high (>= 80%) or significantly better than second choice
                    if best_score >= 0.8 or (
                        len(similar) > 1 and best_score - similar[1][1] > 0.3
                    ):
                        mode = best_match
                        if mode == "explore":
                            mode = "free"
                        console.print(f"[dim]→ Selected: {best_match}[/dim]")
                    else:
                        # Only show suggestions for ambiguous matches
                        console.print("\n[yellow]Did you mean:[/yellow]")
                        for i, (name, score) in enumerate(similar[:3], 1):
                            if score > 0.3:
                                console.print(f"  {i}. {name} ({score*100:.0f}% match)")

                        match_choice = Prompt.ask(
                            "Select (1-3) or press Enter to try again", default=""
                        )

                        if match_choice.isdigit():
                            match_idx = int(match_choice) - 1
                            if 0 <= match_idx < len(similar):
                                mode = similar[match_idx][0]
                                if mode == "explore":
                                    mode = "free"
                            else:
                                console.print("[red]Invalid selection[/red]")
                                Prompt.ask("\nPress Enter to continue")
                                continue
                        else:
                            continue
                else:
                    console.print(f"[red]Invalid option: '{choice}'[/red]")
                    console.print(
                        "[dim]Try: 1-4, tutorial, practice, category, free, or quit[/dim]"
                    )
                    Prompt.ask("\nPress Enter to continue")
                    continue

            # Execute the selected mode
            if mode == "tutorial":
                self.tutorial_mode()
            elif mode == "practice":
                self.practice_mode()
            elif mode == "category":
                self.category_practice_mode()
            elif mode == "free":
                self.free_exploration_mode()
            elif mode == "quit":
                console.print(
                    "[bold green]Thanks for learning PyTorch! 🚀[/bold green]"
                )
                break

            if mode != "quit":
                Prompt.ask("\nPress Enter to return to main menu")


def main():
    """Main function"""
    try:
        # Check if rich is available
        from rich.console import Console

        tui = PyTorchTUI()
        tui.run()

    except ImportError:
        console.print("[red]Error: This TUI requires the 'rich' library.[/red]")
        console.print("[yellow]Install with: pip install rich[/yellow]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[bold green]Thanks for learning PyTorch! 🚀[/bold green]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        traceback.print_exc()


if __name__ == "__main__":
    main()
