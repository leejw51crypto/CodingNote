#!/usr/bin/env python3
"""
PyTorch Cheatsheet Generator
Generates a professional A3 PDF with all essential PyTorch functions
"""

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

# Ask for font preference
font_input = input("Enter font name (or press Enter for default): ").strip()
default_fonts = ["DejaVu Sans", "Arial", "sans-serif"]


# Check if font is available
def is_font_available(font_name):
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    return font_name in available_fonts


# Set style
plt.style.use("default")
if font_input:
    if is_font_available(font_input):
        plt.rcParams["font.family"] = [font_input] + default_fonts
        print(f"✅ Using font: {font_input}")
    else:
        print(f"⚠️  Font '{font_input}' not found. Using default fonts.")
        plt.rcParams["font.family"] = default_fonts
else:
    plt.rcParams["font.family"] = default_fonts

plt.rcParams["font.size"] = 8

# A3 size in inches (landscape) with better margins
fig = plt.figure(figsize=(16.5, 11.7), facecolor="white")
ax = fig.add_subplot(111)
ax.axis("off")

# Professional header with gradient-like effect
header_rect = Rectangle(
    (0, 0.94), 1, 0.06, transform=ax.transAxes, facecolor="#2E3440", alpha=0.9
)
ax.add_patch(header_rect)

# Title with better typography
ax.text(
    0.02,
    0.97,
    "★",
    fontsize=24,
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="orange",
)
ax.text(
    0.06,
    0.97,
    "PyTorch",
    fontsize=22,
    weight="bold",
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="white",
)
ax.text(
    0.18,
    0.97,
    "Complete Function Reference",
    fontsize=16,
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="#D8DEE9",
)

# Version info
ax.text(
    0.98,
    0.97,
    "v2.x",
    fontsize=12,
    weight="bold",
    ha="right",
    va="center",
    transform=ax.transAxes,
    color="#88C0D0",
)

# Categories and functions
categories = {
    "1. Tensor Creation": [
        "torch.tensor(data)",
        "torch.zeros(*size)",
        "torch.ones(*size)",
        "torch.eye(n)",
        "torch.arange(start, end, step)",
        "torch.linspace(start, end, steps)",
        "torch.logspace(start, end, steps)",
        "torch.rand(*size)  # uniform [0,1)",
        "torch.randn(*size)  # normal N(0,1)",
        "torch.randint(low, high, size)",
        "torch.empty(*size)",
        "torch.full(size, fill_value)",
    ],
    "2. Basic Operations": [
        "torch.add(a, b) or a + b",
        "torch.sub(a, b) or a - b",
        "torch.mul(a, b) or a * b",
        "torch.div(a, b) or a / b",
        "torch.matmul(a, b) or a @ b",
        "torch.pow(a, exp) or a ** exp",
        "torch.abs(x)",
        "torch.neg(x)",
        "torch.reciprocal(x)",
    ],
    "3. Shape Manipulation": [
        "x.reshape(*shape)",
        "x.view(*shape)",
        "x.transpose(dim0, dim1)",
        "x.permute(*dims)",
        "x.squeeze(dim)",
        "x.unsqueeze(dim)",
        "x.flatten(start_dim, end_dim)",
        "x.expand(*sizes)",
        "x.repeat(*sizes)",
        "x.contiguous()",
    ],
    "4. Indexing & Slicing": [
        "x[i]  # index",
        "x[i:j]  # slice",
        "x[..., i]  # ellipsis",
        "x[:, -1]  # last column",
        "torch.index_select(x, dim, idx)",
        "torch.masked_select(x, mask)",
        "torch.gather(x, dim, idx)",
        "torch.scatter(x, dim, idx, src)",
        "torch.where(cond, x, y)",
    ],
    "5. Reduction Ops": [
        "x.sum(dim, keepdim)",
        "x.mean(dim, keepdim)",
        "x.std(dim, unbiased)",
        "x.var(dim, unbiased)",
        "x.max(dim)",
        "x.min(dim)",
        "x.argmax(dim)",
        "x.argmin(dim)",
        "x.median(dim)",
        "x.mode(dim)",
        "x.prod(dim)",
        "x.cumsum(dim)",
    ],
    "6. Math Functions": [
        "torch.sin(x), cos(x), tan(x)",
        "torch.asin(x), acos(x), atan(x)",
        "torch.sinh(x), cosh(x), tanh(x)",
        "torch.exp(x), log(x), log10(x)",
        "torch.sqrt(x), rsqrt(x)",
        "torch.floor(x), ceil(x), round(x)",
        "torch.clamp(x, min, max)",
        "torch.sign(x)",
        "torch.sigmoid(x)",
    ],
    "7. Linear Algebra": [
        "torch.mm(a, b)  # 2D only",
        "torch.bmm(a, b)  # batch",
        "torch.mv(mat, vec)",
        "torch.dot(a, b)  # 1D only",
        "torch.det(x)",
        "torch.inverse(x)",
        "torch.svd(x)",
        "torch.eig(x)",
        "torch.linalg.norm(x, ord)",
        "torch.linalg.solve(A, b)",
        "torch.trace(x)",
    ],
    "8. NN Functions (F)": [
        "F.relu(x)",
        "F.leaky_relu(x, neg_slope)",
        "F.gelu(x)",
        "F.sigmoid(x)",
        "F.tanh(x)",
        "F.softmax(x, dim)",
        "F.log_softmax(x, dim)",
        "F.dropout(x, p, training)",
        "F.batch_norm(x, ...)",
        "F.layer_norm(x, shape)",
    ],
    "9. Loss Functions": [
        "F.mse_loss(pred, target)",
        "F.l1_loss(pred, target)",
        "F.cross_entropy(logits, labels)",
        "F.nll_loss(log_probs, labels)",
        "F.binary_cross_entropy(pred, target)",
        "F.kl_div(log_pred, target)",
        "F.cosine_similarity(x1, x2, dim)",
        "F.triplet_margin_loss(...)",
    ],
    "10. Pooling & Conv": [
        "F.max_pool2d(x, kernel_size)",
        "F.avg_pool2d(x, kernel_size)",
        "F.adaptive_max_pool2d(x, output_size)",
        "F.conv2d(x, weight, bias)",
        "F.conv_transpose2d(x, weight)",
        "F.interpolate(x, size, mode)",
        "F.pad(x, pad, mode)",
    ],
    "11. Advanced Ops": [
        "torch.einsum('ij,jk->ik', a, b)",
        "torch.topk(x, k, dim)",
        "torch.sort(x, dim)",
        "torch.argsort(x, dim)",
        "torch.unique(x)",
        "torch.cat([t1, t2], dim)",
        "torch.stack([t1, t2], dim)",
        "torch.split(x, size, dim)",
        "torch.chunk(x, chunks, dim)",
        "torch.broadcast_to(x, shape)",
    ],
    "12. Autograd": [
        "x.requires_grad_(True)",
        "y.backward()",
        "x.grad  # gradient",
        "x.detach()",
        "x.clone()",
        "with torch.no_grad():",
        "torch.autograd.grad(y, x)",
        "optimizer.zero_grad()",
        "optimizer.step()",
        "torch.nn.utils.clip_grad_norm_(params, max)",
    ],
    "13. Device Ops": [
        "torch.cuda.is_available()",
        "torch.device('cuda'/'cpu'/'mps')",
        "x.to(device)",
        "x.cuda(), x.cpu()",
        "x.device  # check device",
        "torch.cuda.empty_cache()",
    ],
    "14. Utilities": [
        "x.dtype, x.shape, x.size()",
        "x.numel()  # number of elements",
        "x.dim()  # number of dimensions",
        "x.is_contiguous()",
        "x.float(), x.int(), x.long()",
        "torch.from_numpy(arr)",
        "x.numpy()  # CPU only",
        "torch.save(obj, path)",
        "torch.load(path)",
        "torch.manual_seed(seed)",
    ],
    "15. Comparison": [
        "torch.eq(a, b) or a == b",
        "torch.ne(a, b) or a != b",
        "torch.gt(a, b) or a > b",
        "torch.lt(a, b) or a < b",
        "torch.ge(a, b) or a >= b",
        "torch.le(a, b) or a <= b",
        "torch.allclose(a, b, rtol, atol)",
        "torch.isnan(x)",
        "torch.isinf(x)",
        "torch.isfinite(x)",
    ],
}

# Layout parameters - optimized for better spacing
num_cols = 5
num_rows = 3
col_width = 1.0 / num_cols
row_height = 0.88 / num_rows

# Professional color scheme (inspired by GitHub/Nord themes)
colors = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#FFA07A",
    "#98D8C8",  # Row 1: warm/cool
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E9",
    "#F8C471",
    "#82E0AA",  # Row 2: soft pastels
    "#AED6F1",
    "#F9E79F",
    "#D7BDE2",
    "#A9DFBF",
    "#F5B7B1",  # Row 3: light tones
]

# Category icons for better visual appeal (using simple symbols)
icons = [
    "■",  # Tensor Creation
    "⚙",  # Basic Operations
    "↻",  # Shape Manipulation
    "◉",  # Indexing & Slicing
    "Σ",  # Reduction Ops
    "∫",  # Math Functions
    "≡",  # Linear Algebra
    "◈",  # NN Functions
    "×",  # Loss Functions
    "▣",  # Pooling & Conv
    "∞",  # Advanced Ops
    "∂",  # Autograd
    "◎",  # Device Ops
    "※",  # Utilities
    "≈",  # Comparison
]

# Draw categories with improved design
cat_list = list(categories.items())
for idx, (cat_name, functions) in enumerate(cat_list):
    row = idx // num_cols
    col = idx % num_cols

    x = col * col_width
    y = 0.92 - row * row_height

    # Main category box with subtle shadow
    shadow_box = FancyBboxPatch(
        (x + 0.007, y - row_height + 0.007),
        col_width - 0.014,
        row_height - 0.014,
        boxstyle="round,pad=0.005",
        facecolor="#000000",
        alpha=0.1,
        transform=ax.transAxes,
    )
    ax.add_patch(shadow_box)

    # Main category box
    bbox = FancyBboxPatch(
        (x + 0.005, y - row_height + 0.01),
        col_width - 0.01,
        row_height - 0.015,
        boxstyle="round,pad=0.005",
        facecolor=colors[idx % len(colors)],
        edgecolor="#34495e",
        linewidth=1.2,
        alpha=0.9,
        transform=ax.transAxes,
    )
    ax.add_patch(bbox)

    # Category header with icon
    header_y = y - 0.006
    ax.text(
        x + 0.012,
        header_y,
        icons[idx] if idx < len(icons) else "📋",
        fontsize=11,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Category title with better formatting
    clean_title = cat_name.replace(f"{idx+1}. ", "").upper()
    ax.text(
        x + 0.035,
        header_y,
        clean_title,
        fontsize=8.5,
        weight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
        color="#2C3E50",
    )

    # Separator line
    ax.plot(
        [x + 0.01, x + col_width - 0.01],
        [header_y - 0.018, header_y - 0.018],
        color="#BDC3C7",
        linewidth=0.8,
        transform=ax.transAxes,
    )

    # Functions with better formatting
    func_text = ""
    for i, func in enumerate(functions[:13]):  # Limit to 13 functions
        # Highlight function names
        if "(" in func:
            func_name = func.split("(")[0]
            func_rest = "(" + func.split("(", 1)[1] if "(" in func else ""
            func_text += f"• {func_name}{func_rest}\n"
        else:
            func_text += f"• {func}\n"

    if len(functions) > 13:
        func_text += f"• ... +{len(functions)-13} more\n"

    ax.text(
        x + 0.012,
        header_y - 0.025,
        func_text.rstrip(),
        fontsize=6.8,
        ha="left",
        va="top",
        family="monospace",
        transform=ax.transAxes,
        linespacing=1.35,
        color="#2C3E50",
    )

# Professional footer with additional info
footer_rect = Rectangle(
    (0, 0),
    1,
    0.04,
    transform=ax.transAxes,
    facecolor="#F8F9FA",
    alpha=0.8,
    edgecolor="#DEE2E6",
    linewidth=1,
)
ax.add_patch(footer_rect)

# Footer content with icons and better formatting
ax.text(
    0.02,
    0.02,
    "◐",
    fontsize=14,
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="#495057",
)
ax.text(
    0.04,
    0.02,
    "Import:",
    fontsize=8,
    weight="bold",
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="#495057",
)
ax.text(
    0.08,
    0.02,
    "import torch",
    fontsize=7,
    ha="left",
    va="center",
    transform=ax.transAxes,
    family="monospace",
    color="#6C757D",
)
ax.text(
    0.16,
    0.02,
    "import torch.nn.functional as F",
    fontsize=7,
    ha="left",
    va="center",
    transform=ax.transAxes,
    family="monospace",
    color="#6C757D",
)

ax.text(
    0.35,
    0.02,
    "⚡",
    fontsize=14,
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="#495057",
)
ax.text(
    0.37,
    0.02,
    "Device:",
    fontsize=8,
    weight="bold",
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="#495057",
)
ax.text(
    0.42,
    0.02,
    "cuda | cpu | mps",
    fontsize=7,
    ha="left",
    va="center",
    transform=ax.transAxes,
    family="monospace",
    color="#6C757D",
)

ax.text(
    0.55,
    0.02,
    "◆",
    fontsize=14,
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="#495057",
)
ax.text(
    0.57,
    0.02,
    "Version:",
    fontsize=8,
    weight="bold",
    ha="left",
    va="center",
    transform=ax.transAxes,
    color="#495057",
)
ax.text(
    0.62,
    0.02,
    "PyTorch 2.x",
    fontsize=7,
    ha="left",
    va="center",
    transform=ax.transAxes,
    family="monospace",
    color="#6C757D",
)

# Quick reference
ax.text(
    0.98,
    0.02,
    "◇ Quick: x.shape | x.dtype | x.device | x.grad",
    fontsize=7,
    ha="right",
    va="center",
    transform=ax.transAxes,
    family="monospace",
    style="italic",
    color="#6C757D",
)

# Save with high quality settings
plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save as PDF with optimized settings
plt.savefig(
    "pytorch_cheatsheet.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
    pad_inches=0.1,
)
print("✓ Professional PyTorch cheatsheet saved as 'pytorch_cheatsheet.pdf'")

# Save as high-quality PNG
plt.savefig(
    "pytorch_cheatsheet.png",
    format="png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
    pad_inches=0.1,
)
print("✓ Also saved as 'pytorch_cheatsheet.png' (200 DPI)")

# Save as SVG for vector graphics
plt.savefig(
    "pytorch_cheatsheet.svg",
    format="svg",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
    pad_inches=0.1,
)
print("✓ Vector version saved as 'pytorch_cheatsheet.svg'")

plt.show()
