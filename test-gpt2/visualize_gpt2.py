import matplotlib.pyplot as plt
import numpy as np


def visualize_gpt2_generation():
    """
    Visualize GPT-2 autoregressive generation process.
    Shows how input tokens map to output logits at each step.
    """

    # Data from out.txt - generation steps
    steps = [
        {
            "input": ["the", "dog", "sat"],
            "output_preds": [",", "'s", "on"],
            "logits": [-29.49, -70.89, -89.01],
            "next_token": "on",
            "top5": [
                ("on", 0.2151),
                ("in", 0.1425),
                ("down", 0.0824),
                ("there", 0.0510),
                ("up", 0.0483),
            ],
        },
        {
            "input": ["the", "dog", "sat", "on"],
            "output_preds": [",", "'s", "on", "the"],
            "logits": [-29.49, -70.89, -89.01, -55.68],
            "next_token": "the",
            "top5": [
                ("the", 0.4404),
                ("his", 0.1048),
                ("a", 0.0933),
                ("her", 0.0872),
                ("my", 0.0606),
            ],
        },
        {
            "input": ["the", "dog", "sat", "on", "the"],
            "output_preds": [",", "'s", "on", "the", "floor"],
            "logits": [-29.49, -70.89, -89.01, -55.68, -82.05],
            "next_token": "floor",
            "top5": [
                ("floor", 0.0715),
                ("ground", 0.0667),
                ("bed", 0.0512),
                ("couch", 0.0463),
                ("other", 0.0356),
            ],
        },
        {
            "input": ["the", "dog", "sat", "on", "the", "floor"],
            "output_preds": [",", "'s", "on", "the", "floor", ","],
            "logits": [-29.49, -70.89, -89.01, -55.68, -82.05, -86.91],
            "next_token": ",",
            "top5": [
                (",", 0.2088),
                ("and", 0.1015),
                (".", 0.0973),
                ("of", 0.0639),
                ("with", 0.0638),
            ],
        },
        {
            "input": ["the", "dog", "sat", "on", "the", "floor", ","],
            "output_preds": [",", "'s", "on", "the", "floor", ",", "and"],
            "logits": [-29.49, -70.89, -89.01, -55.68, -82.05, -86.91, -91.96],
            "next_token": "and",
            "top5": [
                ("and", 0.0906),
                ("his", 0.0430),
                ("the", 0.0364),
                ("her", 0.0218),
                ("but", 0.0216),
            ],
        },
    ]

    # Print each step details
    print("=" * 60)
    print("GPT-2 Generation Steps")
    print("=" * 60)
    for i, step in enumerate(steps):
        print(f"\nSTEP {i+1}")
        print("-" * 40)
        print(f"Input:      {step['input']}")
        print(f"Output:     {step['output_preds']}")
        print(f"Logits:     {step['logits']}")
        print(
            f"Next token: '{step['next_token']}' (from last logit: {step['logits'][-1]:.2f})"
        )
        print(f"Top-5:      {step['top5']}")
    print("\n" + "=" * 60)

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Input-Output Token Mapping
    ax1 = fig.add_subplot(2, 2, 1)
    step = steps[2]  # Step 3 as example
    n = len(step["input"])

    # Draw input tokens
    for i, tok in enumerate(step["input"]):
        ax1.text(
            i,
            1,
            tok,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightblue"),
        )
        ax1.text(i, 0.7, f"pos {i}", ha="center", va="center", fontsize=8, color="gray")

    # Draw output predictions
    for i, pred in enumerate(step["output_preds"]):
        ax1.text(
            i,
            0,
            pred,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgreen"),
        )
        # Arrow from input to output
        ax1.annotate(
            "",
            xy=(i, 0.15),
            xytext=(i, 0.85),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    # Highlight last position
    ax1.add_patch(
        plt.Rectangle(
            (n - 1.5, -0.3),
            1,
            1.6,
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
        )
    )
    ax1.text(
        n - 1,
        -0.5,
        "Used for\ngeneration",
        ha="center",
        va="top",
        fontsize=8,
        color="red",
    )

    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(-0.8, 1.3)
    ax1.set_title(
        "Step 3: Input Tokens → Output Logits\n(Same length: 5 in, 5 out)", fontsize=11
    )
    ax1.axis("off")

    # Plot 2: Top-5 Probabilities at each step
    ax2 = fig.add_subplot(2, 2, 2)

    x = np.arange(5)
    width = 0.15
    colors = plt.cm.viridis(np.linspace(0, 0.8, 5))

    for i, step in enumerate(steps):
        probs = [p[1] for p in step["top5"]]
        bars = ax2.bar(
            x + i * width, probs, width, label=f"Step {i+1}", color=colors[i]
        )

    ax2.set_xlabel("Top-5 Predictions")
    ax2.set_ylabel("Probability")
    ax2.set_title("Top-5 Token Probabilities at Each Step")
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(["1st", "2nd", "3rd", "4th", "5th"])
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: Sequence growth with input/output/logits
    ax3 = fig.add_subplot(2, 2, 3)

    full_sequence = ["the", "dog", "sat", "on", "the", "floor", ",", "and"]
    full_outputs = [",", "'s", "on", "the", "floor", ",", "and", "..."]
    full_logits = [-29.5, -70.9, -89.0, -55.7, -82.1, -86.9, -92.0, 0]

    for step_idx in range(5):
        y = 4 - step_idx
        seq_len = 3 + step_idx

        # Draw input tokens (top row)
        for i in range(seq_len):
            color = "lightblue" if i < 3 else "lightgreen"
            ax3.text(
                i,
                y + 0.15,
                full_sequence[i],
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

        # Draw output predictions (bottom row)
        for i in range(seq_len):
            ax3.text(
                i,
                y - 0.15,
                full_outputs[i],
                ha="center",
                va="center",
                fontsize=7,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )

        # Arrow to next token (from last output)
        if step_idx < 4:
            ax3.annotate(
                "",
                xy=(seq_len - 0.3, y - 0.5),
                xytext=(seq_len - 0.7, y - 0.8),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )

        ax3.text(-1.5, y, f"Step {step_idx+1}", ha="right", va="center", fontsize=9)

    # Legend
    ax3.text(
        7.5,
        4.3,
        "Input",
        ha="center",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    ax3.text(
        7.5,
        3.9,
        "Output",
        ha="center",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    ax3.set_xlim(-2, 8.5)
    ax3.set_ylim(-0.8, 4.8)
    ax3.set_title(
        "Autoregressive Generation\n(Input→Output→Logit for each position)", fontsize=11
    )
    ax3.axis("off")

    # Plot 4: Key insight diagram
    ax4 = fig.add_subplot(2, 2, 4)

    # Draw the key concept
    ax4.text(
        0.5,
        0.9,
        "GPT-2 Autoregressive Generation",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    explanations = [
        "1. Input: [the, dog, sat] (3 tokens)",
        "2. Output: [logits₀, logits₁, logits₂] (3 logits)",
        "3. Only logits₂ (last) predicts next token",
        '4. Append "on" → new input: [the, dog, sat, on]',
        "5. Repeat until done",
        "",
        "Key: Input length = Output length",
        "Previous positions recomputed (KV-cache helps)",
    ]

    for i, text in enumerate(explanations):
        weight = "bold" if "Key:" in text else "normal"
        color = "red" if "Key:" in text else "black"
        ax4.text(
            0.1,
            0.75 - i * 0.09,
            text,
            ha="left",
            va="center",
            fontsize=9,
            fontweight=weight,
            color=color,
        )

    ax4.axis("off")

    plt.tight_layout()
    plt.savefig("out.jpg", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: out.jpg")


if __name__ == "__main__":
    visualize_gpt2_generation()
