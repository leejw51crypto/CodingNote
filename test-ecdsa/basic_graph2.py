"""
MIT License

Copyright (c) 2024 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

⚠️ EDUCATIONAL VISUALIZATION WARNING ⚠️
=====================================
This is an interactive visualization tool for educational purposes only:
1. Uses floating-point arithmetic (not suitable for cryptographic operations)
2. Implements simplified point addition without proper finite field arithmetic
3. Does not handle edge cases properly
4. Not suitable for any cryptographic purposes
5. For visualization and learning only

For actual cryptographic implementations:
- Use established cryptographic libraries
- Implement proper finite field arithmetic
- Use constant-time operations
- Follow cryptographic best practices
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


def point_add(P, Q, a=0, p=None):
    """Add two points on the curve y^2 = x^3 + ax + b"""
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2 and y1 == -y2:
        return None  # Point at infinity

    if P != Q:
        # Point addition
        m = (y2 - y1) / (x2 - x1)
    else:
        # Point doubling
        m = (3 * x1 * x1 + a) / (2 * y1)

    x3 = m * m - x1 - x2
    y3 = m * (x1 - x3) - y1

    return (x3, y3)


def scalar_multiply(k, P, a=0, p=None):
    """Multiply point P by scalar k using double-and-add algorithm"""
    result = None
    addend = P

    while k:
        if k & 1:
            result = point_add(result, addend, a, p)
        addend = point_add(addend, addend, a, p)
        k >>= 1

    return result


class CurveAnimator:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.points = []
        self.lines = []

        # Store clicked points and their sum
        self.clicked_points = []
        self.sum_point = None

        # Connect mouse click event
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # Set zoom limits
        self.x_min, self.x_max = -3, 3
        self.y_min, self.y_max = -4, 4

        # Setup the initial plot
        self.setup_plot()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Find closest point on curve to click
        x_click = event.xdata
        y_click = event.ydata

        # Calculate y values for this x on curve
        y_pos = np.sqrt(x_click**3 + 7)
        y_neg = -np.sqrt(x_click**3 + 7)

        # Choose closest y-value
        if abs(y_click - y_pos) < abs(y_click - y_neg):
            y_curve = y_pos
        else:
            y_curve = y_neg

        new_point = (x_click, y_curve)

        # Store points with chain behavior
        if len(self.clicked_points) == 0:
            # First point (P)
            self.clicked_points.append(new_point)
        elif len(self.clicked_points) == 1:
            # Second point (Q)
            self.clicked_points.append(new_point)
            self.sum_point = point_add(self.clicked_points[0], self.clicked_points[1])
        else:
            # Move Q to P, new point becomes Q
            self.clicked_points = [self.clicked_points[1], new_point]
            self.sum_point = point_add(self.clicked_points[0], self.clicked_points[1])

        # Clear previous points and lines
        self.clear_plot()
        # Redraw everything
        self.draw_current_state()

    def clear_plot(self):
        # Remove all scatter plots and lines except the main curve
        for collection in self.ax.collections[:]:
            collection.remove()
        for line in self.ax.lines[2:]:  # Keep first two lines (the curve)
            line.remove()

    def draw_current_state(self):
        # Plot clicked points with different colors
        if len(self.clicked_points) >= 1:
            self.ax.scatter(
                [self.clicked_points[0][0]],
                [self.clicked_points[0][1]],
                c="blue",
                s=100,
                label="P",
            )
        if len(self.clicked_points) >= 2:
            self.ax.scatter(
                [self.clicked_points[1][0]],
                [self.clicked_points[1][1]],
                c="red",
                s=100,
                label="Q",
            )

        # Plot sum point if we have two points
        if self.sum_point is not None:
            self.ax.scatter(
                [self.sum_point[0]],
                [self.sum_point[1]],
                c="green",
                s=100,
                label="P + Q",
            )
            self.draw_addition_lines()

        self.ax.legend()
        plt.draw()

    def draw_addition_lines(self):
        if len(self.clicked_points) == 2 and self.sum_point is not None:
            P1, P2 = self.clicked_points

            # If points are different, draw the secant line
            if P1 != P2:
                # Check if x-coordinates are too close
                x_diff = abs(P2[0] - P1[0])
                if x_diff < 1e-10:  # Nearly vertical line
                    # Draw a vertical line at the x-coordinate
                    x_coord = (P1[0] + P2[0]) / 2  # Average x coordinate
                    self.ax.plot(
                        [x_coord, x_coord],
                        [self.y_min, self.y_max],
                        "purple",
                        linestyle="--",
                        alpha=0.5,
                        label="Secant line",
                    )
                else:
                    # Normal case - draw secant line
                    m = (P2[1] - P1[1]) / (P2[0] - P1[0])
                    x_extended = np.array(
                        [
                            max(self.x_min, min(P1[0], P2[0]) - 1),
                            min(self.x_max, max(P1[0], P2[0]) + 1),
                        ]
                    )
                    y_extended = m * (x_extended - P1[0]) + P1[1]
                    # Clip y values to prevent extreme values
                    y_extended = np.clip(y_extended, self.y_min, self.y_max)
                    self.ax.plot(
                        x_extended,
                        y_extended,
                        "purple",
                        linestyle="--",
                        alpha=0.5,
                        label="Secant line",
                    )
            else:
                # Draw tangent line for point doubling
                m = (3 * P1[0] * P1[0]) / (2 * P1[1])
                x_extended = np.array(
                    [max(self.x_min, P1[0] - 1), min(self.x_max, P1[0] + 1)]
                )
                y_extended = m * (x_extended - P1[0]) + P1[1]
                # Clip y values to prevent extreme values
                y_extended = np.clip(y_extended, self.y_min, self.y_max)
                self.ax.plot(
                    x_extended,
                    y_extended,
                    "purple",
                    linestyle="--",
                    alpha=0.5,
                    label="Tangent line",
                )

            # Draw vertical line to sum point
            y_min_vert = max(self.y_min, -self.sum_point[1])
            y_max_vert = min(self.y_max, self.sum_point[1])
            self.ax.plot(
                [self.sum_point[0], self.sum_point[0]],
                [y_min_vert, y_max_vert],
                "green",
                linestyle="--",
                alpha=0.5,
                label="Vertical line",
            )

    def setup_plot(self):
        # Plot the curve
        x = np.linspace(self.x_min, self.x_max, 1000)
        y_pos = np.sqrt(x**3 + 7)
        y_neg = -np.sqrt(x**3 + 7)

        self.ax.plot(x, y_pos, "b-", alpha=0.3, label="y² = x³ + 7")
        self.ax.plot(x, y_neg, "b-", alpha=0.3)

        # Setup plot properties
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Interactive Point Addition\nClick to add points!")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect("equal")

        # Set axis limits
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)

        # Add explanation
        explanation = """
        Interactive Point Addition:
        • Click anywhere near the curve to add points
        • First click: Places P (blue)
        • Second click: Places Q (red)
        • Green point shows P + Q
        • Purple line: Secant/Tangent line
        • Green line: Vertical line to result
        • Click again to start over
        """
        self.ax.text(
            3.2, 0, explanation, fontsize=8, bbox=dict(facecolor="white", alpha=0.8)
        )

        self.ax.legend()

    def create_animation(self):
        plt.show()


if __name__ == "__main__":
    animator = CurveAnimator()
    animator.create_animation()
