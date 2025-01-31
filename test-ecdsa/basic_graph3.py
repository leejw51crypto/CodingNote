import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec


def plot_curve():
    """Plot a section of the secp256k1 curve"""
    x = np.linspace(-10, 10, 1000)  # Wider initial range
    y_pos = np.sqrt(x**3 + 7)
    y_neg = -np.sqrt(x**3 + 7)
    return x, y_pos, y_neg


def point_add(P1, P2):
    """Add two points on the curve (for visualization only)"""
    if P1 is None:
        return P2
    if P2 is None:
        return P1

    x1, y1 = P1
    x2, y2 = P2

    if x1 == x2 and y1 == y2:
        # Point doubling
        if y1 == 0:
            return None
        # s = (3x₁² + a) / (2y₁)  where a=0 for secp256k1
        s = (3 * x1 * x1) / (2 * y1)
    else:
        # Point addition
        if x1 == x2:
            return None
        # s = (y₂ - y₁) / (x₂ - x₁)
        s = (y2 - y1) / (x2 - x1)

    # x₃ = s² - x₁ - x₂
    x3 = s * s - x1 - x2
    # y₃ = s(x₁ - x₃) - y₁
    y3 = s * (x1 - x3) - y1

    return (x3, y3)


def calculate_double_and_add(k, G):
    """Calculate points using double-and-add method"""
    points = [(G, "G")]
    current = G

    # Convert k to binary
    binary = bin(k)[2:]  # Remove '0b' prefix
    print(f"Binary representation: {binary}")

    # Calculate and store doubling points (2G, 4G, 8G, etc.)
    doubles = [G]
    double_labels = ["G"]
    for i in range(len(binary) - 1):
        doubled = point_add(doubles[-1], doubles[-1])
        doubles.append(doubled)
        double_labels.append(f"2^{i+1}G ({2**(i+1)}G)")
        points.append((doubled, double_labels[-1]))

    # Calculate final result using necessary additions
    result = None
    for i, bit in enumerate(binary):
        if bit == "1":
            power = len(binary) - 1 - i
            if result is None:
                result = doubles[power]
            else:
                result = point_add(result, doubles[power])
                points.append((result, f"Running sum + {2**power}G"))

    return points


def get_plot_bounds(points, margin=0.2):
    """Calculate optimal plot bounds to show all points with margin"""
    x_coords = [p[0] for p, _ in points]
    y_coords = [p[1] for p, _ in points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add margin
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin

    # Ensure aspect ratio is reasonable
    total_range = max(x_max - x_min, y_max - y_min)
    if (x_max - x_min) < total_range:
        center = (x_max + x_min) / 2
        x_min = center - total_range / 2
        x_max = center + total_range / 2
    if (y_max - y_min) < total_range:
        center = (y_max + y_min) / 2
        y_min = center - total_range / 2
        y_max = center + total_range / 2

    return x_min, x_max, y_min, y_max


def calculate_line(P1, P2):
    """Calculate line parameters (slope and y-intercept) between two points"""
    x1, y1 = P1
    x2, y2 = P2

    # For point doubling, calculate tangent line
    if x1 == x2 and y1 == y2:
        # Tangent line slope: s = (3x₁² + a) / (2y₁) where a=0 for our curve
        s = (3 * x1 * x1) / (2 * y1)
    else:
        # Regular point addition: slope between two points
        s = (y2 - y1) / (x2 - x1)

    # y = sx + b => b = y - sx
    b = y1 - s * x1
    return s, b


def calculate_intersection(s, b, P1, P2):
    """Calculate intersection point of line with curve y² = x³ + 7"""
    # Line equation: y = sx + b
    # Curve equation: y² = x³ + 7
    # Substituting: (sx + b)² = x³ + 7
    # Rearranging: x³ - s²x² - 2sbx - (b² - 7) = 0
    coeffs = [1, -s * s, -2 * s * b, -(b * b - 7)]
    roots = np.roots(coeffs)

    # Get all real roots
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10:
            x = float(root.real)
            y = s * x + b
            real_roots.append((x, y))

    # For point doubling
    if P1[0] == P2[0] and P1[1] == P2[1]:
        # Return the point that's not P1
        for x, y in real_roots:
            if abs(x - P1[0]) > 1e-10 or abs(y - P1[1]) > 1e-10:
                return x, y
    else:
        # For point addition, return the point that's neither P1 nor P2
        for x, y in real_roots:
            if (abs(x - P1[0]) > 1e-10 or abs(y - P1[1]) > 1e-10) and (
                abs(x - P2[0]) > 1e-10 or abs(y - P2[1]) > 1e-10
            ):
                return x, y

    # Fallback (shouldn't happen)
    return real_roots[-1]


class ECDSAVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 16))

        # Create grid for main plot and buttons
        gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1])
        self.ax = plt.subplot(gs[0])
        self.button_ax = plt.subplot(gs[1])

        # Initialize zoom level and center
        self.zoom_level = 1.0
        self.center = [0, 0]

        # Initialize pan variables
        self.is_panning = False
        self.pan_start = None

        # Calculate initial points
        self.setup_points()

        # Plot initial state
        self.plot_current_view()

        # Add navigation toolbar
        self.toolbar = plt.get_current_fig_manager().toolbar

        # Add custom buttons
        self.setup_buttons()

        # Connect mouse events for panning
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # Enable pan/zoom
        self.ax.set_title(
            "Use mouse drag to pan, or use arrow buttons\nUse zoom buttons or mouse wheel to zoom",
            pad=20,
        )

    def setup_points(self):
        # Use smaller generator point coordinates for visualization
        self.Gx = 1
        self.Gy = np.sqrt(self.Gx**3 + 7)
        self.G = (self.Gx, self.Gy)

        # Use a private key that shows the process clearly
        self.private_key = 5
        print(f"Private key: {self.private_key}")

        # Calculate points using double-and-add
        self.all_points = calculate_double_and_add(self.private_key, self.G)

        # Get initial bounds
        self.x_min, self.x_max, self.y_min, self.y_max = get_plot_bounds(
            self.all_points
        )

    def setup_buttons(self):
        # Create button axes
        button_width = 0.1
        button_spacing = 0.02
        button_height = 0.04

        # Zoom buttons
        zoom_in_ax = plt.axes([0.2, 0.02, button_width, button_height])
        zoom_out_ax = plt.axes(
            [0.2 + button_width + button_spacing, 0.02, button_width, button_height]
        )
        reset_ax = plt.axes(
            [
                0.2 + 2 * (button_width + button_spacing),
                0.02,
                button_width,
                button_height,
            ]
        )

        # Pan buttons
        pan_left_ax = plt.axes([0.5, 0.02, button_width / 2, button_height])
        pan_right_ax = plt.axes(
            [
                0.5 + button_width / 2 + button_spacing,
                0.02,
                button_width / 2,
                button_height,
            ]
        )
        pan_up_ax = plt.axes(
            [
                0.5 + button_width / 4,
                0.02 + button_height + button_spacing,
                button_width / 2,
                button_height,
            ]
        )
        pan_down_ax = plt.axes(
            [
                0.5 + button_width / 4,
                0.02 - button_height - button_spacing,
                button_width / 2,
                button_height,
            ]
        )

        # Create buttons
        self.zoom_in_button = Button(zoom_in_ax, "Zoom In")
        self.zoom_out_button = Button(zoom_out_ax, "Zoom Out")
        self.reset_button = Button(reset_ax, "Reset View")

        self.pan_left_button = Button(pan_left_ax, "←")
        self.pan_right_button = Button(pan_right_ax, "→")
        self.pan_up_button = Button(pan_up_ax, "↑")
        self.pan_down_button = Button(pan_down_ax, "↓")

        # Connect button events
        self.zoom_in_button.on_clicked(self.zoom_in)
        self.zoom_out_button.on_clicked(self.zoom_out)
        self.reset_button.on_clicked(self.reset_view)

        self.pan_left_button.on_clicked(lambda x: self.pan_step("left"))
        self.pan_right_button.on_clicked(lambda x: self.pan_step("right"))
        self.pan_up_button.on_clicked(lambda x: self.pan_step("up"))
        self.pan_down_button.on_clicked(lambda x: self.pan_step("down"))

        # Add mouse wheel zoom
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            if event.button == "up":
                self.zoom_in(event)
            else:
                self.zoom_out(event)

    def on_mouse_press(self, event):
        if event.inaxes == self.ax and event.button == 1:  # Left click
            self.is_panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        self.is_panning = False
        self.pan_start = None

    def on_mouse_move(self, event):
        if self.is_panning and event.inaxes == self.ax and self.pan_start:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            self.center[0] -= dx
            self.center[1] -= dy
            self.update_view()
            self.pan_start = (event.xdata, event.ydata)

    def pan_step(self, direction):
        step = (self.x_max - self.x_min) * 0.1
        if direction == "left":
            self.center[0] += step
        elif direction == "right":
            self.center[0] -= step
        elif direction == "up":
            self.center[1] -= step
        elif direction == "down":
            self.center[1] += step
        self.update_view()

    def zoom_in(self, event):
        self.zoom_level *= 0.8
        self.update_view()

    def zoom_out(self, event):
        self.zoom_level *= 1.2
        self.update_view()

    def reset_view(self, event):
        self.zoom_level = 1.0
        self.center = [0, 0]
        self.x_min, self.x_max, self.y_min, self.y_max = get_plot_bounds(
            self.all_points
        )
        self.update_view()

    def update_view(self):
        self.ax.clear()
        self.plot_current_view()
        plt.draw()

    def plot_current_view(self):
        # Calculate current view bounds
        center_x = (self.x_max + self.x_min) / 2 + self.center[0]
        center_y = (self.y_max + self.y_min) / 2 + self.center[1]
        width = (self.x_max - self.x_min) * self.zoom_level
        height = (self.y_max - self.y_min) * self.zoom_level

        current_x_min = center_x - width / 2
        current_x_max = center_x + width / 2
        current_y_min = center_y - height / 2
        current_y_max = center_y + height / 2

        # Plot curve
        x = np.linspace(current_x_min, current_x_max, 1000)
        y_pos = np.sqrt(x**3 + 7)
        y_neg = -np.sqrt(x**3 + 7)
        self.ax.plot(x, y_pos, "b-", alpha=0.3, label="y² = x³ + 7")
        self.ax.plot(x, y_neg, "b-", alpha=0.3)

        # Plot points and lines
        prev_point = None
        for i, (point, label) in enumerate(self.all_points):
            # Different colors for special points
            if i == 0:  # G
                color = "green"
                size = 200
            elif i == len(self.all_points) - 1:  # Final public key
                color = "red"
                size = 200
            elif "Running sum" in label:  # Addition steps
                color = "purple"
                size = 150
            else:  # Doubling steps
                color = "blue"
                size = 150

            # Plot point
            self.ax.scatter([point[0]], [point[1]], c=color, s=size, alpha=0.6)

            # Add label
            self.ax.annotate(
                label,
                (point[0], point[1]),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                fontsize=12,
            )

            # Draw lines and intersection points for point addition/doubling
            if prev_point is not None:
                # Determine if this is a doubling operation
                is_doubling = False
                if "2G" in label or "4G" in label or "8G" in label:
                    # This point was obtained by doubling the previous point
                    is_doubling = True

                # For doubling, use the same point twice
                if is_doubling:
                    P1 = P2 = prev_point
                else:
                    P1, P2 = prev_point, point

                # Calculate line parameters
                s, b = calculate_line(P1, P2)

                # Plot line
                x_line = np.linspace(current_x_min, current_x_max, 100)
                y_line = s * x_line + b

                if is_doubling:
                    self.ax.plot(x_line, y_line, "g--", alpha=0.4, label="Tangent Line")
                else:
                    self.ax.plot(
                        x_line, y_line, "r--", alpha=0.4, label="Addition Line"
                    )

                # Calculate and plot intersection point
                x_intersect, y_intersect = calculate_intersection(s, b, P1, P2)
                self.ax.scatter(
                    [x_intersect],
                    [y_intersect],
                    c="orange",
                    s=100,
                    alpha=0.6,
                    label="Intersection",
                )

                # Plot reflected point (this should match the next point)
                self.ax.scatter(
                    [x_intersect],
                    [-y_intersect],
                    c="yellow",
                    s=100,
                    alpha=0.6,
                    label="Reflection",
                )

                # Draw vertical line from intersection to reflection
                self.ax.plot(
                    [x_intersect, x_intersect],
                    [y_intersect, -y_intersect],
                    "k--",
                    alpha=0.3,
                )

            prev_point = point

        # Add line between G and 4G to show final movement
        G_point = self.all_points[0][0]  # First point (G)
        for point, label in self.all_points:
            if "4G" in label:
                # Found 4G point
                s, b = calculate_line(G_point, point)
                x_line = np.linspace(current_x_min, current_x_max, 100)
                y_line = s * x_line + b
                self.ax.plot(x_line, y_line, "b--", alpha=0.6, label="G to 4G Line")

                # Calculate and plot intersection
                x_intersect, y_intersect = calculate_intersection(s, b, G_point, point)
                self.ax.scatter(
                    [x_intersect], [y_intersect], c="orange", s=100, alpha=0.6
                )
                self.ax.scatter(
                    [x_intersect], [-y_intersect], c="yellow", s=100, alpha=0.6
                )
                self.ax.plot(
                    [x_intersect, x_intersect],
                    [y_intersect, -y_intersect],
                    "k--",
                    alpha=0.3,
                )
                break

        # Set view limits
        self.ax.set_xlim(current_x_min, current_x_max)
        self.ax.set_ylim(current_y_min, current_y_max)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")

        # Add explanation
        explanation = f"""
Double-and-Add Algorithm for {self.private_key}:
• Binary: {bin(self.private_key)[2:]}
• Need powers: {', '.join(str(2**i) + 'G' for i in range(len(bin(self.private_key)[2:])) if bin(self.private_key)[2:][-(i+1)] == '1')}

Key Information:
• Private Key: {self.private_key}
• Generator Point G: ({self.G[0]:.2f}, {self.G[1]:.2f})
• Public Key: ({self.all_points[-1][0][0]:.2f}, {self.all_points[-1][0][1]:.2f})

Color Coding:
• Green: Generator Point (G)
• Blue: Doubling Steps (2G, 4G, 8G, ...)
• Purple: Addition Steps
• Red: Final Public Key
• Orange: Intersection Points
• Yellow: Reflected Points
• Dashed Green: Tangent Lines (Doubling)
• Dashed Red: Addition Lines
"""
        # Position explanation box based on current view
        text_x = current_x_max + (current_x_max - current_x_min) * 0.05
        self.ax.text(
            text_x, 0, explanation, fontsize=14, bbox=dict(facecolor="white", alpha=0.8)
        )

        # Add legend with unique entries
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(1.15, 1),
            loc="upper right",
            fontsize=12,
        )


# Create and show the visualization
visualizer = ECDSAVisualizer()
plt.tight_layout()
plt.show()
