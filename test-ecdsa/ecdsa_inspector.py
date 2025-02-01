import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider
import matplotlib.gridspec as gridspec


class ECDSAInspector:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 12))
        self.setup_layout()
        self.setup_curve_params()
        self.setup_controls()
        self.setup_event_handlers()
        self.update_visualization()

    def setup_layout(self):
        """Setup the layout with main plot and control panel"""
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[4, 1])
        self.ax_main = plt.subplot(gs[0, 0])  # Main curve plot
        self.ax_info = plt.subplot(gs[0, 1])  # Information panel
        self.ax_controls = plt.subplot(gs[1, :])  # Controls panel

        # Clear info panel and set title
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.set_title("ECDSA Information", pad=20)

    def setup_curve_params(self):
        """Initialize curve parameters"""
        self.curve_a = 0  # secp256k1 parameter a
        self.curve_b = 7  # secp256k1 parameter b
        self.Gx = 1  # Generator point x (simplified for visualization)
        self.Gy = np.sqrt(self.Gx**3 + self.curve_b)  # Generator point y
        self.G = (self.Gx, self.Gy)

        # View parameters
        self.zoom_level = 1.0
        self.center = [0, 0]
        self.private_key = 5
        self.points = []
        self.point_operations = []  # Store operation details for each point
        self.calculate_points()

    def calculate_line_params(self, P1, P2=None):
        """Calculate line parameters for point addition or doubling"""
        if P2 is None or (P1[0] == P2[0] and P1[1] == P2[1]):
            # Point doubling
            x1, y1 = P1
            s = (3 * x1 * x1 + self.curve_a) / (2 * y1)
            is_doubling = True
        else:
            # Point addition
            x1, y1 = P1
            x2, y2 = P2
            s = (y2 - y1) / (x2 - x1)
            is_doubling = False

        b = y1 - s * x1
        return s, b, is_doubling

    def calculate_intersection(self, s, b):
        """Calculate intersection point with the curve"""
        # Solve: (sx + b)² = x³ + 7
        # x³ - s²x² - 2sbx - (b² - 7) = 0
        coeffs = [1, -s * s, -2 * s * b, -(b * b - 7)]
        roots = np.roots(coeffs)

        # Find real roots
        real_roots = [
            (float(root.real), s * float(root.real) + b)
            for root in roots
            if abs(root.imag) < 1e-10
        ]
        return real_roots

    def calculate_points(self):
        """Calculate points for current private key"""
        self.points = []
        self.point_operations = []
        current = self.G
        self.points.append((current, "G"))
        self.point_operations.append(
            {
                "type": "initial",
                "prev_point": None,
                "line_params": None,
                "intersection": None,
            }
        )

        # Convert private key to binary
        binary = bin(self.private_key)[2:]

        # Calculate doubling points
        doubles = [self.G]
        for i in range(len(binary) - 1):
            prev_point = doubles[-1]
            s, b, _ = self.calculate_line_params(prev_point)
            intersections = self.calculate_intersection(s, b)

            # Find the new point (not the original point)
            for x, y in intersections:
                if abs(x - prev_point[0]) > 1e-10 or abs(y - prev_point[1]) > 1e-10:
                    doubled = (x, -y)  # Reflect y-coordinate
                    break

            doubles.append(doubled)
            self.points.append((doubled, f"2^{i+1}G"))
            self.point_operations.append(
                {
                    "type": "doubling",
                    "prev_point": prev_point,
                    "line_params": (s, b),
                    "intersection": (x, y),
                }
            )

        # Calculate running sum
        result = None
        for i, bit in enumerate(binary):
            if bit == "1":
                power = len(binary) - 1 - i
                if result is None:
                    result = doubles[power]
                else:
                    prev_point = result
                    add_point = doubles[power]
                    s, b, _ = self.calculate_line_params(prev_point, add_point)
                    intersections = self.calculate_intersection(s, b)

                    # Find the new point
                    for x, y in intersections:
                        if (
                            abs(x - prev_point[0]) > 1e-10
                            or abs(y - prev_point[1]) > 1e-10
                        ) and (
                            abs(x - add_point[0]) > 1e-10
                            or abs(y - add_point[1]) > 1e-10
                        ):
                            result = (x, -y)  # Reflect y-coordinate
                            break

                    self.points.append((result, f"Sum + {2**power}G"))
                    self.point_operations.append(
                        {
                            "type": "addition",
                            "prev_point": prev_point,
                            "add_point": add_point,
                            "line_params": (s, b),
                            "intersection": (x, y),
                        }
                    )

        self.public_key = result

    def setup_controls(self):
        """Setup control panel with buttons and input fields"""
        # Private key input
        self.private_key_box = TextBox(
            plt.axes([0.1, 0.15, 0.2, 0.05]),
            "Private Key:",
            initial=str(self.private_key),
        )

        # Zoom controls
        self.zoom_slider = Slider(
            plt.axes([0.4, 0.15, 0.2, 0.05]), "Zoom", 0.1, 5.0, valinit=1.0
        )

        # Navigation buttons
        button_width = 0.1
        button_height = 0.05
        spacing = 0.02

        self.btn_reset = Button(
            plt.axes([0.7, 0.15, button_width, button_height]), "Reset View"
        )

        self.btn_prev = Button(
            plt.axes(
                [0.7 + button_width + spacing, 0.15, button_width / 2, button_height]
            ),
            "←",
        )

        self.btn_next = Button(
            plt.axes(
                [
                    0.7 + 1.5 * button_width + spacing,
                    0.15,
                    button_width / 2,
                    button_height,
                ]
            ),
            "→",
        )

    def setup_event_handlers(self):
        """Setup event handlers for controls"""
        self.private_key_box.on_submit(self.on_private_key_change)
        self.zoom_slider.on_changed(self.on_zoom_change)
        self.btn_reset.on_clicked(self.on_reset)
        self.btn_prev.on_clicked(lambda x: self.navigate_points(-1))
        self.btn_next.on_clicked(lambda x: self.navigate_points(1))

        # Mouse events for panning
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.is_panning = False
        self.pan_start = None
        self.current_point_index = 0

    def on_private_key_change(self, text):
        """Handle private key input change"""
        try:
            new_key = int(text)
            if new_key > 0:
                self.private_key = new_key
                self.calculate_points()
                self.update_visualization()
        except ValueError:
            pass

    def on_zoom_change(self, val):
        """Handle zoom slider change"""
        self.zoom_level = val
        self.update_visualization()

    def on_reset(self, event):
        """Reset view to default"""
        self.zoom_level = 1.0
        self.center = [0, 0]
        self.current_point_index = 0
        self.zoom_slider.set_val(1.0)
        self.update_visualization()

    def navigate_points(self, direction):
        """Navigate through points (1 for next, -1 for previous)"""
        self.current_point_index = (self.current_point_index + direction) % len(
            self.points
        )
        self.center = [
            self.points[self.current_point_index][0][0],
            self.points[self.current_point_index][0][1],
        ]
        self.update_visualization()

    def on_mouse_press(self, event):
        """Handle mouse press for panning"""
        if event.inaxes == self.ax_main and event.button == 1:
            self.is_panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        """Handle mouse release for panning"""
        self.is_panning = False
        self.pan_start = None

    def on_mouse_move(self, event):
        """Handle mouse movement for panning"""
        if self.is_panning and event.inaxes == self.ax_main and self.pan_start:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            self.center[0] -= dx
            self.center[1] -= dy
            self.update_visualization()
            self.pan_start = (event.xdata, event.ydata)

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes == self.ax_main:
            if event.button == "up":
                self.zoom_level = min(5.0, self.zoom_level * 1.1)
            else:
                self.zoom_level = max(0.1, self.zoom_level / 1.1)
            self.zoom_slider.set_val(self.zoom_level)

    def update_visualization(self):
        """Update the entire visualization"""
        self.ax_main.clear()
        self.ax_info.clear()
        self.ax_info.axis("off")

        # Calculate view bounds
        x_range = 10 / self.zoom_level
        y_range = 10 / self.zoom_level
        x_min = self.center[0] - x_range
        x_max = self.center[0] + x_range
        y_min = self.center[1] - y_range
        y_max = self.center[1] + y_range

        # Plot curve
        x = np.linspace(x_min, x_max, 1000)
        mask = x**3 + self.curve_b >= 0
        y_pos = np.zeros_like(x)
        y_neg = np.zeros_like(x)
        y_pos[mask] = np.sqrt((x**3 + self.curve_b)[mask])
        y_neg[mask] = -np.sqrt((x**3 + self.curve_b)[mask])

        self.ax_main.plot(x, y_pos, "b-", alpha=0.3, label="y² = x³ + 7")
        self.ax_main.plot(x, y_neg, "b-", alpha=0.3)

        # Plot all points with reduced emphasis
        for i, (point, label) in enumerate(self.points):
            if i == self.current_point_index:
                continue  # Skip current point, will plot it later with emphasis
            color = "gray"
            self.ax_main.scatter([point[0]], [point[1]], c=color, s=50, alpha=0.4)
            self.ax_main.annotate(
                label,
                (point[0], point[1]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(facecolor="white", alpha=0.3),
                alpha=0.4,
            )

        # Plot current point with emphasis and its operation
        if self.current_point_index < len(self.points):
            current_point, current_label = self.points[self.current_point_index]
            operation = self.point_operations[self.current_point_index]

            # Plot operation lines and points
            if operation["type"] != "initial":
                # Plot line
                s, b = operation["line_params"]
                x_line = np.linspace(x_min, x_max, 100)
                y_line = s * x_line + b
                line_style = "g--" if operation["type"] == "doubling" else "r--"
                self.ax_main.plot(
                    x_line,
                    y_line,
                    line_style,
                    alpha=0.8,
                    label=(
                        "Tangent Line"
                        if operation["type"] == "doubling"
                        else "Addition Line"
                    ),
                )

                # Plot intersection point
                x_int, y_int = operation["intersection"]
                self.ax_main.scatter(
                    [x_int], [y_int], c="orange", s=100, alpha=0.8, label="Intersection"
                )
                self.ax_main.scatter(
                    [x_int], [-y_int], c="yellow", s=100, alpha=0.8, label="Reflection"
                )
                self.ax_main.plot([x_int, x_int], [y_int, -y_int], "k--", alpha=0.5)

                # Plot previous point(s)
                prev_point = operation["prev_point"]
                self.ax_main.scatter(
                    [prev_point[0]],
                    [prev_point[1]],
                    c="blue",
                    s=100,
                    alpha=0.8,
                    label="Previous Point",
                )

                if operation["type"] == "addition" and "add_point" in operation:
                    add_point = operation["add_point"]
                    self.ax_main.scatter(
                        [add_point[0]],
                        [add_point[1]],
                        c="purple",
                        s=100,
                        alpha=0.8,
                        label="Addition Point",
                    )

            # Plot current point
            self.ax_main.scatter(
                [current_point[0]],
                [current_point[1]],
                c="red",
                s=150,
                alpha=1.0,
                label="Current Point",
            )
            self.ax_main.annotate(
                current_label,
                (current_point[0], current_point[1]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(facecolor="white", alpha=0.9),
            )

        # Update info panel with operation details
        info_text = f"""
        Curve: y² = x³ + {self.curve_b}
        
        Private Key: {self.private_key}
        Binary: {bin(self.private_key)[2:]}
        
        Generator Point (G):
        x: {self.G[0]:.4f}
        y: {self.G[1]:.4f}
        
        Public Key:
        x: {self.public_key[0]:.4f}
        y: {self.public_key[1]:.4f}
        
        Current Point:
        {self.points[self.current_point_index][1]}
        x: {self.points[self.current_point_index][0][0]:.4f}
        y: {self.points[self.current_point_index][0][1]:.4f}
        
        Operation: {self.point_operations[self.current_point_index]['type'].title()}
        """

        if self.point_operations[self.current_point_index]["type"] != "initial":
            s, b = self.point_operations[self.current_point_index]["line_params"]
            info_text += f"""
        Line Equation:
        y = {s:.4f}x + {b:.4f}
        
        Intersection Point:
        x: {self.point_operations[self.current_point_index]['intersection'][0]:.4f}
        y: {self.point_operations[self.current_point_index]['intersection'][1]:.4f}
        """

        self.ax_info.text(
            0.1,
            0.9,
            info_text,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # Set view limits and properties
        self.ax_main.set_xlim(x_min, x_max)
        self.ax_main.set_ylim(y_min, y_max)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect("equal")
        self.ax_main.set_title("ECDSA Point Operations")
        self.ax_main.legend(loc="upper right")

        plt.draw()


def main():
    inspector = ECDSAInspector()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
