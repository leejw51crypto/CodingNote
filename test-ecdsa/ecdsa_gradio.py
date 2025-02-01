import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import io
from PIL import Image


class ECDSAInspector:
    def __init__(self):
        self.setup_curve_params()
        self.calculate_points()

    def setup_curve_params(self):
        """Initialize curve parameters"""
        self.curve_a = 0  # secp256k1 parameter a
        self.curve_b = 7  # secp256k1 parameter b
        self.Gx = 1  # Generator point x (simplified for visualization)
        self.Gy = np.sqrt(self.Gx**3 + self.curve_b)  # Generator point y
        self.G = (self.Gx, self.Gy)
        self.zoom_level = 1.0
        self.center = [0, 0]
        self.private_key = 5
        self.points = []
        self.point_operations = []
        self.current_point_index = 0
        self.sum_formulas = []  # Track sum formulas

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
        coeffs = [1, -s * s, -2 * s * b, -(b * b - 7)]
        roots = np.roots(coeffs)
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
                "sum_formula": "G",  # Initial point
            }
        )

        binary = bin(self.private_key)[2:]
        doubles = [self.G]

        # Calculate doubling points
        for i in range(len(binary) - 1):
            prev_point = doubles[-1]
            s, b, _ = self.calculate_line_params(prev_point)
            intersections = self.calculate_intersection(s, b)

            for x, y in intersections:
                if abs(x - prev_point[0]) > 1e-10 or abs(y - prev_point[1]) > 1e-10:
                    doubled = (x, -y)
                    break

            doubles.append(doubled)
            self.points.append((doubled, f"2^{i+1}G"))
            self.point_operations.append(
                {
                    "type": "doubling",
                    "prev_point": prev_point,
                    "line_params": (s, b),
                    "intersection": (x, y),
                    "sum_formula": f"2 × {2**i}G = {2**(i+1)}G",  # Doubling formula
                }
            )

        # Calculate running sum
        result = None
        running_sum = 0
        for i, bit in enumerate(binary):
            if bit == "1":
                power = len(binary) - 1 - i
                power_val = 2**power
                if result is None:
                    result = doubles[power]
                    running_sum = power_val
                else:
                    prev_point = result
                    add_point = doubles[power]
                    s, b, _ = self.calculate_line_params(prev_point, add_point)
                    intersections = self.calculate_intersection(s, b)

                    for x, y in intersections:
                        if (
                            abs(x - prev_point[0]) > 1e-10
                            or abs(y - prev_point[1]) > 1e-10
                        ) and (
                            abs(x - add_point[0]) > 1e-10
                            or abs(y - add_point[1]) > 1e-10
                        ):
                            result = (x, -y)
                            break

                    prev_sum = running_sum
                    running_sum += power_val
                    self.points.append((result, f"Sum + {power_val}G"))
                    self.point_operations.append(
                        {
                            "type": "addition",
                            "prev_point": prev_point,
                            "add_point": add_point,
                            "line_params": (s, b),
                            "intersection": (x, y),
                            "sum_formula": f"{prev_sum}G + {power_val}G = {running_sum}G",  # Addition formula
                        }
                    )

        self.public_key = result

    def generate_plot(self):
        """Generate plot for current state"""
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

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

        plt.plot(x, y_pos, "b-", alpha=0.3, label="y² = x³ + 7")
        plt.plot(x, y_neg, "b-", alpha=0.3)

        # Plot all points with reduced emphasis
        for i, (point, label) in enumerate(self.points):
            if i == self.current_point_index:
                continue
            color = "gray"
            plt.scatter([point[0]], [point[1]], c=color, s=50, alpha=0.4)
            plt.annotate(
                label,
                (point[0], point[1]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(facecolor="white", alpha=0.3),
                alpha=0.4,
            )

        # Plot current point and its operation
        if self.current_point_index < len(self.points):
            current_point, current_label = self.points[self.current_point_index]
            operation = self.point_operations[self.current_point_index]

            if operation["type"] != "initial":
                # Plot line
                s, b = operation["line_params"]
                x_line = np.linspace(x_min, x_max, 100)
                y_line = s * x_line + b
                line_style = "g--" if operation["type"] == "doubling" else "r--"
                plt.plot(x_line, y_line, line_style, alpha=0.8)

                # Plot intersection point
                x_int, y_int = operation["intersection"]
                plt.scatter([x_int], [y_int], c="orange", s=100, alpha=0.8)
                plt.scatter([x_int], [-y_int], c="yellow", s=100, alpha=0.8)
                plt.plot([x_int, x_int], [y_int, -y_int], "k--", alpha=0.5)

                # Plot previous point(s)
                prev_point = operation["prev_point"]
                plt.scatter(
                    [prev_point[0]], [prev_point[1]], c="blue", s=100, alpha=0.8
                )

                if operation["type"] == "addition" and "add_point" in operation:
                    add_point = operation["add_point"]
                    plt.scatter(
                        [add_point[0]], [add_point[1]], c="purple", s=100, alpha=0.8
                    )

            # Plot current point
            plt.scatter(
                [current_point[0]], [current_point[1]], c="red", s=150, alpha=1.0
            )
            plt.annotate(
                current_label,
                (current_point[0], current_point[1]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(facecolor="white", alpha=0.9),
            )

        plt.grid(True, alpha=0.3)
        plt.title("ECDSA Point Operations")
        plt.axis("equal")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # Save plot to bytes buffer and convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    def get_info_text(self):
        """Get information text for current state"""
        if not self.points:  # Handle case where no points are calculated
            return """
            No points calculated.
            Please enter a valid private key.
            """

        info_text = f"""
        Curve: y² = x³ + {self.curve_b}
        
        Private Key: {self.private_key}
        Binary: {bin(self.private_key)[2:]}
        
        Generator Point (G):
        x: {self.G[0]:.4f}
        y: {self.G[1]:.4f}
        """

        if self.public_key:  # Only add public key info if it exists
            info_text += f"""
        Public Key:
        x: {self.public_key[0]:.4f}
        y: {self.public_key[1]:.4f}
        """

        if self.points and self.current_point_index < len(self.points):
            current_point = self.points[self.current_point_index]
            operation = self.point_operations[self.current_point_index]

            info_text += f"""
        Current Point:
        {current_point[1]}
        x: {current_point[0][0]:.4f}
        y: {current_point[0][1]:.4f}
        
        Operation: {operation['type'].title()}
        Formula: {operation['sum_formula']}
        """

            if operation["type"] != "initial":
                s, b = operation["line_params"]
                info_text += f"""
        Line Equation:
        y = {s:.4f}x + {b:.4f}
        
        Intersection Point:
        x: {operation['intersection'][0]:.4f}
        y: {operation['intersection'][1]:.4f}
        """

        return info_text


def update_private_key(private_key, zoom, inspector):
    try:
        if private_key is None or private_key <= 0:
            return (
                None,
                "Please enter a positive integer for the private key.",
                gr.update(),
            )

        inspector.private_key = int(private_key)
        inspector.zoom_level = float(zoom)
        inspector.calculate_points()
        plot_image = inspector.generate_plot()
        info_text = inspector.get_info_text()
        return (
            plot_image,
            info_text,
            gr.update(maximum=len(inspector.points) - 1, value=0),
        )
    except ValueError as e:
        return None, f"Invalid input: {str(e)}", gr.update()
    except Exception as e:
        return None, f"Error: {str(e)}", gr.update()


def update_zoom(zoom, private_key, inspector):
    try:
        inspector.zoom_level = float(zoom)
        plot_image = inspector.generate_plot()
        return plot_image
    except ValueError:
        return None


def update_point_index(point_index, inspector):
    inspector.current_point_index = int(point_index)
    plot_image = inspector.generate_plot()
    info_text = inspector.get_info_text()
    return plot_image, info_text


def create_ui():
    inspector = ECDSAInspector()

    # Generate initial values
    plot_image = inspector.generate_plot()
    info_text = inspector.get_info_text()

    with gr.Blocks(title="ECDSA Inspector") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                plot_output = gr.Image(label="ECDSA Visualization", value=plot_image)
            with gr.Column(scale=1):
                info_output = gr.Textbox(label="Information", lines=20, value=info_text)

        with gr.Row():
            private_key_input = gr.Number(value=5, label="Private Key", precision=0)
            zoom_slider = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, label="Zoom Level"
            )
            point_slider = gr.Slider(
                minimum=0,
                maximum=len(inspector.points) - 1,
                value=0,
                step=1,
                label="Point Index",
            )

        # Set up event handlers
        private_key_input.change(
            fn=lambda pk, z: update_private_key(pk, z, inspector),
            inputs=[private_key_input, zoom_slider],
            outputs=[plot_output, info_output, point_slider],
        )

        zoom_slider.change(
            fn=lambda z, pk: update_zoom(z, pk, inspector),
            inputs=[zoom_slider, private_key_input],
            outputs=plot_output,
        )

        point_slider.change(
            fn=lambda p: update_point_index(p, inspector),
            inputs=[point_slider],
            outputs=[plot_output, info_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
