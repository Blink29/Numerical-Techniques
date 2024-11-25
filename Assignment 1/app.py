import numpy as np
import matplotlib.pyplot as plt
from numdifftools import Derivative

def f(x):
    return np.sin(x / 4) + np.cos(x / 4)

class PiecewiseLagrangeInterpolator:
    def __init__(self, func, num_points):
        self.func = func
        self.num_points = num_points
        self.x_vals = None
        self.y_vals = None

    def generate_points(self, x_min=0, x_max=1000):
        self.x_vals = np.linspace(x_min, x_max, self.num_points)
        self.y_vals = self.func(self.x_vals)

    def lagrange_2nd_order(self, x, x_window, y_window):
        term0 = y_window[0] * ((x - x_window[1]) * (x - x_window[2])) / ((x_window[0] - x_window[1]) * (x_window[0] - x_window[2]))
        term1 = y_window[1] * ((x - x_window[0]) * (x - x_window[2])) / ((x_window[1] - x_window[0]) * (x_window[1] - x_window[2]))
        term2 = y_window[2] * ((x - x_window[0]) * (x - x_window[1])) / ((x_window[2] - x_window[0]) * (x_window[2] - x_window[1]))
        return term0 + term1 + term2

    def interpolate(self, x_dense):
        y_interp = np.zeros_like(x_dense)
        n = len(self.x_vals)

        for i in range(n - 2):  # Slide window of 3 points
            x_window = self.x_vals[i:i + 3]
            y_window = self.y_vals[i:i + 3]

            # Find points in the current segment to interpolate
            mask = (x_dense >= x_window[0]) & (x_dense <= x_window[-1])
            y_interp[mask] = [self.lagrange_2nd_order(x, x_window, y_window) for x in x_dense[mask]]

        return y_interp

    def lagrange_derivative(self, x_dense):
        y_derivative = np.zeros_like(x_dense)
        n = len(self.x_vals)

        for i in range(n - 2):  # Slide window of 3 points
            x_window = self.x_vals[i:i + 3]
            y_window = self.y_vals[i:i + 3]

            # Derivative terms for 2nd-order Lagrange polynomial
            def derivative_lagrange(x):
                term0 = y_window[0] * ((x - x_window[1]) + (x - x_window[2])) / ((x_window[0] - x_window[1]) * (x_window[0] - x_window[2]))
                term1 = y_window[1] * ((x - x_window[0]) + (x - x_window[2])) / ((x_window[1] - x_window[0]) * (x_window[1] - x_window[2]))
                term2 = y_window[2] * ((x - x_window[0]) + (x - x_window[1])) / ((x_window[2] - x_window[0]) * (x_window[2] - x_window[1]))
                return term0 + term1 + term2

            # Find points in the current segment to compute the derivative
            mask = (x_dense >= x_window[0]) & (x_dense <= x_window[-1])
            y_derivative[mask] = [derivative_lagrange(x) for x in x_dense[mask]]

        return y_derivative

    def plot_and_save(self):
        x_dense = np.linspace(self.x_vals[0], self.x_vals[-1], 10000)
        y_orig = self.func(x_dense)
        y_interp = self.interpolate(x_dense)

        # Plot original and interpolation
        plt.figure(figsize=(12, 6))
        plt.plot(x_dense, y_orig, label="Original Function", color="blue", linewidth=2)
        plt.plot(x_dense, y_interp, label="Lagrange Interpolation", linestyle="--", color="red", linewidth=2)
        plt.title("Piecewise 2nd-Order Lagrange Interpolation vs Original Function")
        plt.legend()
        plt.grid(True)
        plt.savefig("lagrange_interpolation_plot.png")
        plt.show()

        # Derivative comparison
        orig_derivative = Derivative(self.func)(x_dense)
        lagrange_derivative = self.lagrange_derivative(x_dense)
        derivative_error = np.abs(orig_derivative - lagrange_derivative)

        # Plot derivative error
        plt.figure(figsize=(12, 6))
        plt.plot(x_dense, derivative_error, label="Derivative Error (|f'(x) - L'(x)|)", color="purple", linewidth=2)
        plt.title("Error Between Derivatives of Original Function and Lagrange Interpolation")
        plt.xlabel("x")
        plt.ylabel("Error")
        plt.grid(True)
        plt.legend()
        plt.savefig("lagrange_derivative_error_plot.png")
        plt.show()

        print(f"Maximum derivative error: {np.max(derivative_error):.6f}")

if __name__ == "__main__":
    num_points = 1000
    lagrange_interp = PiecewiseLagrangeInterpolator(f, num_points)
    lagrange_interp.generate_points()
    lagrange_interp.plot_and_save()
