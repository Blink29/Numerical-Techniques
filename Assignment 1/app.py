import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd

def f(x):
    return np.sin(x/4) + np.cos(x/4)  # Example function

class LagrangeInterpolator:
    def __init__(self, func):
        self.func = func
        self.num_points = 1000  # Use 1000 points for interpolation
        self.x_vals = None
        self.y_vals = None

    def generate_points(self, x_min=0, x_max=1000):
        """Generate evenly spaced points from the function"""
        self.x_vals = np.linspace(x_min, x_max, self.num_points)
        self.y_vals = self.func(self.x_vals)

    def lagrange_polynomial(self, x, idx):
        """Calculate the Lagrange polynomial for the window defined by idx (3 consecutive points)"""
        i, j, k = idx
        y0, y1, y2 = self.y_vals[i], self.y_vals[j], self.y_vals[k]
        x0, x1, x2 = self.x_vals[i], self.x_vals[j], self.x_vals[k]

        L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
        L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
        L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))

        return y0 * L0 + y1 * L1 + y2 * L2

    def plot(self):
        if self.x_vals is None:
            raise ValueError("Points not generated")
        
        x_plot = np.linspace(self.x_vals[0], self.x_vals[-1], 10000)
        y_interp = []

        # Slide through the data in windows of 3 points to get the interpolation curve
        for i in range(self.num_points - 2):  # We need at least 3 points
            idx = (i, i + 1, i + 2)
            y_interp_window = [self.lagrange_polynomial(x, idx) for x in x_plot]
            y_interp.append(y_interp_window)
        
        # Flatten the y_interp list to make it a 1D array for plotting
        y_interp = np.array(y_interp).flatten()
        x_interp = np.repeat(x_plot, self.num_points - 2)

        # Plot the original function and the Lagrange interpolation
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_vals, self.y_vals, 'ko', label='Original Points', markersize=4)
        plt.plot(x_interp, y_interp, 'r--', label='Lagrange Interpolation (Sliding Window)')
        plt.legend()
        plt.grid(True)
        plt.title('Lagrange Interpolation Using Sliding Windows of 3 Points')
        plt.show()

    def plot_error(self, x_points):
        if self.x_vals is None:
            raise ValueError("Points not generated")
            
        # Calculate derivatives
        orig_slope = nd.Derivative(self.func, n=1)
        interp_slope = nd.Derivative(self.lagrange_polynomial, n=1)
        
        errors = [abs(orig_slope(x) - interp_slope(x)) for x in x_points]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_points, errors, 'g-', label='Slope Error')
        plt.xlabel('x')
        plt.ylabel('Error')
        plt.title('Interpolation Slope Error')
        plt.grid(True)
        plt.show()

# Usage
if __name__ == "__main__":
    lagrange_interp = LagrangeInterpolator(f)
    lagrange_interp.generate_points()  # Generate points using the original function
    lagrange_interp.plot()  # Plot the interpolation with sliding window
    
    test_points = np.linspace(0, 1000, 100)
    lagrange_interp.plot_error(test_points)  # Plot the error in the slope
