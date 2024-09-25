import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd

class CubicSplineInterpolator:
    def __init__(self, func, num_points):
        self.func = func
        self.num_points = num_points
        self.x_vals = None
        self.y_vals = None
        self.a = None
        self.b = None
        self.c = None
        self.d = None
    
    def generate_points(self):
        self.x_vals = np.sort(np.random.uniform(0, 1000, self.num_points))
        self.y_vals = self.func(self.x_vals)
    
    def compute_cubic_spline(self):
        n = self.num_points - 1
        h = np.diff(self.x_vals)
        alpha = np.zeros(n)
        
        for i in range(1, n):
            alpha[i] = (3/h[i] * (self.y_vals[i+1] - self.y_vals[i]) - 3/h[i-1] * (self.y_vals[i] - self.y_vals[i-1]))

        l = np.ones(n+1)
        mu = np.zeros(n)
        z = np.zeros(n+1)

        for i in range(1, n):
            l[i] = 2 * (self.x_vals[i+1] - self.x_vals[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

        self.c = np.zeros(n+1)
        self.b = np.zeros(n)
        self.d = np.zeros(n)
        self.a = self.y_vals[:-1]

        for j in range(n-1, -1, -1):
            self.c[j] = z[j] - mu[j] * self.c[j+1]
            self.b[j] = (self.y_vals[j+1] - self.y_vals[j]) / h[j] - h[j] * (self.c[j+1] + 2 * self.c[j]) / 3
            self.d[j] = (self.c[j+1] - self.c[j]) / (3 * h[j])
    
    def spline(self, x):
        i = np.searchsorted(self.x_vals, x) - 1
        i = max(min(i, self.num_points - 2), 0)
        dx = x - self.x_vals[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx**2 + self.d[i] * dx**3

    def compute_error(self, x_random):
        f_prime = np.array([nd.Derivative(self.func)(x) for x in x_random])
        spline_prime = np.array([self.spline_derivative(x) for x in x_random])
        return f_prime, spline_prime, np.abs(f_prime - spline_prime)

    def spline_derivative(self, x):
        i = np.searchsorted(self.x_vals, x) - 1
        i = max(min(i, self.num_points - 2), 0)
        dx = x - self.x_vals[i]
        return self.b[i] + 2 * self.c[i] * dx + 3 * self.d[i] * dx**2

    def plot(self):
        x_interp = np.linspace(0, 1000, 1000)
        y_interp = np.array([self.spline(x) for x in x_interp])
        y_orig = self.func(x_interp)

        plt.figure(figsize=(12, 6))
        plt.plot(x_interp, y_orig, label="Original Function", color="blue", linewidth=2)
        plt.plot(x_interp, y_interp, label="Cubic Spline Interpolation", linestyle="--", color="red", linewidth=2)
        plt.title("Cubic Spline Interpolation vs Original Function")
        plt.legend()
        plt.savefig('spline_vs_function2.png')  
        plt.show()

    def plot_error(self, x_random):
        f_prime, spline_prime, error = self.compute_error(x_random)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(x_random, error, label="Slope Error (|f'(x) - S'(x)|)", color="purple", marker='o')
        plt.title("Error between Function Slope and Cubic Spline Slope (Scatter Plot)")
        plt.xlabel("x")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig('slope_error_scatter2.png')  
        plt.show()

        avg_error = np.mean(error)
        print(f"Average error in slope: {avg_error:.6f}")

# Example usage
if __name__ == "__main__":
    # Define a sample function
    def f(x):
       return np.sin(x/100) + np.cos(x/100)

    # Create an instance of the class with the function and number of points
    num_points = 1000
    spline_interp = CubicSplineInterpolator(f, num_points)

    # Generate random points
    spline_interp.generate_points()

    # Compute the cubic spline
    spline_interp.compute_cubic_spline()

    # Plot the original function and the cubic spline interpolation
    spline_interp.plot()

    # Generate random points to evaluate the error
    random_points = np.random.uniform(0, 1000, 1000)
    # random_points = np.random(0, 1000)
    
    # Plot the error between the original function's slope and cubic spline's slope
    spline_interp.plot_error(random_points)