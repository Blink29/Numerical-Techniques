import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd

class LagrangeInterpolator:
    def __init__(self, func, num_points):
        self.func = func
        self.num_points = num_points
        self.x_vals = None
        self.y_vals = None
    
    def generate_points(self):
        self.x_vals = np.sort(np.random.uniform(0, 1000, self.num_points))
        self.y_vals = self.func(self.x_vals)
    
    def lagrange_polynomial(self, x):
        total = 0.0
        for i in range(self.num_points):
            term = self.y_vals[i]
            for j in range(self.num_points):
                if i != j:
                    term *= (x - self.x_vals[j]) / (self.x_vals[i] - self.x_vals[j])
            total += term
        return total
    
    def lagrange_derivative(self, x):
        total = 0.0
        for i in range(self.num_points):
            term = self.y_vals[i]
            for j in range(self.num_points):
                if i != j:
                    term *= (x - self.x_vals[j]) / (self.x_vals[i] - self.x_vals[j])
            total += term
        derivative = 0.0
        for i in range(self.num_points):
            term = 1.0
            for j in range(self.num_points):
                if i != j:
                    term *= (x - self.x_vals[j]) / (self.x_vals[i] - self.x_vals[j])
            derivative += term * self.y_vals[i] / (self.x_vals[i] - x)
        return derivative
    
    def compute_error(self, x_random):
        f_prime = np.array([nd.Derivative(self.func)(x) for x in x_random])
        lagrange_prime = np.array([self.lagrange_derivative(x) for x in x_random])
        return f_prime, lagrange_prime, np.abs(f_prime - lagrange_prime)

    def plot(self):
        x_interp = np.linspace(0, 1000, 1000)
        y_interp = np.array([self.lagrange_polynomial(x) for x in x_interp])
        y_orig = self.func(x_interp)

        plt.figure(figsize=(12, 6))
        plt.plot(x_interp, y_orig, label="Original Function", color="blue", linewidth=2)
        plt.plot(x_interp, y_interp, label="Lagrange Interpolation", linestyle="--", color="red", linewidth=2)
        plt.title("Lagrange Interpolation vs Original Function")
        plt.legend()
        plt.savefig('lagrange_vs_function.png')  
        plt.show()

    def plot_error(self, x_random):
        f_prime, lagrange_prime, error = self.compute_error(x_random)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(x_random, error, label="Slope Error (|f'(x) - S'(x)|)", color="purple", marker='o')
        plt.title("Error between Function Slope and Lagrange Slope (Scatter Plot)")
        plt.xlabel("x")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig('slope_error_scatter_lagrange.png')  
        plt.show()

        avg_error = np.mean(error)
        print(f"Average error in slope: {avg_error:.6f}")

# Example usage
if __name__ == "__main__":
    # Define a sample function
    def f(x):
        return np.sin(x) + np.cos(x)

    # Create an instance of the class with the function and number of points
    num_points = 1000
    lagrange_interp = LagrangeInterpolator(f, num_points)

    # Generate random points
    lagrange_interp.generate_points()

    # Plot the original function and the Lagrange interpolation
    lagrange_interp.plot()

    # Generate random points to evaluate the error
    random_points = np.random.uniform(0, 1000, 1000)

    # Plot the error between the original function's slope and Lagrange's slope
    lagrange_interp.plot_error(random_points)
