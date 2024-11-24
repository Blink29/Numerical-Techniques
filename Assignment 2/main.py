import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Parameters
alpha = 0.0001
L = 1.0
N = 120  
dx = L / N

# Create finer grid
x = np.linspace(0, L, N+1)

# Initialize arrays with new size
Q = np.zeros(N+1)
Q[int(0.25 * N)] = 1
Q[int(0.75 * N)] = -1

# Recreate coefficient matrix A and vector b with new size
A = np.zeros((N+1, N+1))
b = np.zeros(N+1)

# Fill A and b with finer discretization
for i in range(1, N):
    A[i, i-1] = alpha / dx**2
    A[i, i] = -2 * alpha / dx**2
    A[i, i+1] = alpha / dx**2
    b[i] = -Q[i]

# Apply boundary conditions
A[0, 0] = 1
A[N, N] = 1
b[0] = 0
b[N] = 0

# Direct method using numpy's linear solver
def gaussian_elimination(A, b):
    n = len(A)
    # Make copies to avoid modifying original arrays
    A = A.copy()
    b = b.copy()
    x = np.zeros(n)
    
    # Forward elimination
    for i in range(n-1):
        # Find pivot
        pivot = A[i][i]
        
        # Eliminate column i
        for j in range(i+1, n):
            factor = A[j][i] / pivot
            for k in range(i, n):
                A[j][k] = A[j][k] - factor * A[i][k]
            b[j] = b[j] - factor * b[i]
    
    # Back substitution
    x[n-1] = b[n-1] / A[n-1][n-1]
    for i in range(n-2, -1, -1):
        sum_ax = 0
        for j in range(i+1, n):
            sum_ax += A[i][j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i][i]
    
    return x

# Use the function with your existing A and b
T_gauss_direct = gaussian_elimination(A, b)

# Gauss-Seidel method
def gauss_seidel(A, b, tol=1e-10, max_iter=1000):
    T = np.zeros_like(b)
    for _ in range(max_iter):
        T_old = T.copy()
        for i in range(len(T)):
            sigma = 0
            for j in range(len(T)):
                if j != i:
                    sigma += A[i, j] * T[j]
            T[i] = (b[i] - sigma) / A[i, i]
        if np.linalg.norm(T - T_old, ord=np.inf) < tol:
            break
    return T

T_gauss_seidal = gauss_seidel(A, b)

# Calculate error
error = np.abs(T_gauss_direct - T_gauss_seidal)

# Create finer x points for smooth plotting
x_fine = np.linspace(0, L, 1000)

# Create interpolation functions
T_direct_interp = interp1d(x, T_gauss_direct, kind='cubic')
T_gs_interp = interp1d(x, T_gauss_seidal, kind='cubic')
error_interp = interp1d(x, error, kind='cubic')

# First Figure - Temperature Distribution
plt.figure(1, figsize=(10, 6))
plt.plot(x_fine, T_direct_interp(x_fine), label="Direct Method")
plt.plot(x_fine, T_gs_interp(x_fine), label="Gauss-Seidel", linestyle='--')
plt.xlabel("x")
plt.ylabel("Temperature (T)")
plt.title("Temperature Distribution")
plt.legend()
plt.grid(True)
plt.savefig("temperature_distribution.png")

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot temperature distribution
ax1.plot(x_fine, T_direct_interp(x_fine), label="Direct Method")
ax1.plot(x_fine, T_gs_interp(x_fine), label="Gauss-Seidel", linestyle='--')
ax1.set_xlabel("x")
ax1.set_ylabel("Temperature (T)")
ax1.set_title("Temperature Distribution")
ax1.legend()
ax1.grid(True)

# Plot error
ax2.plot(x_fine, error_interp(x_fine), 'r-', label="Absolute Error")
ax2.set_xlabel("x")
ax2.set_ylabel("Absolute Error")
ax2.set_title("Error between Direct and Gauss-Seidel Methods")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("temperature_and_error.png")
plt.show()

# Print the results for verification
print("Gaussian Direct Solution:", T_gauss_direct)
print("Gauss-Seidel Method Solution:", T_gauss_seidal)
print("Maximum Error:", np.max(error))
