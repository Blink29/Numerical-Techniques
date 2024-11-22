# Parameters
alpha = 0.0001  # Diffusion coefficient
L = 1.0         # Length of the domain
N = 10          # Number of intervals
h = L / N       # Grid spacing

# Discretize the domain
x = [i * h for i in range(N + 1)]

# Source term Q
Q = [0] * (N + 1)
Q[int(0.25 * N)] = 1.0  # Q at x=0.25
Q[int(0.75 * N)] = -1.0 # Q at x=0.75

# Assemble matrix A and vector b
A = [[0 for _ in range(N + 1)] for _ in range(N + 1)]
b = [-Q[i] * h**2 / alpha for i in range(N + 1)]

# Apply central difference for internal points
for i in range(1, N):
    A[i][i - 1] = 1
    A[i][i] = -2
    A[i][i + 1] = 1

# Boundary conditions (natural: dT/dx = 0)
A[0][0] = 1
A[0][1] = -1
b[0] = 0

A[N][N] = 1
A[N][N - 1] = -1
b[N] = 0

# --- Direct Method: Gaussian Elimination ---
def gaussian_elimination(A, b):
    n = len(b)
    # Forward elimination
    for i in range(n):
        # Make the diagonal element 1 by dividing the row
        pivot = A[i][i]
        for j in range(i, n):
            A[i][j] /= pivot
        b[i] /= pivot
        # Eliminate the current column from rows below
        for k in range(i + 1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]

    # Back substitution
    T = [0] * n
    for i in range(n - 1, -1, -1):
        T[i] = b[i] - sum(A[i][j] * T[j] for j in range(i + 1, n))
    return T

T_direct = gaussian_elimination([row[:] for row in A], b[:])

# --- Iterative Method: Gauss-Seidel ---
def gauss_seidel(A, b, tol=1e-6, max_iter=10000):
    n = len(b)
    T = [0] * n
    for _ in range(max_iter):
        T_old = T[:]
        for i in range(n):
            sum1 = sum(A[i][j] * T[j] for j in range(i))
            sum2 = sum(A[i][j] * T_old[j] for j in range(i + 1, n))
            T[i] = (b[i] - sum1 - sum2) / A[i][i]
        # Convergence check
        if max(abs(T[i] - T_old[i]) for i in range(n)) < tol:
            break
    return T

T_gs = gauss_seidel([row[:] for row in A], b[:])

# Print Results
print("Solution using Direct Method (Gaussian Elimination):")
print(T_direct)

print("\nSolution using Gauss-Seidel Method:")
print(T_gs)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(x, T_direct, label="Direct Method")
plt.plot(x, T_gs, label="Gauss-Seidel", linestyle='--')
plt.xlabel("x")
plt.ylabel("Temperature (T)")
plt.title("Temperature Distribution")
plt.legend()
plt.grid(True)
plt.show()
