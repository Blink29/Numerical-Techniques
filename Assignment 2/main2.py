import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.0001
L = 1.0
N = 10  # Number of intervals
dx = L / N
Q = np.zeros(N+1)
Q[int(0.25 * N)] = 1
Q[int(0.75 * N)] = -1

# Discretize the domain
x = np.linspace(0, L, N+1)

# Coefficient matrix A and right-hand side vector b
A = np.zeros((N+1, N+1))
b = np.zeros(N+1)

# Fill the coefficient matrix A and vector b
for i in range(1, N):
    A[i, i-1] = alpha / dx**2
    A[i, i] = -2 * alpha / dx**2
    A[i, i+1] = alpha / dx**2
    b[i] = -Q[i]

# Natural boundary conditions (dT/dx = 0 at x=0 and x=1)
A[0, 0] = 1
A[N, N] = 1
b[0] = 0
b[N] = 0

# Direct method using numpy's linear solver
T_direct = np.linalg.solve(A, b)

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

T_gs = gauss_seidel(A, b)

# Plot the results
plt.plot(x, T_direct, label="Direct Method")
plt.plot(x, T_gs, label="Gauss-Seidel", linestyle='--')
plt.xlabel("x")
plt.ylabel("Temperature (T)")
plt.title("Temperature Distribution")
plt.legend()
plt.grid(True)

plt.savefig("temperature_distribution.png")

plt.show()

print("Direct Method Solution:", T_direct)
print("Gauss-Seidel Method Solution:", T_gs)
