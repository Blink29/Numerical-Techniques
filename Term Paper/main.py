import numpy as np
import matplotlib.pyplot as plt

# Input data
U = 1  # Velocity amplitude [cm/s]
T = 500  # Oscillation period [s]
T1 = 1000  # Second oscillation period [s]
v = 0.01  # Kinematic viscosity [cm^2/s]
v1 = 0.1  # Second kinematic viscosity [cm^2/s]
Dt = 1  # Time step [s]
Dz = 1  # Vertical step [cm]
nm = 19  # Number of vertical discretization layers
im = nm + 1  # Number of grid points

tm = 5000  # Number of time steps

# Initialization of the velocity fields
u1 = np.zeros(im)
u2 = np.zeros(im)
u3 = np.zeros(im)
un1 = np.zeros(im)
un2 = np.zeros(im)
un3 = np.zeros(im)

# Boundary condition at the fixed plate
u1[-1] = 0
u2[-1] = 0
u3[-1] = 0

# Arrays to store shear stress
tau1 = np.zeros(tm)
tau2 = np.zeros(tm)
tau3 = np.zeros(tm)

# Main program
for k in range(1, tm + 1):
    Dtt = Dt * k

    # Boundary condition at the oscillating plate
    u1[0] = U * np.sin(2 * np.pi * Dtt / T)
    u2[0] = U * np.sin(2 * np.pi * Dtt / T1)
    u3[0] = U * np.sin(2 * np.pi * Dtt / T)

    # Calculation of the horizontal velocities between the two plates
    for k1 in range(1, nm):
        un1[k1] = u1[k1] + (v * Dt / (Dz**2)) * (u1[k1 + 1] - 2 * u1[k1] + u1[k1 - 1])
        un2[k1] = u2[k1] + (v * Dt / (Dz**2)) * (u2[k1 + 1] - 2 * u2[k1] + u2[k1 - 1])
        un3[k1] = u3[k1] + (v1 * Dt / (Dz**2)) * (u3[k1 + 1] - 2 * u3[k1] + u3[k1 - 1])

    # Update velocity values
    u1[1:nm] = un1[1:nm]
    u2[1:nm] = un2[1:nm]
    u3[1:nm] = un3[1:nm]

    # Calculate shear stresses
    tau1[k - 1] = ((u1[0] - u1[1]) / Dz) * v
    tau2[k - 1] = ((u2[0] - u2[1]) / Dz) * v
    tau3[k - 1] = ((u3[0] - u3[1]) / Dz) * v1

# Plotting the results
plt.plot(range(1, tm + 1), tau1, 'r', linewidth=1.5, label='tau1: T=500s, v=0.01cm^2/s')
plt.plot(range(1, tm + 1), tau2, 'g', linewidth=1.5, label='tau2: T=1000s, v=0.01cm^2/s')
plt.plot(range(1, tm + 1), tau3, 'b', linewidth=1.5, label='tau3: T=500s, v=0.1cm^2/s')

plt.xlabel('Time steps')
plt.ylabel('Shear stress along the oscillating plate [(cm/s)^2]')
plt.axis([0, 5000, -0.049, 0.035])
plt.legend(loc='best')
plt.grid(True)
plt.title('Shear Stress for Oscillating Couette Flow')
plt.show()
