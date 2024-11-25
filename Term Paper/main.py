import numpy as np
import matplotlib.pyplot as plt

# Input data
U = 1 # Velocity amplitude [cm/s]
T = 500 # Oscillation period [s]
T1 = 1000 # Second oscillation period [s]
v = 0.01 # Kinematic viscosity [cm^2/s]
v1 = 0.1 # Second kinematic viscosity [cm^2/s]
epsilon = 0.1 # Non-Newtonian fluid constant
Dz = 2 # Vertical step [cm]
Dt = 0.5 # Time step [s]
nm = 19 # Number of vertical discretization layers
im = nm + 1 # Number of grid points
tm = 5000 # Number of time steps

# Initialization
u1 = np.zeros(im)
u2 = np.zeros(im)
u3 = np.zeros(im)
tau1 = []
tau2 = []
tau3 = []
half_shear = []
velocity_distributions = []

# Main simulation
for k in range(1, tm + 1):
    Dtt = Dt * k
    
    # Boundary conditions
    u1[0] = U * np.sin(2 * np.pi * Dtt / T)
    u2[0] = U * np.sin(2 * np.pi * Dtt / T1)
    u3[0] = U * np.sin(2 * np.pi * Dtt / T)
    u3[im-1] = 0  # Upper plate stationary for u3
    
    # Velocity calculation
    for k1 in range(1, nm):
        u1[k1] = u1[k1] + (v * Dt / Dz**2) * (u1[k1 + 1] - 2 * u1[k1] + u1[k1 - 1])
        u2[k1] = u2[k1] + (v * Dt / Dz**2) * (u2[k1 + 1] - 2 * u2[k1] + u2[k1 - 1])
        u3[k1] = u3[k1] + (v1 * Dt / Dz**2) * (u3[k1 + 1] - 2 * u3[k1] + u3[k1 - 1])
    
    # Shear stress calculations
    tau1.append((v * (u1[0] - u1[1]) / Dz))
    tau2.append((v * (u2[0] - u2[1]) / Dz))
    tau3.append((v1 * (u3[0] - u3[1]) / Dz))
    
    # Halfway shear stress calculation
    half_shear.append((v1 * (u3[nm // 2] - u3[nm // 2 + 1]) / Dz))
    
    # Store velocity profiles for every 100 time steps between 3000 and 4000
    if 3000 <= k <= 4000 and k % 100 == 0:
        velocity_distributions.append(u3.copy())

# Non-Newtonian fluid shear stress
non_newtonian_tau = []
t = np.linspace(0, T, tm)

for k in range(tm):
    # Plate velocity at time t
    plate_velocity = U * np.sin(2 * np.pi * t[k] / T)
    
    # Velocity gradient near the wall (difference between plate and adjacent fluid)
    velocity_gradient = (plate_velocity - 0) / Dz  # Assuming fluid at rest initially
    
    # Non-Newtonian shear stress
    tau = epsilon * (velocity_gradient)**2
    non_newtonian_tau.append(tau)

# Upper Plate Scenarios:
# (a) Half the velocity, in phase
upper_u_in_phase = [0.5 * U * np.sin(2 * np.pi * Dt * k / T) for k in range(1, tm + 1)]
# (b) Half the velocity, out of phase (180-degree phase shift)
upper_u_out_phase = [-0.5 * U * np.sin(2 * np.pi * Dt * k / T) for k in range(1, tm + 1)]

lower_plate_tau_in_phase = [(0.05 * (U * np.sin(2 * np.pi * Dt * k / T) - upper_u_in_phase[k - 1]) / Dz) for k in range(1, tm + 1)]
lower_plate_tau_out_phase = [(0.05 * (U * np.sin(2 * np.pi * Dt * k / T) - upper_u_out_phase[k - 1]) / Dz) for k in range(1, tm + 1)]

# Plotting results
plt.figure(figsize=(12, 8))

# Shear stress adjacent to the oscillating plate
plt.subplot(3, 1, 1)
plt.plot(range(tm), tau1, 'r', label='tau1: T=500s, v=0.01 cm²/s')
plt.plot(range(tm), tau2, 'g', label='tau2: T=1000s, v=0.01 cm²/s')
plt.plot(range(tm), tau3, 'b', label='tau3: T=500s, v=0.1 cm²/s')
plt.xlabel('Time steps')
plt.ylabel('Shear stress [(cm/s)^2]')
plt.legend()

# Halfway shear stress
plt.subplot(3, 1, 2)
plt.plot(range(tm), half_shear, 'c', label='Halfway shear stress: T=500s, v=0.1 cm²/s')
plt.xlabel('Time steps')
plt.ylabel('Shear stress [(cm/s)^2]')
plt.legend()

# Non-Newtonian fluid shear stress
plt.subplot(3, 1, 3)
plt.plot(range(tm), non_newtonian_tau, 'm', label='Non-Newtonian tau: T=500s, v=0.1 cm²/s')
plt.xlabel('Time steps')
plt.ylabel('Shear stress [(cm/s)^2]')
plt.legend()

plt.tight_layout()
plt.show()

# Plot velocity profiles
for i, profile in enumerate(velocity_distributions, start=1):
    plt.plot(profile, label=f'Time step: {3000 + i * 100}')
plt.xlabel('Vertical layers')
plt.ylabel('Velocity [cm/s]')
plt.title('Velocity Profiles')
plt.legend()
plt.show()

# Lower plate stresses
plt.plot(range(tm), lower_plate_tau_in_phase, 'y', label='In-phase')
plt.plot(range(tm), lower_plate_tau_out_phase, 'k', label='Out-of-phase')
plt.xlabel('Time steps')
plt.ylabel('Shear stress [(cm/s)^2]')
plt.legend()
plt.title('Shear stress on lower plate')
plt.show()


