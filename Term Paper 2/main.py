# %%
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import math
k= 50 # W/mk
Lx, Ly = 1.0, 2.0  # Dimensions of the plate
Nx, Ny = 21, 41  # Grid size (21x41 grid)
tolerance = 0.01  # Convergence criterion
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
# dx = dy = 1/20 
beta_2 = (dx / dy) ** 2
observation_table = []
# Boundary conditions
T_bottom = 100.0  # Bottom boundary temperature (100 C)
T_other = 30.0  # Other boundaries' temperature (30 C)

# %%
def calculate_error(T, T_new):
    return np.sum(np.abs(T_new - T))

def point_gauss_seidel(T, max_iter=5000,w=1):
    T_new = T.copy()
    for iteration in range(max_iter):
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                T_new[j,i] =  (T_new[j-1,i]*beta_2 + T_new[j,i-1]*beta_2 +
                 T[j+1,i] + T[j,i+1]) / (2+2*beta_2)
                T_new[j,i] = T[j,i] + (T_new[j,i] - T[j,i])*w
        
        # Check for convergence
        error = calculate_error(T, T_new)
        if error < tolerance:
            print(f"Converged after {iteration} iterations with error {error:.5f}")
            return T_new, iteration
        
        T[:] = T_new  # Update the temperature grid
    print(f"Max iterations reached with error {error:.5f}")
    return T_new, max_iter

# %%
# Plotting the steady state temperature contour
def plotResult(T,name):
    plt.contourf(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny), T, levels=50, cmap='jet')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Steady State Temperature Contour {name}')
    plt.xlabel('x (units)')
    plt.ylabel('y (units)')
    plt.show()

    # Calculate heat transfer along the bottom boundary
    heat_transfer_bottom = -k * (T[1, :] - T[0, :]) / dy
    total_heat_transfer = np.sum(heat_transfer_bottom) * dx
    print(f'heat transfer along the bottom boundary of the plate : {total_heat_transfer}')

# %%
# grid init
T = np.ones((Ny,Nx)) * T_other
T[0, :] = T_bottom 
T_pgs, num_iterations_pgs = point_gauss_seidel(T)
plotResult(T_pgs,'Point Gauss Seidel')
observation_table.append({'Scheme Used':'Point Guass Seidel' , 'Grid Size':'42 x 21' , 
'No. of Iterations':num_iterations_pgs,'relaxation factor':1})

# %%
# grid init
T = np.ones((Ny,Nx)) * T_other
T[0, :] = T_bottom 
T_psor,num_iterations_psor= point_gauss_seidel(T,w=0.5) # adding relaxation of 0.5
plotResult(T_psor,'PSOR')
w = np.arange(0.1,2,0.1)

# %%
#Variation of number of iteration with relaxation factor. 
relaxation_factors = np.arange(0.1,2.1,0.1)
iterations = []
for w in relaxation_factors:
    #initializing boundary conditions
    T = np.ones((Ny,Nx)) * T_other
    T[0, :] = T_bottom 
    _,iter = point_gauss_seidel(T,w=w)
    iterations.append(iter)
    if(math.isclose(w, 0.5) or math.isclose(w, 1.5)):
        observation_table.append({'Scheme Used':'PSOR' , 'Grid Size':'42 x 21' , 
        'No. of Iterations':iter,'relaxation factor':w})

plt.plot(relaxation_factors,iterations)
plt.title('relaxation factor vs iterations needed (PSOR)')
plt.xlabel('w')
plt.ylabel('iterations')
plt.grid(True)
plt.show()

# %%
def line_gauss_seidel(T,max_iter=5000,w=1):
    for it in range(1, max_iter):
        T_old = T.copy()  # Copy the old temperature grid

        for i in range(1, Nx - 1):
            # Thomas Algorithm for tri-diagonal system
            alpha = np.ones(Ny)*-1
            beta = np.ones(Ny)*(2+2*beta_2)
            gamma = np.ones(Ny)*-1
            alpha[0]=0
            gamma[-1]=0
            b=beta_2 * (T_old[:, i + 1] + T[:, i - 1])
            b[0]+=T_bottom
            b[-1]+=T_other
            e = np.ones(Ny)
            f = np.ones(Ny)
            e[0]=gamma[0]/beta[0]
            f[0] = b[0]/beta[0]

            for j in range(1,Ny-1):
                d = (beta[j] - alpha[j]*e[j-1])
                e[j]=gamma[j]/d
                f[j] = (b[j]-alpha[j]*f[j-1])/d
            
            T[Ny-1,i] = (alpha[Ny-1]*f[Ny-2] - b[Ny-1]) / (alpha[Ny-1]*e[Ny-2] -
             beta[Ny-1])
            for j in range(Ny-2,-1,-1):
                T[j,i]=f[j] - e[j]*T[j+1,i]
            
        T = T_old + (T - T_old)*w
        # Calculate error and check for convergence
        error= calculate_error(T_old, T)
        if error <= tolerance:
            print(f"Converged after {it} iterations with error {error:.5f}")
            return T,it
    print(f"Max iterations reached with error {error:.5f}")
    return T,max_iter
    

# %%
# grid init
T = np.ones((Ny,Nx)) * T_other
T[0, :] = T_bottom 
T_lgs, num_iterations_lgs = line_gauss_seidel(T)
plotResult(T_lgs,"Line Guass Seidel")
observation_table.append({'Scheme Used':'Line Guass Seidel' , 'Grid Size':'42 x 21' ,
 'No. of Iterations':num_iterations_lgs,'relaxation factor':1})

# %%
# grid init
T = np.ones((Ny,Nx)) * T_other
T[0, :] = T_bottom 
T_lsor,num_iterations_lsor = line_gauss_seidel(T,w=0.5)
plotResult(T_lsor,"LSOR")


# %%
iterations = []
for w in relaxation_factors:
    #initializing boundary conditions
    T = np.ones((Ny,Nx)) * T_other
    T[0, :] = T_bottom 
    _,iter = line_gauss_seidel(T,w=w)
    iterations.append(iter)
    if(math.isclose(w, 0.5) or math.isclose(w, 1.5)):
        observation_table.append({'Scheme Used':'LSOR' , 'Grid Size':'42 x 21' ,
         'No. of Iterations':iter,'relaxation factor':w})

plt.plot(relaxation_factors,iterations)
plt.title('relaxation factor vs iterations needed LSOR')
plt.xlabel('w')
plt.ylabel('iterations')
plt.grid(True)
plt.show()

# %%
def gauss_seidel_with_adi(T,max_iter=10000,w=1):
    for it in range(1, max_iter):
        T_old = T.copy()  # Copy the old temperature grid

        for i in range(1, Nx - 1):
            # Thomas Algorithm for tri-diagonal system
            alpha = np.ones(Ny)*-1
            beta = np.ones(Ny)*(2+2*beta_2)
            gamma = np.ones(Ny)*-1
            alpha[0]=0
            gamma[-1]=0
            b=beta_2 * (T_old[:, i + 1] + T[:, i - 1])
            b[0]+=T_bottom
            b[-1]+=T_other
            e = np.ones(Ny)
            f = np.ones(Ny)
            e[0]=gamma[0]/beta[0]
            f[0] = b[0]/beta[0]

            for j in range(1,Ny-1):
                d = (beta[j] - alpha[j]*e[j-1])
                e[j]=gamma[j]/d
                f[j] = (b[j]-alpha[j]*f[j-1])/d
            
            T[Ny-1,i] = (alpha[Ny-1]*f[Ny-2] - b[Ny-1]) / (alpha[Ny-1]*e[Ny-2] - 
            beta[Ny-1])
            for j in range(Ny-2,-1,-1):
                T[j,i]=f[j] - e[j]*T[j+1,i]
    
        T_old = T.copy() 

        for j in range(1, Ny - 1):
            # Thomas Algorithm for tri-diagonal system
            alpha = np.ones(Nx)*-1*beta_2
            beta = np.ones(Nx)*(2+2*beta_2)
            gamma = np.ones(Nx)*-1*beta_2
            alpha[0]=0
            gamma[-1]=0
            b= (T_old[j+1,:] + T[j-1,:])
            b[0]+=T_other
            b[-1]+=T_other
            e = np.ones(Nx)
            f = np.ones(Nx)
            e[0]=gamma[0]/beta[0]
            f[0] = b[0]/beta[0]

            for i in range(1,Nx-1):
                d = (beta[i] - alpha[i]*e[i-1])
                e[i]=gamma[i]/d
                f[i] = (b[i]-alpha[i]*f[i-1])/d
            
            T[j,Nx-1] = (alpha[Nx-1]*f[Nx-2] - b[Nx-1]) / (alpha[Nx-1]*e[Nx-2] - 
            beta[Nx-1])
            for i in range(Nx-2,-1,-1):
                T[j,i]=f[i] - e[i]*T[j,i+1]

            
        T = T_old + (T - T_old)*w
        # Calculate error and check for convergence
        error= calculate_error(T_old, T)
        if error <= tolerance:
            print(f"Converged after {it} iterations with error {error:.5f}")
            return T,it
    print(f"Max iterations reached with error {error:.5f}")
    return T,max_iter
    

# %%
# grid init
T = np.ones((Ny,Nx)) * T_other
T[0, :] = T_bottom 
T_adi,num_iterations_adi = gauss_seidel_with_adi(T)
plotResult(T_adi,"Gauss Seidel with ADI")
observation_table.append({'Scheme Used':'Gauss Seidel with ADI' , 'Grid Size':'42 x 21' ,
 'No. of Iterations':num_iterations_adi,'relaxation factor':1})

# %%
# grid init
T = np.ones((Ny,Nx)) * T_other
T[0, :] = T_bottom 
T_adiw,num_iterations_adiw = gauss_seidel_with_adi(T,w=0.5)
plotResult(T_adiw,"Gauss seidel with ADI and relaxation")

# %%
#Variation of number of iteration with relaxation factor. 
relaxation_factors = np.arange(0.1,1.9,0.1)
iterations = []
for w in relaxation_factors:
    #initializing boundary conditions
    T = np.ones((Ny,Nx)) * T_other
    T[0, :] = T_bottom 
    _,iter = gauss_seidel_with_adi(T,w=w)
    iterations.append(iter)
    if(math.isclose(w, 0.5) or math.isclose(w, 1.5)):
        observation_table.append({'Scheme Used':'ADI with relaxation' , 
        'Grid Size':'42 x 21' , 'No. of Iterations':iter,'relaxation factor':w})

plt.plot(relaxation_factors,iterations)
plt.title('relaxation factor vs iterations needed ADI')
plt.xlabel('w')
plt.ylabel('iterations')
plt.grid(True)
plt.show()

# %%
table = pd.DataFrame(observation_table)
print(tabulate(table, headers='keys', tablefmt='grid'))