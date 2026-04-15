import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 2000                 # Number of particles
T = 60                   # Total time (seconds)
h = 0.1                  # Time step
D = 0.02                 # Diffusion coefficient
u = np.array([0.3, 0.0]) # Advection velocity (x, y)
num_steps = int(T / h)

# All particles start at (0, 0)
particles = np.zeros((N, 2))

# Times to save snapshots
save_times = [15, 30, 45, 60]
snapshots = {}

# Euler-Maruyama simulation
for step in range(1, num_steps + 1):
    current_t = round(step * h, 2)

    # Random normal variables for all particles
    Z = np.random.normal(0, 1, size=(N, 2))

    # Update rule
    particles += u * h + np.sqrt(2 * D * h) * Z

    # Save particle positions at selected times
    if current_t in save_times:
        snapshots[current_t] = particles.copy()

# Plot snapshots
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, t in enumerate(save_times):
    pos = snapshots[t]
    axes[i].scatter(
    pos[:, 0], pos[:, 1],
    s=1,
    alpha=0.5,
    color="blue"
)
    axes[i].set_title(f"Time: {t} s")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")
    axes[i].set_xlim([0, 25])
    axes[i].set_ylim([-5, 5])
    axes[i].grid(True)
    

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Task 2(a): concentration field estimation
# --------------------------------------------------

epsilon = 0.1
nx, ny = 200, 100

# Uniform grid over Omega = [0,25] x [-5,5]
x_vals = np.linspace(0, 25, nx)
y_vals = np.linspace(-5, 5, ny)
Xg, Yg = np.meshgrid(x_vals, y_vals)

def gaussian_kernel_2d(dx, dy, epsilon):
    r2 = dx**2 + dy**2
    return (1.0 / (2.0 * np.pi * epsilon**2)) * np.exp(-r2 / (2.0 * epsilon**2))

def estimate_concentration(particle_positions, Xg, Yg, epsilon):

    # Estimate concentration field C(x,y,t) from particle positions
    
    N = particle_positions.shape[0]
    C = np.zeros_like(Xg)

    #sum the dirac delta funcion for all N.
    for k in range(N):
        xk, yk = particle_positions[k]
        dx = Xg - xk
        dy = Yg - yk
        C += gaussian_kernel_2d(dx, dy, epsilon)

    C /= N
    return C

# Compute concentration field for each saved time
concentration_fields = {}

for t in save_times:
    positions_t = snapshots[t]
    concentration_fields[t] = estimate_concentration(positions_t, Xg, Yg, epsilon)

# --------------------------------------------------
# Task 2(b): plot concentration fields with contourf
# --------------------------------------------------

# Use one common color scale for all plots
all_C_values = np.concatenate([concentration_fields[t].ravel() for t in save_times])
vmin = all_C_values.min()
vmax = all_C_values.max()

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
axes = axes.flatten()

contour_plot = None

for i, t in enumerate(save_times):
    C = concentration_fields[t]
    contour_plot = axes[i].contourf(Xg, Yg, C, levels=30, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[i].set_title(f"Concentration at t = {t} s")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")
    axes[i].set_xlim([0, 25])
    axes[i].set_ylim([-5, 5])

fig.colorbar(contour_plot, ax=axes, orientation="horizontal", fraction=0.06, pad=0.08, label="Concentration")
plt.tight_layout()
plt.show()