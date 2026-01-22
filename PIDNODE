import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.01
t_end = 20.0
t_values = np.arange(0, t_end, dt)

# PID gains
kp_v = 10.0
kp_w = 15.0
ki_w = 0.05
kd_w = 0.2

# Circular reference trajectory
R = 2.0
omega_d = 1.0

# Velocity limits (important for stability)
v_max = 2.0
omega_max = 3.0

def rk4_step(f, t, state, dt):
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt/2 * k1)
    k3 = f(t + dt/2, state + dt/2 * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)



# PID memory
int_alpha = 0.0
prev_alpha = 0.0

def true_wmr_dynamics(t, state):
    global int_alpha, prev_alpha

    x, y, theta = state

    # Reference trajectory
    x_d = R * np.cos(omega_d * t)
    y_d = R * np.sin(omega_d * t)

    # Errors
    e_d = np.hypot(x_d - x, y_d - y)    # e_d = np.sqrt((x_d - x)**2 + (y_d - y)**2)
    alpha = np.arctan2(y_d - y, x_d - x) - theta
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    # PID (angular)
    int_alpha += alpha * dt
    d_alpha = (alpha - prev_alpha) / dt
    prev_alpha = alpha

    omega = kp_w * alpha + ki_w * int_alpha + kd_w * d_alpha
    v = kp_v * e_d

    # Saturation
    v = np.clip(v, 0.0, v_max)
    omega = np.clip(omega, -omega_max, omega_max)

    # Dynamics
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dtheta = omega
    
    return np.array([dx, dy, dtheta])

# Generate Ground-Truth Trajectories (RK4)
state = np.array([0.0, 0.5, 0.0])
true_states = np.zeros((len(t_values), 3))

# Simulation Loop
# ========================================
omega_values = np.zeros(len(t_values))
vel_values = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    true_states[i] = state   
    state = rk4_step(true_wmr_dynamics, t, state, dt)


t_torch = torch.tensor(t_values, dtype=torch.float32)
true_traj = torch.tensor(true_states, dtype=torch.float32)
state0 = true_traj[0]

# This NN learns closed-loop kinematics:
class NeuralWMR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 192),
            nn.Tanh(),
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 3)
        )

    #def forward(self, t, state):
    #    t_input = t.expand(state.shape[0], 1)
    #    return self.net(torch.cat([state, t_input], dim=1))

    def forward(self, t, state):
        return self.net(state)


func = NeuralWMR()
optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)

epochs = 10000
loss_val = []

for epoch in range(epochs):
    pred_traj = odeint(func, state0, t_torch)
    loss = torch.mean((pred_traj - true_traj) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val.append(loss.item())

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")


with torch.no_grad():
    learned_traj = odeint(func, state0, t_torch).numpy()

# Plot results
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(t_values, vel_values, 'k', label="Linear Velocity v(t)")
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity (m/s)')
plt.legend()
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(t_values, omega_values, 'k', label="angular Velocity (t)")
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('angular Velocity (rad/s)')
plt.legend()
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(t_values, true_states[:,0], 'k', label="x (m)")
plt.plot(t_values, learned_traj[:,0], 'r', label="Neural ODE")
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel("x (m)")
plt.legend()
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(t_values, true_states[:,1], 'k', label="y (m)")
plt.plot(t_values, learned_traj[:,1], 'r', label="Neural ODE")
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel("y (m)")
plt.legend()
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(loss_val, 'r', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Function")
plt.grid()
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(true_states[:,0], true_states[:,1], 'k--', label="RK4 Ground Truth")
plt.plot(learned_traj[:,0], learned_traj[:,1], 'r', label="Neural ODE")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.title("Closed-loop WMR Trajectory Learning")
plt.grid()
plt.show()
