# Vehicular Platoon - Cyclic Interconnection of 5 Systems
# =======================================================
# Each vehicle follows the one ahead in a ring topology:
# Vehicle 1 → Vehicle 2 → Vehicle 3 → Vehicle 4 → Vehicle 5 → Vehicle 1
#
# State: x_i = [position_i, velocity_i]
# Input: u_i = acceleration command
# Output: y_i = position (used by next vehicle)
#
# Each vehicle dynamics: double integrator
# x_dot = A*x + B*u
# y = C*x


# Sampling time for discrete system
Ts = 0.01

# Number of vehicles in platoon
N = 5

# ===== Individual Vehicle Dynamics (Continuous) =====
# State: [position, velocity]
# Double integrator: x_dot = [0 1; 0 0]*x + [0; 1]*u
A_cont = [[0, 1], 
          [0, 0]]

B_cont = [[0], 
          [1]]

C_cont = [[1, 0]]  # Output is position

D_cont = [[0]]

# ===== Local Controller Gains =====
# Each vehicle uses PD control: u = -Kp*(x_i - x_{i-1}) - Kd*(v_i - v_{i-1})
# Simplified: u = -K * (my_state - neighbor_state)

Kp = 1.0   # Position gain
Kd = 2.0   # Velocity/damping gain
K = [[Kp, Kd]]  # Controller gain matrix

# ===== Coupling Weight =====
# Cyclic adjacency: each vehicle is coupled to the one behind it
# eta = coupling strength (0 to 1)
eta = 0.8

# ===== Discrete-Time Matrices (ZOH discretization) =====
# For double integrator with Ts:
# Ad = [1, Ts; 0, 1]
# Bd = [0.5*Ts^2; Ts]
A_disc = [[1, Ts], 
          [0, 1]]

B_disc = [[0.5 * Ts**2], 
          [Ts]]

C_disc = [[1, 0]]
D_disc = [[0]]

# ===== Reference Signals =====
# Leader reference (vehicle 0 setpoint)
leader_pos = 10.0
leader_vel = 0.0

# Desired inter-vehicle spacing
spacing = 2.0

# Reference positions for each vehicle
ref_positions = [leader_pos - i * spacing for i in range(N)]
# ref_positions = [10.0, 8.0, 6.0, 4.0, 2.0]

# ===== Initial Conditions =====
# All vehicles start at position 0 with zero velocity
init_pos = [0.0, 0.0, 0.0, 0.0, 0.0]
init_vel = [0.0, 0.0, 0.0, 0.0, 0.0]

# ===== Reference for each vehicle =====
ref1 = 10.0
ref2 = 8.0
ref3 = 6.0
ref4 = 4.0
ref5 = 2.0

# ===== Print Summary =====
print(f"Platoon Configuration: {N} vehicles")
print(f"Coupling strength (eta): {eta}")
print(f"Controller gains: Kp={Kp}, Kd={Kd}")
print(f"Sampling time: {Ts} s")
