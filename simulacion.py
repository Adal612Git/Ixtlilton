import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# segment lengths (meters)
l_thigh = 0.4
l_shank = 0.4
l_foot = 0.2

# simplified gait angles (degrees -> radians)
hip_deg = [10, 5, 0, -5, -10, -5, 0, 5, 10]
knee_deg = [0, 10, 30, 50, 60, 50, 30, 10, 0]
ankle_deg = [-10, -5, 0, 5, 10, 5, 0, -5, -10]

hip_angles = np.radians(hip_deg)
knee_angles = np.radians(knee_deg)
ankle_angles = np.radians(ankle_deg)

phase = np.linspace(0, 1, len(hip_angles))
t = np.linspace(0, 1, 200)

# interpolate angles over the cycle
theta1 = np.interp(t, phase, hip_angles)
theta2 = np.interp(t, phase, knee_angles)
theta3 = np.interp(t, phase, ankle_angles)

# hip trajectory (simple horizontal translation)
hip_x = 0.3 * t
hip_y = 0.0

# compute joint positions (mirrored for natural orientation)
x_knee = hip_x - l_thigh * np.sin(theta1)
y_knee = hip_y - l_thigh * np.cos(theta1)

x_ankle = x_knee - l_shank * np.sin(theta1 + theta2)
y_ankle = y_knee - l_shank * np.cos(theta1 + theta2)

x_foot = x_ankle - l_foot * np.sin(theta1 + theta2 + theta3)
y_foot = y_ankle - l_foot * np.cos(theta1 + theta2 + theta3)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-0.1, 0.5)
ax.set_ylim(-1.0, 0.2)

thigh_line, = ax.plot([], [], 'o-', lw=4, color='blue')
shank_line, = ax.plot([], [], 'o-', lw=4, color='blue')
foot_line, = ax.plot([], [], 'o-', lw=4, color='blue')


def init():
    thigh_line.set_data([], [])
    shank_line.set_data([], [])
    foot_line.set_data([], [])
    return thigh_line, shank_line, foot_line


def update(frame):
    hx = hip_x[frame]
    hy = hip_y
    kx = x_knee[frame]
    ky = y_knee[frame]
    axp = x_ankle[frame]
    ay = y_ankle[frame]
    fx = x_foot[frame]
    fy = y_foot[frame]

    thigh_line.set_data([hx, kx], [hy, ky])
    shank_line.set_data([kx, axp], [ky, ay])
    foot_line.set_data([axp, fx], [ay, fy])
    return thigh_line, shank_line, foot_line


ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=20)
plt.title('Simulacion de marcha con rodilla y tobillo')
plt.show()
