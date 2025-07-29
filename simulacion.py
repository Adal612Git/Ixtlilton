import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# physical parameters
m1 = 7.0  # thigh mass [kg]
m2 = 3.5  # shank mass [kg]
l1 = 0.4  # thigh length [m]
l2 = 0.4  # shank length [m]
a1 = l1/2
a2 = l2/2
I1 = m1 * l1**2 / 12
I2 = m2 * l2**2 / 12
g = 9.81

# PD gains
KP = 200.0
KD = 40.0

# reference trajectories for hip (theta1) and knee (theta2)
def refs(t, phase=0.0):
    w = 2 * np.pi  # 1 Hz gait
    q1 = 0.3 * np.sin(w * (t + phase))
    q2 = 0.7 * np.sin(w * (t + phase))
    dq1 = 0.3 * w * np.cos(w * (t + phase))
    dq2 = 0.7 * w * np.cos(w * (t + phase))
    return q1, q2, dq1, dq2

# dynamics of a single leg

def leg_dynamics(t, y):
    q1, q2, dq1, dq2 = y
    q1_ref, q2_ref, dq1_ref, dq2_ref = refs(t)
    tau1 = KP * (q1_ref - q1) + KD * (dq1_ref - dq1)
    tau2 = KP * (q2_ref - q2) + KD * (dq2_ref - dq2)
    M11 = I1 + I2 + m1 * a1**2 + m2 * (l1**2 + a2**2 + 2 * l1 * a2 * np.cos(q2))
    M12 = I2 + m2 * (a2**2 + l1 * a2 * np.cos(q2))
    M21 = M12
    M22 = I2 + m2 * a2**2
    C1 = -m2 * l1 * a2 * np.sin(q2) * (2 * dq1 * dq2 + dq2**2)
    C2 = m2 * l1 * a2 * np.sin(q2) * dq1**2
    G1 = (m1 * a1 + m2 * l1) * g * np.sin(q1) + m2 * a2 * g * np.sin(q1 + q2)
    G2 = m2 * a2 * g * np.sin(q1 + q2)
    M = np.array([[M11, M12], [M21, M22]])
    rhs = np.array([tau1 - C1 - G1, tau2 - C2 - G2])
    ddq1, ddq2 = np.linalg.solve(M, rhs)
    return [dq1, dq2, ddq1, ddq2]

# simulate one gait cycle (2 seconds)
t = np.linspace(0, 2, 400)
sol = solve_ivp(leg_dynamics, [t[0], t[-1]], [0, 0, 0, 0], t_eval=t)

# left leg angles
theta1 = sol.y[0]
theta2 = sol.y[1]
# right leg angles (phase shifted by half cycle)
theta1_r, theta2_r, _, _ = refs(t, phase=0.5)

# compute positions
x_hip = 0
y_hip = 0
xk = l1 * np.sin(theta1)
yk = l1 * np.cos(theta1)
xa = xk + l2 * np.sin(theta1 + theta2)
ya = yk + l2 * np.cos(theta1 + theta2)

xk_r = l1 * np.sin(theta1_r)
yk_r = l1 * np.cos(theta1_r)
xa_r = xk_r + l2 * np.sin(theta1_r + theta2_r)
ya_r = yk_r + l2 * np.cos(theta1_r + theta2_r)

fig, ax = plt.subplots()
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.1, 0.9)
ax.set_aspect('equal')

left_thigh, = ax.plot([], [], 'o-', lw=4, color='blue')
left_shank, = ax.plot([], [], 'o-', lw=4, color='blue')
right_thigh, = ax.plot([], [], 'o-', lw=4, color='red')
right_shank, = ax.plot([], [], 'o-', lw=4, color='red')

def init():
    left_thigh.set_data([], [])
    left_shank.set_data([], [])
    right_thigh.set_data([], [])
    right_shank.set_data([], [])
    return left_thigh, left_shank, right_thigh, right_shank


def update(frame):
    lt_x = [x_hip, xk[frame]]
    lt_y = [y_hip, yk[frame]]
    ls_x = [xk[frame], xa[frame]]
    ls_y = [yk[frame], ya[frame]]
    rt_x = [x_hip, xk_r[frame]]
    rt_y = [y_hip, yk_r[frame]]
    rs_x = [xk_r[frame], xa_r[frame]]
    rs_y = [yk_r[frame], ya_r[frame]]
    left_thigh.set_data(lt_x, lt_y)
    left_shank.set_data(ls_x, ls_y)
    right_thigh.set_data(rt_x, rt_y)
    right_shank.set_data(rs_x, rs_y)
    return left_thigh, left_shank, right_thigh, right_shank

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=25)
plt.title('Simulación de marcha 2D')
plt.show()
