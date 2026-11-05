import numpy as np


class TwoLinkArm:
    def __init__(self):
        # Physical parameters
        self.m1, self.m2 = 1.0, 1.0          # Link masses (kg)
        self.l1, self.l2 = 1.0, 1.0          # Link lengths (m)
        self.lc1, self.lc2 = 0.5, 0.5        # Centers of mass (m)
        self.I1 = (1.0/12.0) * self.m1 * self.l1**2
        self.I2 = (1.0/12.0) * self.m2 * self.l2**2
        self.g = 9.81                        # Gravity (m/s^2)

        # Unmodeled true physical effects: Joint Friction & Damping
        self.damping = np.array([0.4, 0.4])   # Viscous damping
        self.coulomb = np.array([0.2, 0.2])   # Coulomb friction

    def inertia_matrix(self, q):
        q2 = q[1]
        m11 = (self.m1 * self.lc1**2 + self.I1 +
               self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * np.cos(q2)) + self.I2)
        m12 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * np.cos(q2)) + self.I2
        m21 = m12
        m22 = self.m2 * self.lc2**2 + self.I2
        return np.array([[m11, m12], [m21, m22]])

    def coriolis_vector(self, q, dq):
        """Coriolis & Centrifugal forces (quadratic in dq)."""
        h = self.m2 * self.l1 * self.lc2 * np.sin(q[1])
        c1 = -h * (2 * dq[0] * dq[1] + dq[1]**2)
        c2 = h * dq[0]**2
        return np.array([c1, c2])

    def gravity_vector(self, q):
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(q[0]) + \
             self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])
        g2 = self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])
        return np.array([g1, g2])

    def friction_vector(self, dq):
        """Viscous + smooth Coulomb friction."""
        return self.damping * dq + self.coulomb * np.tanh(dq * 10.0)

    # ------------------ True Forward Dynamics ------------------
    def forward_dynamics_true(self, x, u):
        """True physical dynamics: includes Coriolis, friction, gravity, inertia."""
        q, dq = x[:2], x[2:]
        M = self.inertia_matrix(q)
        C_dq = self.coriolis_vector(q, dq)
        G = self.gravity_vector(q)
        F = self.friction_vector(dq)

        # M(q) * ddq + C(q, dq)*dq + G(q) + F(dq) = u
        ddq = np.linalg.solve(M, u - C_dq - G - F)
        return np.concatenate([dq, ddq])

    # ----------------- Nominal Forward Dynamics ----------------
    def forward_dynamics_nom(self, x, u):
        """Nominal controller model: neglects Coriolis and joint friction."""
        q, dq = x[:2], x[2:]
        M = self.inertia_matrix(q)
        G = self.gravity_vector(q)

        # Nominal assumes: M(q) * ddq + G(q) = u (C=0, F=0)
        ddq_nom = np.linalg.solve(M, u - G)
        return np.concatenate([dq, ddq_nom])

def rk4_step(dynamics_func, x, u, dt):
    """Runge-Kutta 4th-order numerical integrator."""
    k1 = dynamics_func(x, u)
    k2 = dynamics_func(x + 0.5 * dt * k1, u)
    k3 = dynamics_func(x + 0.5 * dt * k2, u)
    k4 = dynamics_func(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def generate_controlled_trajectory_new(robot, duration=5.0, dt=0.01, regime='slow', A_custom=None, w_custom=None):
    t_steps = int(duration / dt)
    time = np.linspace(0, duration, t_steps)

    if regime == 'slow':
        w = 1.0 if w_custom is None else w_custom
        A = np.array([0.5, 0.5]) if A_custom is None else A_custom
    elif regime == 'aggressive':
        w = 4.5 if w_custom is None else w_custom
        A = np.array([1.2, 1.2]) if A_custom is None else A_custom

    states = np.zeros((t_steps, 4))
    inputs = np.zeros((t_steps, 2))
    res_ddq = np.zeros((t_steps, 2))

    # 1. Match initial state to the trajectory at t = 0 (No initial shock!)
    # q_des(0) = 0, dq_des(0) = A * w
    x = np.array([0.0, 0.0, A[0] * w, A[1] * w])

    # Critical damping gains for decoupled double-integrator: Kd = 2 * sqrt(Kp)
    kp_val = 100.0
    kd_val = 2.0 * np.sqrt(kp_val) # 20.0 (critically damped)
    Kp = np.diag([kp_val, kp_val])
    Kd = np.diag([kd_val, kd_val])

    for i in range(t_steps):
        t = time[i]

        # Reference trajectory
        q_des   = A * np.sin(w * t)
        dq_des  = A * w * np.cos(w * t)
        ddq_des = -A * (w**2) * np.sin(w * t)

        q, dq = x[:2], x[2:]
        e = q_des - q
        de = dq_des - dq

        # 2. COMPUTED TORQUE CONTROL (Inertia decoupled)
        M_nom = robot.inertia_matrix(q)
        G_nom = robot.gravity_vector(q)
        v = ddq_des + Kp @ e + Kd @ de
        u = M_nom @ v + G_nom

        # Instantaneous residual
        x_dot_true = robot.forward_dynamics_true(x, u)
        x_dot_nom = robot.forward_dynamics_nom(x, u)

        res_ddq[i] = x_dot_true[2:] - x_dot_nom[2:]
        states[i] = x
        inputs[i] = u

        # Step forward
        x = rk4_step(robot.forward_dynamics_true, x, u, dt)

    return states, inputs, res_ddq, time

def collect_multitraj_dataset(robot, mode='aggressive', dt=0.01):
    all_states, all_inputs, all_res = [], [], []

    # 3 aggressive trajectories with slightly different frequencies & amplitudes
    configs = [
        {'w': 4.0, 'A': np.array([1.0, 1.0])},
        {'w': 4.5, 'A': np.array([1.2, 1.2])},
        {'w': 5.0, 'A': np.array([1.1, 0.9])},
        {'w': 1.0, 'A': np.array([0.5, 0.5])},
        {'w': 1.2, 'A': np.array([0.55, 0.45])},
        {'w': 0.9, 'A': np.array([0.45, 0.55])}
    ]
    configs = configs[:3] if mode == 'aggressive' else configs[3:]
    # configs = [
    #     {'w': 5.0, 'A': np.array([1.4, 1.2])},
    #     {'w': 6.0, 'A': np.array([1.7, 1.5])},
    #     {'w': 5.5, 'A': np.array([1.6, 1.3])}
    # ]

    for cfg in configs:
        s, u, r, _ = generate_controlled_trajectory_new(robot, duration=3.0, dt=dt,
                                                    w_custom=cfg['w'], A_custom=cfg['A'])
        all_states.append(s)
        all_inputs.append(u)
        all_res.append(r)

    states = np.vstack(all_states)
    inputs = np.vstack(all_inputs)
    residuals = np.vstack(all_res)

    # Subsample 400 points uniformly across the tube
    idx = np.random.choice(len(states), size=600, replace=False)
    return states[idx], inputs[idx], residuals[idx]
