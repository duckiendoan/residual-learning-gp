import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class GPResidualModel:
    """Multi-output GP wrapper (one scalar GP per joint acceleration)."""
    def __init__(self, feature_type='q_dq'):
        self.feature_type = feature_type
        # SE kernel + small observation noise kernel
        constant_ker = ConstantKernel(1.0, (1e-3, 1e2))
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + \
                 WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-4, 1e-1))
        self.gp1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4, random_state=42)
        self.gp2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4, random_state=42)
        self.x_scaler = StandardScaler()
        # self.y_scaler = StandardScaler()

    def _extract_features(self, states, inputs):
        q = states[:, :2]
        dq = states[:, 2:]
        if self.feature_type == 'q':
            return q
        elif self.feature_type == 'q_dq':
            return np.hstack([q, dq])
        elif self.feature_type == 'q_dq_u':
            return np.hstack([q, dq, inputs])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, states, inputs, residuals):
        X = self._extract_features(states, inputs)
        X = self.x_scaler.fit_transform(X)
        # self.y_scaler.fit(residuals)
        # residuals = self.y_scaler.transform(residuals)
        print(f"Training GP [{self.feature_type}] on {X.shape[1]} input dimensions...")
        self.gp1.fit(X, residuals[:, 0])
        self.gp2.fit(X, residuals[:, 1])
        print(f"GP1: {self.gp1.kernel_}")
        print(f"GP2: {self.gp2.kernel_}")

    def predict(self, states, inputs, return_std=False):
        X = self._extract_features(states, inputs)
        X = self.x_scaler.transform(X)
        mean1, std1 = self.gp1.predict(X, return_std=True)
        mean2, std2 = self.gp2.predict(X, return_std=True)
        mean = np.column_stack([mean1, mean2])
        std = np.column_stack([std1, std2])
        # mean = self.y_scaler.inverse_transform(mean)
        # std = self.y_scaler.inverse_transform(std)
        # if return_std:
        #     return mean, std
        return (mean, std) if return_std else mean

def physics_basis(dq):
    """
    Computes linear and quadratic velocity features:
    [dq1, dq2, dq1^2, dq2^2, dq1*dq2]
    """
    dq1 = dq[:, 0:1]
    dq2 = dq[:, 1:2]
    return np.hstack([
        dq1, dq2,           # Linear damping / viscous terms
        dq1**2, dq2**2,     # Centrifugal terms
        dq1 * dq2           # Coriolis cross-coupling
    ])

# ==============================================================================
# 2. Physics-Informed (Gray-Box) GP Model
# ==============================================================================
class PhysicsInformedGP:
    def __init__(self):
        # Parametric Physics Base Model (Ridge regression for stability)
        self.phys_model1 = Ridge(alpha=1e-3, fit_intercept=False)
        self.phys_model2 = Ridge(alpha=1e-3, fit_intercept=False)

        # Non-parametric GP for residual of the physics model
        self.scaler = StandardScaler()
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + \
                 WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.0))
        self.gp1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
        self.gp2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)

    def fit(self, states, residuals):
        q = states[:, :2]
        dq = states[:, 2:]

        # Step 1: Fit parametric physics model on velocity monomials
        Phi = physics_basis(dq)
        self.phys_model1.fit(Phi, residuals[:, 0])
        self.phys_model2.fit(Phi, residuals[:, 1])

        # Step 2: Compute remaining modeling error (residual-of-the-residual)
        m_phys1 = self.phys_model1.predict(Phi)
        m_phys2 = self.phys_model2.predict(Phi)
        delta_res1 = residuals[:, 0] - m_phys1
        delta_res2 = residuals[:, 1] - m_phys2

        # Step 3: Train GP on what physics couldn't explain
        X_scaled = self.scaler.fit_transform(states)
        self.gp1.fit(X_scaled, delta_res1)
        self.gp2.fit(X_scaled, delta_res2)

    def predict(self, states, return_std=False):
        dq = states[:, 2:]
        Phi = physics_basis(dq)

        # Asymptotic physics extrapolation
        m_phys1 = self.phys_model1.predict(Phi)
        m_phys2 = self.phys_model2.predict(Phi)

        # Local GP correction
        X_scaled = self.scaler.transform(states)
        gp_mu1, std1 = self.gp1.predict(X_scaled, return_std=True)
        gp_mu2, std2 = self.gp2.predict(X_scaled, return_std=True)

        mu_total = np.column_stack([m_phys1 + gp_mu1, m_phys2 + gp_mu2])
        std_total = np.column_stack([std1, std2])

        return (mu_total, std_total) if return_std else mu_total

class RobustPhysicsGP:
    def __init__(self, robot):
        self.robot = robot
        # Light Ridge regularization: won't zero out small Coriolis signals!
        self.phys_model1 = Ridge(alpha=1e-4, fit_intercept=False)
        self.phys_model2 = Ridge(alpha=1e-4, fit_intercept=False)

        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.5, 5.0)) + \
                 WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-3, 0.1))
        self.gp1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
        self.gp2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
        self.scaler = StandardScaler()

    def _torque_features(self, states):
        q = states[:, :2]
        dq = states[:, 2:]
        s2 = np.sin(q[:, 1:2])

        # Exact Euler-Lagrange Coriolis structures:
        # Joint 1 Coriolis: -h * (2*dq1*dq2 + dq2^2)
        c1 = -s2 * (2 * dq[:, 0:1] * dq[:, 1:2] + dq[:, 1:2]**2)
        # Joint 2 Coriolis: h * dq1^2
        c2 = s2 * dq[:, 0:1]**2

        f_coulomb = np.tanh(dq * 10.0)

        # Joint 1 uses [dq1, tanh1, c1]
        # Joint 2 uses [dq2, tanh2, c2]
        phi1 = np.hstack([dq[:, 0:1], f_coulomb[:, 0:1], c1])
        phi2 = np.hstack([dq[:, 1:2], f_coulomb[:, 1:2], c2])
        return phi1, phi2

    def fit(self, states, res_ddq):
        # 1. Convert acceleration residuals to TORQUE domain: tau = M * ddq
        tau_res = np.zeros_like(res_ddq)
        for i in range(len(states)):
            M = self.robot.inertia_matrix(states[i, :2])
            tau_res[i] = M @ res_ddq[i]

        # 2. Fit regularized physics per joint
        phi1, phi2 = self._torque_features(states)
        self.phys_model1.fit(phi1, tau_res[:, 0])
        self.phys_model2.fit(phi2, tau_res[:, 1])

        print("\nJoint 1 Learned [dq1, tanh1, c1]:", np.round(self.phys_model1.coef_, 4))
        print("Joint 2 Learned [dq2, tanh2, c2]:", np.round(self.phys_model2.coef_, 4))

        # 3. GP on the remainder
        m_tau1 = self.phys_model1.predict(phi1)
        m_tau2 = self.phys_model2.predict(phi2)
        delta_tau = tau_res - np.column_stack([m_tau1, m_tau2])

        X_scaled = self.scaler.fit_transform(states)
        self.gp1.fit(X_scaled, delta_tau[:, 0])
        self.gp2.fit(X_scaled, delta_tau[:, 1])

    def predict(self, states):
        phi1, phi2 = self._torque_features(states)
        tau_phys1 = self.phys_model1.predict(phi1)
        tau_phys2 = self.phys_model2.predict(phi2)

        X_scaled = self.scaler.transform(states)
        gp_tau1 = self.gp1.predict(X_scaled)
        gp_tau2 = self.gp2.predict(X_scaled)

        total_tau_res = np.column_stack([tau_phys1 + gp_tau1, tau_phys2 + gp_tau2])

        pred_ddq = np.zeros_like(states[:, :2])
        for i in range(len(states)):
            M = self.robot.inertia_matrix(states[i, :2])
            pred_ddq[i] = np.linalg.solve(M, total_tau_res[i])
        return pred_ddq
