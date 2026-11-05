import matplotlib.pyplot as plt
import numpy as np

from models import GPResidualModel, PhysicsInformedGP, RobustPhysicsGP
from simulation import *

np.random.seed(42)
dt = 0.01

def interpolation_exp(mode='aggressive'):
    robot = TwoLinkArm()

    print("=== Step 1: Generating Datasets ===")

    X_tr_s, U_tr, Y_tr = collect_multitraj_dataset(robot, dt=dt)

    # 2. Test data: aggressive high-speed trajectory
    X_test_state, U_test, Y_test_res, t_test = generate_controlled_trajectory_new(robot, duration=4.0, dt=dt, regime='aggressive')

    print("\n=== Step 2: Training 3 GP Residual Models ===")
    models = {
        'GP(q)': GPResidualModel(feature_type='q'),
        'GP(q, dq)': GPResidualModel(feature_type='q_dq'),
        'GP(q, dq, u)': GPResidualModel(feature_type='q_dq_u')
    }

    for name, model in models.items():
        model.fit(X_tr_s, U_tr, Y_tr)

    print("\n=== Step 3: One-Step Prediction Errors on Aggressive Test Regime ===")
    for name, model in models.items():
        preds = model.predict(X_test_state, U_test)
        rmse = np.sqrt(np.mean((preds - Y_test_res)**2))
        print(f"[{name}] 1-Step RMSE on ddq: {rmse:.4f} rad/s^2")
    print(f"[Nominal] 1-Step RMSE on ddq: {np.sqrt(np.mean((-Y_test_res)**2))}")

    print("\n=== Step 4: Multi-step evaluation ===")
    # Horizon of 30 steps (0.3 seconds)
    H = 30
    rollout_len = len(t_test)
    x0 = X_test_state[0].copy()
    pred_errors = {'Nominal': [], 'GP(q)': [], 'GP(q, dq)': []}

    for t_start in range(0, rollout_len - H, H):
        x_true_init = X_test_state[t_start]

        # State accumulators for this horizon
        x_sim_nom = x_true_init.copy()
        x_sim_q = x_true_init.copy()
        x_sim_q_dq = x_true_init.copy()

        for h in range(H):
            idx = t_start + h
            u_h = U_test[idx]

            # 1. Step Nominal Model
            x_sim_nom = rk4_step(robot.forward_dynamics_nom, x_sim_nom, u_h, dt)

            # 2. Step GP(q) Model
            mu_q = models['GP(q)'].predict(x_sim_q[None, :], u_h[None, :])[0]
            x_dot_q = robot.forward_dynamics_nom(x_sim_q, u_h)
            x_dot_q[2:] += mu_q
            x_sim_q += dt * x_dot_q

            # 3. Step GP(q, dq) Model
            mu_q_dq = models['GP(q, dq)'].predict(x_sim_q_dq[None, :], u_h[None, :])[0]
            x_dot_q_dq = robot.forward_dynamics_nom(x_sim_q_dq, u_h)
            x_dot_q_dq[2:] += mu_q_dq
            x_sim_q_dq += dt * x_dot_q_dq

        # Measure error against the TRUE robot at the end of the horizon
        x_true_end = X_test_state[t_start + H]
        pred_errors['Nominal'].append(np.linalg.norm(x_sim_nom[:2] - x_true_end[:2]))
        pred_errors['GP(q)'].append(np.linalg.norm(x_sim_q[:2] - x_true_end[:2]))
        pred_errors['GP(q, dq)'].append(np.linalg.norm(x_sim_q_dq[:2] - x_true_end[:2]))

    print(f"Mean {H*dt}s Rollout Drift:")
    print(f"Nominal:   {np.mean(pred_errors['Nominal']):.4f} rad")
    print(f"GP(q):     {np.mean(pred_errors['GP(q)']):.4f} rad")
    print(f"GP(q, dq): {np.mean(pred_errors['GP(q, dq)']):.4f} rad")

    print("\n=== Step 4: Closed-loop tracking ===")
    if mode == 'aggressive':
        w = 4.5
        A = np.array([1.2, 1.2])
    else:
        w = 1.0
        A = np.array([0.5, 0.5])

    Kp = np.diag([25.0, 25.0])   # Moderate gains
    Kd = np.diag([5.0, 5.0])

    controllers = ['Nominal_Controller', 'GP(q)_Controller', 'GP(q, dq)_Controller']
    tracking_results = {c: np.zeros((rollout_len, 4)) for c in controllers}
    std_gp_q = np.zeros((rollout_len, 2))
    std_gp_q_dq = np.zeros((rollout_len, 2))

    for c_name in controllers:
        x = x0.copy()
        tracking_results[c_name][0] = x

        for i in range(rollout_len - 1):
            t = t_test[i]
            q_des = A * np.sin(w * t)
            dq_des = A * w * np.cos(w * t)
            ddq_des = -A * (w**2) * np.sin(w * t)

            q, dq = x[:2], x[2:]
            e = q_des - q
            de = dq_des - dq

            # 1. Base nominal torque
            M_nom = robot.inertia_matrix(q)
            G_nom = robot.gravity_vector(q)

            # 2. Add GP residual compensation (if using GP)
            # We want: ddq_nom + mu_res = ddq_des + Kp*e + Kd*de
            # So: ddq_nom = ddq_des + Kp*e + Kd*de - mu_res
            v_cmd = ddq_des + Kp @ e + Kd @ de

            if c_name == 'Nominal_Controller':
                u = M_nom @ v_cmd + G_nom
            elif c_name == 'GP(q)_Controller':
                mu_res, std_res = models['GP(q)'].predict(x[None, :], np.zeros((1, 2)), return_std=True)
                u = M_nom @ (v_cmd - mu_res[0]) + G_nom
                std_gp_q[i+1] = std_res[0]
            elif c_name == 'GP(q, dq)_Controller':
                mu_res, std_res = models['GP(q, dq)'].predict(x[None, :], np.zeros((1, 2)), return_std=True)
                u = M_nom @ (v_cmd - mu_res[0]) + G_nom
                std_gp_q_dq[i+1] = std_res[0]

            # STEP THE TRUE PHYSICAL ROBOT
            x = rk4_step(robot.forward_dynamics_true, x, u, dt)
            tracking_results[c_name][i+1] = x

    # Plot Tracking Errors ||q_des - q_true||
    plt.figure(figsize=(10, 4))
    for c_name in controllers:
        q_err = np.linalg.norm(tracking_results[c_name][:, :2] - (A * np.sin(w * t_test[:, None])), axis=1)
        plt.plot(t_test, q_err, label=f"{c_name} (Mean Err: {np.mean(q_err):.4f} rad)")

    plt.ylabel("Tracking Error ||q - q_des|| (rad)")
    plt.xlabel("Time (s)")
    plt.title("Closed-Loop Tracking Performance on the TRUE Robot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("close_loop_tracking_errors_int.pdf")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_test, std_gp_q[:, 1], label=r'Uncertainty $\sigma$ [GP(q)]')
    plt.plot(t_test, std_gp_q_dq[:, 1], label=r'Uncertainty $\sigma$ [GP(q, dq)]')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Residual GP $\sigma$ (rad/s$^2$)')
    plt.title('Posterior Uncertainty along Trajectory')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig("close_loop_tracking_errors_std.pdf")
    plt.tight_layout()

def extrapolation_exp_fail():
    robot = TwoLinkArm()
    # 1. Generate Training Data in the SLOW regime (|dq| <= 1.5 rad/s)
    X_train, U_train, Y_train, _ = generate_controlled_trajectory_new(robot, duration=5.0, dt=0.01, regime='slow')
    # Subsample 200 points
    idx = np.random.choice(len(X_train), size=200, replace=False)
    X_tr, Y_tr = X_train[idx], Y_train[idx]

    # 2. Generate Test Data in the AGGRESSIVE regime (|dq| up to 10 rad/s)
    X_test, U_test, Y_test, t_test = generate_controlled_trajectory_new(robot, duration=4.0, dt=0.01, regime='aggressive')

    # 3. Train Standard Black-Box GP
    standard_gp = GPResidualModel(feature_type='q_dq')
    standard_gp.fit(X_tr, U_train[idx], Y_tr)

    # 4. Train Physics-Informed (Gray-Box) GP
    physics_gp = PhysicsInformedGP()
    physics_gp.fit(X_tr, Y_tr)

    # 5. Evaluate One-Step Prediction on the AGGRESSIVE Test Set
    pred_std = standard_gp.predict(X_test, U_test)
    pred_phys = physics_gp.predict(X_test)

    rmse_std = np.sqrt(np.mean((pred_std - Y_test)**2))
    rmse_phys = np.sqrt(np.mean((pred_phys - Y_test)**2))
    rmse_nom = np.sqrt(np.mean((0.0 - Y_test)**2)) # Baseline: pure nominal model (predicting 0)

    print("=" * 60)
    print(f"EXTRAPOLATION TEST (Trained on Slow -> Tested on Aggressive)")
    print("=" * 60)
    print(f"Nominal Model (Zero Residual) RMSE : {rmse_nom:.4f} rad/s^2")
    print(f"Standard GP (Black-Box) RMSE      : {rmse_std:.4f} rad/s^2")
    print(f"Physics-Informed GP (Gray-Box) RMSE: {rmse_phys:.4f} rad/s^2")
    print("=" * 60)

def extrapolation_exp_successful():
    robot = TwoLinkArm()
    # 1. Generate Training Data in the SLOW regime (|dq| <= 1.5 rad/s)
    X_train, U_train, Y_train, _ = generate_controlled_trajectory_new(robot, duration=5.0, dt=0.01, regime='slow')
    # Subsample 200 points
    idx = np.random.choice(len(X_train), size=200, replace=False)
    X_tr, Y_tr = X_train[idx], Y_train[idx]

    # 2. Generate Test Data in the AGGRESSIVE regime (|dq| up to 10 rad/s)
    w_fast = 9.0
    A_fast = np.array([1.5, 1.5])
    X_test, U_test, Y_test, t_test = generate_controlled_trajectory_new(robot, duration=4.0, dt=0.01, regime='aggressive')
    print(f"train_max_velo={np.max(np.abs(X_tr[:, 2]))}", f"test_max_velo={np.max(np.abs(X_test[:, 2]))}")
    # 3. Train Standard Black-Box GP
    standard_gp = GPResidualModel(feature_type='q_dq')
    standard_gp.fit(X_tr, U_train[idx], Y_tr)

    # 4. Train Physics-Informed (Gray-Box) GP
    physics_gp = RobustPhysicsGP(robot)
    physics_gp.fit(X_tr, Y_tr)

    # 5. Evaluate One-Step Prediction on the AGGRESSIVE Test Set
    pred_std = standard_gp.predict(X_test, U_test)
    pred_phys = physics_gp.predict(X_test)

    rmse_std = np.sqrt(np.mean((pred_std - Y_test)**2))
    rmse_phys = np.sqrt(np.mean((pred_phys - Y_test)**2))
    rmse_nom = np.sqrt(np.mean((0.0 - Y_test)**2)) # Baseline: pure nominal model (predicting 0)

    print("=" * 60)
    print(f"EXTRAPOLATION TEST (Trained on Slow -> Tested on Aggressive)")
    print("=" * 60)
    print(f"Nominal Model (Zero Residual) RMSE : {rmse_nom:.4f} rad/s^2")
    print(f"Standard GP (Black-Box) RMSE      : {rmse_std:.4f} rad/s^2")
    print(f"Physics-Informed GP (Gray-Box) RMSE: {rmse_phys:.4f} rad/s^2")
    print("=" * 60)
    print("Joint 1 Learned [dq1, dq2, tanh1, tanh2, c1, c2]:")
    print(np.round(physics_gp.phys_model1.coef_, 3))

    print("Joint 2 Learned [dq1, dq2, tanh1, tanh2, c1, c2]:")
    print(np.round(physics_gp.phys_model2.coef_, 3))

    # PLOTTING
    _, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # ----------------- Subplot 1: Joint 1 Residual Acceleration -----------------
    axes[0].plot(t_test, Y_test[:, 0], 'k-', lw=2.2, label='True Residual (Coriolis + Friction)')
    axes[0].axhline(0, color='gray', linestyle=':', label='Nominal Model (Zero Residual)')
    axes[0].plot(t_test, pred_std[:, 0], 'r--', label='Standard GP (Black-Box)')
    axes[0].plot(t_test, pred_phys[:, 0], 'g-', label='Torque-Space Physics GP')
    axes[0].set_ylabel('Joint 1 Res. $\ddot{q}_1$ (rad/s$^2$)')
    axes[0].set_title('Extrapolation Test: Model Trained on SLOW, Evaluated on AGGRESSIVE Trajectory', fontweight='bold')
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.3)

    # ----------------- Subplot 2: Absolute Prediction Error Norm -----------------
    err_nom = np.linalg.norm(Y_test, axis=1)
    err_std = np.linalg.norm(pred_std - Y_test, axis=1)
    err_torque = np.linalg.norm(pred_phys - Y_test, axis=1)

    axes[1].plot(t_test, err_nom, color='gray', linestyle=':', label=f'Nominal Error (Mean: {np.mean(err_nom):.2f})')
    axes[1].plot(t_test, err_std, 'r--', label=f'Standard GP Error (Mean: {np.mean(err_std):.2f})')
    axes[1].plot(t_test, err_torque, 'g-', lw=2.0, label=f'Torque-Space GP Error (Mean: {np.mean(err_torque):.2f})')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(r'$\|\ddot{q}_{\mathrm{pred}} - \ddot{q}_{\mathrm{true}}\|$ (rad/s$^2$)')
    axes[1].set_title('Instantaneous Error Norm Over Time', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("residual_by_step_ex.pdf")
    plt.show()

if __name__ == '__main__':
    interpolation_exp("slow")
    interpolation_exp("aggressive")
    extrapolation_exp_fail()
    extrapolation_exp_successful()
