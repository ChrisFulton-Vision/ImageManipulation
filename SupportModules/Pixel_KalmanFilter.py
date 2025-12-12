import numpy as np
from numpy import cos, sin, deg2rad
from datetime import datetime as dt, timedelta
from copy import deepcopy
from matplotlib import pyplot as plt


class KalmanFilter:
    def __init__(self):
        self.x = None
        self.lastMeasTime = None
        self.sigma_proc: float = 0.05
        self.var_proc: float = self.sigma_proc ** 2
        self.sigma_meas: float = 0.05
        self.dt = 1.0 / 30.0

        # Gating parameters
        #  - max_mahalanobis_sq ~= chi^2 threshold with dof=2
        #    5.99 ~ 95%, 9.21 ~ 99%, 13.8 ~ 99.9%
        self.max_mahalanobis_sq: float = 0.1
        # Absolute pixel jump gate in normalized [0,1] coords (per frame)
        self.max_pixel_jump: float = 0.02

        # Track whether the last call used the measurement or not
        self.last_used_measurement: bool = False

        # Standard KF matrices
        self.P = np.eye(4) * 10.0
        self.H = np.zeros((2, 4))
        self.H[0, 0], self.H[1, 1] = 1.0, 1.0
        self.R = np.eye(2) * self.sigma_meas ** 2

        self.F = np.eye(4)
        self.Q = np.zeros((4, 4))
        self.update_matrices(self.dt)

    def update_matrices(self, new_delta_t: float):
        self.dt = new_delta_t
        self.update_F(new_delta_t)
        self.update_Q(new_delta_t)

    def update_F(self, new_delta_t: float):
        self.F[[0, 1], [2, 3]] = new_delta_t

    def update_Q(self, new_delta_t: float):
        dt = new_delta_t
        dt_sq = dt * dt
        dt_cb = dt_sq * dt
        dt_qu = dt_cb * dt

        self.Q[[0, 1], [0, 1]] = 0.25 * dt_qu * self.var_proc
        self.Q[[0, 1, 2, 3], [2, 3, 0, 1]] = 0.50 * dt_cb * self.var_proc
        self.Q[[2, 3], [2, 3]] = dt_sq * self.var_proc

    def update_KF(self, new_time=None, z: np.array = None):
        """
        Run one KF step. If z is an outlier (unrealistic jump),
        the update is rejected and we only use the prediction.
        """

        # Default: assume no measurement used this step
        self.last_used_measurement = False

        # First-ever measurement initializes the filter
        if self.x is None:
            if z is None:
                return None

            self.x = np.zeros(4,)
            self.x[:2] = deepcopy(z)
            self.lastMeasTime = new_time
            self.last_used_measurement = True
            return self.updated_state()

        if new_time is None:
            new_time = self.dt + self.lastMeasTime

        delta_t = new_time - self.lastMeasTime

        if delta_t < 0.001:
            # Too small a time step; ignore
            return self.updated_state()

        self.lastMeasTime = new_time

        # ---- Prediction step ----
        self.update_matrices(delta_t)
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # If no measurement, just accept prediction
        if z is None:
            self.x = x_pred
            self.P = P_pred
            return self.updated_state()

        # ---- Gating / outlier rejection ----
        # Innovation (residual) and its covariance
        y = z - (self.H @ x_pred)          # innovation (2x1)
        S = self.H @ P_pred @ self.H.T + self.R  # innovation covariance (2x2)

        # 1) Absolute pixel jump gate (norm in normalized image space)
        pixel_jump = float(np.linalg.norm(y))
        if pixel_jump > self.max_pixel_jump:
            # Unreasonably large jump in pixel space -> reject measurement
            self.x = x_pred
            self.P = P_pred
            # last_used_measurement remains False
            return self.updated_state()

        # 2) Mahalanobis distance gate (chi-square test in innovation space)
        try:
            # Solve S^{-1} y via linear solve (more stable than explicit inverse)
            m_sq = float(y.T @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            # If S is singular/ill-conditioned, be conservative and skip update
            self.x = x_pred
            self.P = P_pred
            return self.updated_state()

        if m_sq > self.max_mahalanobis_sq:
            # Innovation too large compared to expected uncertainty -> outlier
            self.x = x_pred
            self.P = P_pred
            return self.updated_state()

        # ---- Standard KF measurement update (only if passed gates) ----
        I2 = np.eye(S.shape[0])
        K = P_pred @ self.H.T @ np.linalg.solve(S, I2)

        self.x = x_pred + K @ y
        I4 = np.eye(P_pred.shape[0])
        self.P = (I4 - K @ self.H) @ P_pred @ (I4 - K @ self.H).T + K @ self.R @ K.T

        # Clamp to reasonable ranges
        self.x[:2] = np.clip(self.x[:2], 0.0, 1.0)
        self.x[2:] = np.clip(self.x[2:], -10.0, 10.0)

        self.last_used_measurement = True
        return self.updated_state()

    def updated_state(self):
        return self.x, np.sqrt(np.diag(self.P))


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, threshold=np.inf)
    test_kf = KalmanFilter()

    start_time = dt(2025, 8, 30, 9, 0, 0)
    next_time = start_time

    test_kf.update_KF(start_time, np.array([0.5, 0.5]))

    xh = []
    vxh = []
    xt = []
    vxt = []
    z = []
    cov_x = []
    cov_vx = []
    time = []

    for t in range(1000):
        time.append(t)
        next_time = next_time + timedelta(milliseconds=10)

        # True trajectory
        true_x = 0.5 + 0.1 * sin(deg2rad(t))
        true_vx = 0.1 * cos(deg2rad(t))

        # Noisy measurement
        z_individ = np.array([
            true_x + np.random.normal(0.0, test_kf.sigma_meas),
            0.5 + np.random.normal(0.0, test_kf.sigma_meas)
        ])

        # Occasionally inject a big outlier jump (simulating a false positive)
        if t % 200 == 150:
            z_individ = np.array([
                0.05 + np.random.normal(0.0, test_kf.sigma_meas),
                0.05 + np.random.normal(0.0, test_kf.sigma_meas)
            ])

        state, conf = test_kf.update_KF(next_time, z_individ)

        z.append(z_individ[0])
        xt.append(true_x)
        vxt.append(true_vx)

        xh.append(state[0])
        vxh.append(state[2])
        cov_x.append(conf[0])
        cov_vx.append(conf[2])

    time = np.array(time)
    xh = np.array(xh)
    xt = np.array(xt)
    vxh = np.array(vxh)
    vxt = np.array(vxt)
    z = np.array(z)
    cov_x = np.array(cov_x)
    cov_vx = np.array(cov_vx)

    fig, ax = plt.subplots(4, sharex=True)

    ax[0].plot(time, xt, label="True x")
    ax[0].plot(time, z, alpha=0.3, label="Measurements")
    ax[0].plot(time, xh, label="KF estimate")
    ax[0].legend()

    ax[1].plot(time, vxt, label="True vx")
    ax[1].plot(time, vxh, label="KF vx")
    ax[1].legend()

    ax[2].plot(time, xh - xt, label="Position error")
    ax[2].fill_between(time, -cov_x * 3.0, cov_x * 3.0, alpha=0.3, label="3σ band")
    ax[2].legend()

    ax[3].plot(time, vxh - vxt, label="Velocity error")
    ax[3].fill_between(time, -cov_vx * 3.0, cov_vx * 3.0, alpha=0.3, label="3σ band")
    ax[3].set_ylim([-1.0, 1.0])
    ax[3].legend()

    plt.show()
