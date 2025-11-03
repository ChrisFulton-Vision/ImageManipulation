import numpy as np
from numpy import cos, sin, rad2deg, deg2rad
from datetime import datetime as dt, timedelta
from numpy.linalg import inv
from copy import copy, deepcopy
from matplotlib import pyplot as plt

class KalmanFilter:
    def __init__(self):
        self.x = None
        self.lastMeasTime = None
        self.sigma_proc: float = 2.0
        self.var_proc: float = self.sigma_proc ** 2
        self.sigma_meas: float = 0.02
        self.dt = 1.0 / 30.0

        self.P = np.eye(4) * 10.0
        self.H = np.zeros((2,4))
        self.H[0,0], self.H[1,1] = 1.0, 1.0
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


    def update_KF(self, new_time: dt, z : np.array = None):

        if self.x is None:
            if z is None:
                return None

            self.x = np.zeros(4,)
            self.x[:2] = deepcopy(z)
            self.lastMeasTime = new_time
            return self.updated_state()

        delta_t = (new_time - self.lastMeasTime).total_seconds()

        if delta_t < 0.001:
            return self.updated_state()

        self.lastMeasTime = new_time

        self.update_matrices(delta_t)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        if z is None:
            return self.updated_state()

        S = self.H @ self.P @ self.H.T + self.R
        I = np.eye(S.shape[0])
        K = self.P @ self.H.T @ np.linalg.solve(S, I)

        self.x += K @ (z - self.H @ self.x)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

        self.x[:2] = np.clip(self.x[:2], 0.0, 1.0)
        self.x[2:] = np.clip(self.x[2:], -1.0, 1.0)
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
        z_individ = np.array([0.5 + 0.1 * sin(deg2rad(t)) + np.random.normal(0.0, test_kf.sigma_meas), 0.5+ np.random.normal(0.0, test_kf.sigma_meas)])
        state, conf = test_kf.update_KF(next_time, z_individ)
        z.append(z_individ[0])
        xt.append(0.5 + 0.1 * sin(deg2rad(t)))
        vxt.append(0.1 * cos(deg2rad(t)))

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

    fig, ax = plt.subplots(4)

    ax[0].plot(time, xh)
    ax[0].plot(time, xt)
    ax[0].plot(time, z, alpha=0.3)

    ax[1].plot(time, vxh)
    ax[1].plot(time, vxt)

    ax[2].plot(time, xh - xt)
    ax[2].fill_between(time, -cov_x * 3.0, cov_x * 3.0, alpha=0.5)

    ax[3].plot(time, vxh - vxt)
    ax[3].fill_between(time, -cov_vx * 3.0, cov_vx * 3.0, alpha=0.5)
    ax[3].set_ylim([-1.0, 1.0])
    plt.show()