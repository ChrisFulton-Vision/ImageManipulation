import numpy as np
from numpy import cos, sin, deg2rad
from datetime import datetime as dt, timedelta
from SupportModules.include_numba import njit, prange

class KalmanFilter:
    def __init__(self):
        self.x = None
        self.lastMeasTime = None
        self.sigma_proc: float = 0.5
        self.var_proc: float = self.sigma_proc ** 2.0
        self.sigma_meas: float = 0.001
        self.var_meas: float = self.sigma_meas ** 2.0
        self.dt = 1.0 / 30.0

        # Gating parameters
        #  - max_mahalanobis_sq ~= chi^2 threshold with dof=2
        #    0.0069 ~ 95%, 0.010 ~ 99%, 0.0160 ~ 99.9%
        self.max_mahalanobis_sq: float = 0.001
        # Absolute pixel jump gate in normalized [0,1] coords (per frame)
        self.max_pixel_jump: float = 0.0001

        # Track whether the last call used the measurement or not
        self.last_used_measurement: bool = False

        # Standard KF matrices
        self.P = np.eye(4) * 10.0
        self.H = np.zeros((2, 4))
        self.H[0, 0], self.H[1, 1] = 1.0, 1.0
        self.R = np.eye(2) * self.sigma_meas ** 2.0

        self.F = np.eye(4)
        self.Q = np.zeros((4, 4))
        self.update_matrices(self.dt)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _kf_step_inplace(x, P, dt, z0, z1, has_meas,
                         var_proc, var_meas,
                         max_pixel_jump, max_mahalanobis_sq):
        """
        In-place constant-velocity KF step for one track.
        Args are strictly numeric => Numba-friendly.
        Updates x[:] and P[:,:] in place.
        Returns: used_measurement (uint8)
        """

        used = 0

        # ---- build Q terms (matches your update_matrices(dt) structure) :contentReference[oaicite:1]{index=1}
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q00 = 0.25 * dt4 * var_proc
        q02 = 0.50 * dt3 * var_proc
        q22 = dt2 * var_proc

        # ---- Predict x_pred = F x (F is CV model) :contentReference[oaicite:2]{index=2}
        x0p = x[0] + dt * x[2]
        x1p = x[1] + dt * x[3]
        x2p = x[2]
        x3p = x[3]

        # ---- Predict P_pred = F P F^T + Q (explicit 4x4 math)
        # P is 4x4
        # FP = F*P
        FP00 = P[0, 0] + dt * P[2, 0]
        FP01 = P[0, 1] + dt * P[2, 1]
        FP02 = P[0, 2] + dt * P[2, 2]
        FP03 = P[0, 3] + dt * P[2, 3]
        FP10 = P[1, 0] + dt * P[3, 0]
        FP11 = P[1, 1] + dt * P[3, 1]
        FP12 = P[1, 2] + dt * P[3, 2]
        FP13 = P[1, 3] + dt * P[3, 3]
        FP20 = P[2, 0]
        FP21 = P[2, 1]
        FP22 = P[2, 2]
        FP23 = P[2, 3]
        FP30 = P[3, 0]
        FP31 = P[3, 1]
        FP32 = P[3, 2]
        FP33 = P[3, 3]

        # Pp = FP * F^T
        Pp00 = FP00 + dt * FP02
        Pp01 = FP01 + dt * FP03
        Pp02 = FP02
        Pp03 = FP03

        Pp10 = FP10 + dt * FP12
        Pp11 = FP11 + dt * FP13
        Pp12 = FP12
        Pp13 = FP13

        Pp20 = FP20 + dt * FP22
        Pp21 = FP21 + dt * FP23
        Pp22 = FP22
        Pp23 = FP23

        Pp30 = FP30 + dt * FP32
        Pp31 = FP31 + dt * FP33
        Pp32 = FP32
        Pp33 = FP33

        # add Q
        Pp00 += q00
        Pp11 += q00
        Pp02 += q02
        Pp13 += q02
        Pp20 += q02
        Pp31 += q02
        Pp22 += q22
        Pp33 += q22

        if has_meas == 0:
            # accept prediction
            x[0] = x0p
            x[1] = x1p
            x[2] = x2p
            x[3] = x3p
            P[0, 0] = Pp00
            P[0, 1] = Pp01
            P[0, 2] = Pp02
            P[0, 3] = Pp03
            P[1, 0] = Pp10
            P[1, 1] = Pp11
            P[1, 2] = Pp12
            P[1, 3] = Pp13
            P[2, 0] = Pp20
            P[2, 1] = Pp21
            P[2, 2] = Pp22
            P[2, 3] = Pp23
            P[3, 0] = Pp30
            P[3, 1] = Pp31
            P[3, 2] = Pp32
            P[3, 3] = Pp33
            return used

        # ---- innovation y = z - H x_pred, H picks x,y :contentReference[oaicite:3]{index=3}
        y0 = z0 - x0p
        y1 = z1 - x1p

        # pixel-jump gate
        pj = y0 * y0 + y1 * y1
        if pj > max_pixel_jump * max_pixel_jump:
            x[0] = x0p
            x[1] = x1p
            x[2] = x2p
            x[3] = x3p
            P[0, 0] = Pp00
            P[0, 1] = Pp01
            P[0, 2] = Pp02
            P[0, 3] = Pp03
            P[1, 0] = Pp10
            P[1, 1] = Pp11
            P[1, 2] = Pp12
            P[1, 3] = Pp13
            P[2, 0] = Pp20
            P[2, 1] = Pp21
            P[2, 2] = Pp22
            P[2, 3] = Pp23
            P[3, 0] = Pp30
            P[3, 1] = Pp31
            P[3, 2] = Pp32
            P[3, 3] = Pp33
            return used

        # ---- S = H P H^T + R => top-left 2x2 + var_meas*I :contentReference[oaicite:5]{index=5}
        a = Pp00 + var_meas
        b = Pp01
        c = Pp10
        d = Pp11 + var_meas

        det = a * d - b * c
        if abs(det) < 1e-18:
            # skip update if ill-conditioned
            x[0] = x0p
            x[1] = x1p
            x[2] = x2p
            x[3] = x3p
            P[0, 0] = Pp00
            P[0, 1] = Pp01
            P[0, 2] = Pp02
            P[0, 3] = Pp03
            P[1, 0] = Pp10
            P[1, 1] = Pp11
            P[1, 2] = Pp12
            P[1, 3] = Pp13
            P[2, 0] = Pp20
            P[2, 1] = Pp21
            P[2, 2] = Pp22
            P[2, 3] = Pp23
            P[3, 0] = Pp30
            P[3, 1] = Pp31
            P[3, 2] = Pp32
            P[3, 3] = Pp33
            return used

        inv_det = 1.0 / det
        i00 = d * inv_det
        i01 = -b * inv_det
        i10 = -c * inv_det
        i11 = a * inv_det

        # Mahalanobis gate: m_sq = y^T inv(S) y
        invSy0 = i00 * y0 + i01 * y1
        invSy1 = i10 * y0 + i11 * y1
        m_sq = y0 * invSy0 + y1 * invSy1
        if m_sq > max_mahalanobis_sq:
            x[0] = x0p
            x[1] = x1p
            x[2] = x2p
            x[3] = x3p
            P[0, 0] = Pp00
            P[0, 1] = Pp01
            P[0, 2] = Pp02
            P[0, 3] = Pp03
            P[1, 0] = Pp10
            P[1, 1] = Pp11
            P[1, 2] = Pp12
            P[1, 3] = Pp13
            P[2, 0] = Pp20
            P[2, 1] = Pp21
            P[2, 2] = Pp22
            P[2, 3] = Pp23
            P[3, 0] = Pp30
            P[3, 1] = Pp31
            P[3, 2] = Pp32
            P[3, 3] = Pp33
            return used

        # ---- K = P_pred H^T inv(S) => K = Pp[:,0:2] * inv(S)
        k00 = Pp00 * i00 + Pp01 * i10
        k01 = Pp00 * i01 + Pp01 * i11
        k10 = Pp10 * i00 + Pp11 * i10
        k11 = Pp10 * i01 + Pp11 * i11
        k20 = Pp20 * i00 + Pp21 * i10
        k21 = Pp20 * i01 + Pp21 * i11
        k30 = Pp30 * i00 + Pp31 * i10
        k31 = Pp30 * i01 + Pp31 * i11

        # x = x_pred + K y
        x0 = x0p + k00 * y0 + k01 * y1
        x1 = x1p + k10 * y0 + k11 * y1
        x2 = x2p + k20 * y0 + k21 * y1
        x3 = x3p + k30 * y0 + k31 * y1

        # P = (I - K H) Pp  (simple form; matches your structure) :contentReference[oaicite:7]{index=7}
        A00 = 1.0 - k00
        A01 = -k01
        A10 = -k10
        A11 = 1.0 - k11
        A20 = -k20
        A21 = -k21
        A30 = -k30
        A31 = -k31

        P00 = A00 * Pp00 + A01 * Pp10
        P01 = A00 * Pp01 + A01 * Pp11
        P02 = A00 * Pp02 + A01 * Pp12
        P03 = A00 * Pp03 + A01 * Pp13

        P10 = A10 * Pp00 + A11 * Pp10
        P11 = A10 * Pp01 + A11 * Pp11
        P12 = A10 * Pp02 + A11 * Pp12
        P13 = A10 * Pp03 + A11 * Pp13

        P20 = A20 * Pp00 + A21 * Pp10 + Pp20
        P21 = A20 * Pp01 + A21 * Pp11 + Pp21
        P22 = A20 * Pp02 + A21 * Pp12 + Pp22
        P23 = A20 * Pp03 + A21 * Pp13 + Pp23

        P30 = A30 * Pp00 + A31 * Pp10 + Pp30
        P31 = A30 * Pp01 + A31 * Pp11 + Pp31
        P32 = A30 * Pp02 + A31 * Pp12 + Pp32
        P33 = A30 * Pp03 + A31 * Pp13 + Pp33

        # clamp like your existing post-update clamps :contentReference[oaicite:8]{index=8}
        if x0 < 0.0: x0 = 0.0
        if x0 > 1.0: x0 = 1.0
        if x1 < 0.0: x1 = 0.0
        if x1 > 1.0: x1 = 1.0
        if x2 < -1.0: x2 = -1.0
        if x2 > 1.0: x2 = 1.0
        if x3 < -1.0: x3 = -1.0
        if x3 > 1.0: x3 = 1.0

        # write back
        x[0] = x0
        x[1] = x1
        x[2] = x2
        x[3] = x3
        P[0, 0] = P00
        P[0, 1] = P01
        P[0, 2] = P02
        P[0, 3] = P03
        P[1, 0] = P10
        P[1, 1] = P11
        P[1, 2] = P12
        P[1, 3] = P13
        P[2, 0] = P20
        P[2, 1] = P21
        P[2, 2] = P22
        P[2, 3] = P23
        P[3, 0] = P30
        P[3, 1] = P31
        P[3, 2] = P32
        P[3, 3] = P33

        used = 1
        return used

    @staticmethod
    @njit(parallel=True, cache=True, fastmath=True)
    def _kf_bank_step_inplace(
            t_now,
            meas_x, meas_y, meas_valid,  # (M,) arrays for this row
            X, P, last_t, init,
            var_proc, var_meas,
            max_pixel_jump, max_mahalanobis_sq
    ):
        """
        One row update for ALL features.
        - meas_x/y are per-feature measurements for this row (already normalized if desired)
        - meas_valid[j] = 1 if measurement exists, else 0 (predict-only)
        """
        M = X.shape[0]
        used = np.zeros(M, dtype=np.uint8)

        for j in prange(M):
            if init[j] == 0:
                if meas_valid[j] != 0:
                    X[j, 0] = meas_x[j]
                    X[j, 1] = meas_y[j]
                    X[j, 2] = 0.0
                    X[j, 3] = 0.0
                    init[j] = 1
                    last_t[j] = t_now
                    used[j] = 1
                continue

            dt = t_now - last_t[j]
            if dt < 0.001:
                continue
            last_t[j] = t_now

            # call your existing per-track compiled step
            z0 = meas_x[j]
            z1 = meas_y[j]
            has_meas = meas_valid[j]

            used[j] = _KF_STEP(
                X[j], P[j], dt,
                z0, z1, has_meas,
                var_proc, var_meas,
                max_pixel_jump, max_mahalanobis_sq
            )

        return used

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
        Same signature as before, but uses a Numba-accelerated core.
        """

        self.last_used_measurement = False

        # init
        if self.x is None:
            if z is None:
                return None
            self.x = np.zeros(4, dtype=np.float64)
            self.x[0:2] = z[0:2]
            self.P = np.eye(4, dtype=np.float64) * 10.0  # matches your init :contentReference[oaicite:9]{index=9}
            self.lastMeasTime = new_time
            self.last_used_measurement = True
            return self.updated_state()

        # default time
        if new_time is None:
            new_time = self.dt + self.lastMeasTime

        delta_t = new_time - self.lastMeasTime
        if isinstance(delta_t, timedelta):
            delta_t = delta_t.total_seconds()

        if delta_t < 0.001:
            return self.updated_state()

        self.lastMeasTime = new_time

        # measurement presence + scalar extract
        if z is None:
            has_meas = 0
            z0 = 0.0
            z1 = 0.0
        else:
            has_meas = 1
            z = np.asarray(z, dtype=np.float64)
            z0 = float(z[0]);
            z1 = float(z[1])

        used = KalmanFilter._KF_STEP(
            self.x, self.P, float(delta_t),
            z0, z1, has_meas,
            float(self.var_proc), float(self.var_meas),
            float(self.max_pixel_jump), float(self.max_mahalanobis_sq),
        )

        self.last_used_measurement = bool(used)
        return self.updated_state()

    def updated_state(self):
        return self.x, np.diag(self.P)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

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

# Numba-friendly global alias (lets njit code call the compiled step without referencing the class object)
_KF_STEP = KalmanFilter._kf_step_inplace