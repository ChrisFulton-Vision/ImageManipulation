import numpy as np
import datetime
import matplotlib.pyplot as plt
from numpy import linalg as la
import copy
import pickle as pkl
from dataclasses import dataclass


@dataclass(slots=True)
class PredictedState:
    query_time: float | None = None
    state_time: float | None = None
    dt: float = 0.0

    r_T_d: np.ndarray | None = None
    r_V_d: np.ndarray | None = None
    r_A_d: np.ndarray | None = None

    var_x: float | None = None
    var_y: float | None = None
    var_z: float | None = None

    is_extrapolated: bool = False



@dataclass(slots=True)
class FactorGraphFrameOutput:
    curr_FG_pixel: np.ndarray | None = None
    curr_r_T_d: np.ndarray | None = None
    curr_r_V_d: np.ndarray | None = None
    curr_r_A_d: np.ndarray | None = None

    curr_var_x: float | None = None
    curr_var_y: float | None = None
    curr_var_z: float | None = None


@dataclass(slots=True)
class HyperFocusPass:
    x_axes: float
    y_axes: float
    dim_factor: float


@dataclass(slots=True)
class HyperFocusPlan:
    center: tuple[float, float] | np.ndarray | None = None
    passes: tuple[HyperFocusPass, ...] = ()

    next_radius: float | None = None
    next_min_radius: float | None = None
    desired_yolo_conf: float | None = None

def build_hyper_focus_plan(ctx,
                           radius: float,
                           min_radius: float,
                           frame_shape) -> HyperFocusPlan | None:
    """
    Build the policy/state-update plan for hyper focus.

    This owns:
      - mode selection (YOLO-only vs FG-guided)
      - radius / min_radius state evolution
      - desired YOLO confidence schedule

    The GUI still owns:
      - actually drawing the dimmed regions
      - writing yoloSession.conf
    """
    yolo = ctx.yolo.get_or(False)
    if not yolo:
        return None

    # --- Mode 1: YOLO-only spotlight ---
    if not ctx.fg.is_set():
        smoothed_radius = float(radius)

        if getattr(yolo, "last_bounding_box_size", None) is not None:
            bbox_w, bbox_h = yolo.last_bounding_box_size
            smoothed_radius = (float(bbox_w) + float(bbox_h) + float(radius) * 4.0) / 5.0

        next_radius = min(800.0, smoothed_radius + 12.0)
        desired_yolo_conf = (0.8 - 0.5) * next_radius / 800.0 + 0.5

        center = getattr(yolo, "last_yolo_center", None)
        if center is None:
            return HyperFocusPlan(
                center=None,
                passes=(),
                next_radius=next_radius,
                next_min_radius=min_radius,
                desired_yolo_conf=desired_yolo_conf,
            )

        return HyperFocusPlan(
            center=center,
            passes=(
                HyperFocusPass(x_axes=3.0 * smoothed_radius,
                               y_axes=3.0 * smoothed_radius,
                               dim_factor=0.00),
                HyperFocusPass(x_axes=1.5 * smoothed_radius,
                               y_axes=1.5 * smoothed_radius,
                               dim_factor=0.50),
            ),
            next_radius=next_radius,
            next_min_radius=min_radius,
            desired_yolo_conf=desired_yolo_conf,
        )

    # --- Mode 2: FG-guided ellipse ---
    fg = ctx.fg.get()

    next_min_radius = float(min_radius)
    if getattr(yolo, "last_bounding_box_size", None) is not None:
        bbox_w, bbox_h = yolo.last_bounding_box_size
        # 1.0 for single feature, 1.5 for drogue
        next_min_radius = (float(bbox_w) + float(bbox_h)) * 1.0

    center = getattr(fg, "curr_FG_pixel", None)
    if center is None:
        return HyperFocusPlan(
            center=None,
            passes=(),
            next_radius=radius,
            next_min_radius=next_min_radius,
            desired_yolo_conf=None,
        )

    x = float(center[0])
    y = float(center[1])
    h, w = frame_shape[:2]

    if x < 0 or y < 0 or x > w or y > h:
        return HyperFocusPlan(
            center=None,
            passes=(),
            next_radius=radius,
            next_min_radius=next_min_radius,
            desired_yolo_conf=None,
        )

    var_y = 0.0 if getattr(fg, "curr_var_y", None) is None else float(fg.curr_var_y)
    var_z = 0.0 if getattr(fg, "curr_var_z", None) is None else float(fg.curr_var_z)

    # 5.0 for single feature, 50.0 for drogue
    ellipse_width = 5.0 * var_y + next_min_radius
    ellipse_height = 5.0 * var_z + next_min_radius

    return HyperFocusPlan(
        center=center,
        passes=(
            HyperFocusPass(x_axes=ellipse_width,
                           y_axes=ellipse_height,
                           dim_factor=0.10),
            HyperFocusPass(x_axes=ellipse_width * 2.0,
                           y_axes=ellipse_height * 2.0,
                           dim_factor=0.00),
        ),
        next_radius=radius,
        next_min_radius=next_min_radius,
        desired_yolo_conf=None,
    )

def extract_measurement_from_yolo(yolo) -> np.ndarray | None:
    """
    Prefer QnP tvec, then PnP tvec, then the legacy YOLO width-based 3D estimate.
    """
    if yolo is None:
        return None

    pose = getattr(yolo, "pose", None)
    if pose is not None:
        if getattr(pose, "qnp_tvec", None) is not None:
            return np.asarray(pose.qnp_tvec, dtype=float).reshape(3)
        if getattr(pose, "pnp_tvec", None) is not None:
            return np.asarray(pose.pnp_tvec, dtype=float).reshape(3)

    last_est = getattr(yolo, "last_yolo_3d_estimate", None)
    if last_est is not None:
        return np.asarray(last_est, dtype=float).reshape(3)

    return None


def run_factor_graph_step(fg,
                          yolo,
                          img_time,
                          last_time_update: float,
                          q_wr=None,
                          R_wr=None):
    """
    Owns:
      - lazy FG creation
      - measurement extraction
      - reset-on-time-regression behavior
      - addRecvMeas
      - predict

    Returns:
      fg, last_time_update, has_measurement, pred
    """
    if fg is None:
        fg = FactorGraph()

    meas_3d = extract_measurement_from_yolo(yolo)
    has_measurement = meas_3d is not None

    if has_measurement:
        if img_time is None or img_time < last_time_update:
            fg.reset()

        fg.addRecvMeas(meas_3d, t=img_time, q_wr=q_wr, R_wr=R_wr)

        # Avoid poisoning last_time_update with None.
        if img_time is not None:
            last_time_update = float(img_time)

    pred = None
    if img_time is not None and fg.numMeas > 2:
        pred = fg.predict(img_time, q_wr=q_wr, R_wr=R_wr)

    return fg, last_time_update, has_measurement, pred


def factor_graph_projection_matrix(calibration, markup_frame: np.ndarray, yolo=None) -> np.ndarray:
    """
    Match the old behavior:
      - use current calibration K directly when YOLO already produced a native 3D estimate
      - otherwise temporarily scale calibration to the markup frame height
    """
    K = calibration.getCameraMatrix()

    last_yolo_3d_estimate = None if yolo is None else getattr(yolo, "last_yolo_3d_estimate", None)
    if last_yolo_3d_estimate is None:
        curr_scale = copy.deepcopy(getattr(calibration, "scale", None))
        try:
            calibration.scaleCalibration(markup_frame.shape[0])
            K = calibration.getCameraMatrix()
        finally:
            if curr_scale is not None:
                calibration.scale = curr_scale

    return np.asarray(K, dtype=float)


def build_factor_graph_output(pred: PredictedState | None,
                              K: np.ndarray) -> FactorGraphFrameOutput | None:
    if pred is None or pred.r_T_d is None:
        return None

    r_T_d = np.asarray(pred.r_T_d, dtype=float).reshape(3)
    threeD_proj = np.asarray(K, dtype=float).dot(r_T_d)

    pixel = None
    if abs(float(threeD_proj[2])) > 1e-12:
        pixel = threeD_proj[:2] / threeD_proj[2]

    return FactorGraphFrameOutput(
        curr_FG_pixel=pixel,
        curr_r_T_d=r_T_d,
        curr_r_V_d=None if pred.r_V_d is None else np.asarray(pred.r_V_d, dtype=float).reshape(3),
        curr_r_A_d=None if pred.r_A_d is None else np.asarray(pred.r_A_d, dtype=float).reshape(3),
        curr_var_x=pred.var_x,
        curr_var_y=pred.var_y,
        curr_var_z=pred.var_z,
    )

class SolutionData:
    def __init__(self, numMeas):
        self.numMeas = numMeas
        self.drg_pos = np.zeros((numMeas, 3))
        self.drg_vel = np.zeros((numMeas, 3))
        self.drg_acc = np.zeros((numMeas, 3))

        self.r_T_d = np.zeros((3,))
        self.r_V_d = np.zeros((3,))
        self.r_A_d = np.zeros((3,))

        self.populated = False

    def ingest_xh(self, states) -> None:
        self.r_T_d, self.r_V_d, self.r_A_d = states
        self.populated = True

    def __str__(self):
        if self.populated:
            posStr = ''
            velStr = ''

            for idx in range(self.numMeas):
                posStr += f'{idx}, '
                posStr += f'd_R: {self.drg_pos[idx]}\n'
                velStr += f'{idx}, dV_R: {self.drg_vel[idx]}\n'

            return posStr + '\n' + velStr
        else:
            return 'Not populated yet...'


class FactorGraph:
    def __init__(self, printStuff=False, startT=None):
        self.optComplete = False
        np.set_printoptions(threshold=np.inf, precision=5, suppress=True)
        self.solution = None
        self.printStuff = printStuff

        self.r_T_d = np.zeros((1, 3))
        self.r_V_d = np.zeros((1, 3))
        self.r_A_d = np.zeros((1, 3))

        self.meas = np.zeros((1,))
        self.curr_meas = np.zeros((1,))

        self.L = np.zeros((1, 1))
        self.y = np.zeros(1, )

        self.init_guess = np.zeros(1, )
        self.init_residual = np.zeros((1,))
        self.num_iters = 0

        self.cam_cov = np.array([10.0, 10.0, 1.0]) * (5.0 ** 2)

        # Position-dynamics residual weight
        self.V_cov = np.array([1.0, 1.0, 1.0]) * 5.0

        # Velocity-dynamics residual weight
        self.A_cov = np.array([1.0, 1.0, 1.0]) * 10.0

        # Acceleration smoothness residual weight
        self.Adot_cov = np.array([1.0, 1.0, 1.0]) * 1000.

        self.numMeas = 0

        if startT is None:
            epoch = datetime.datetime(1970, 1, 1)
            self.startTime = (datetime.datetime.now() - epoch).total_seconds()
        else:
            self.startTime = startT
        self.lastMeasTime = datetime.time()
        self.time_log = np.zeros((1,))
        self.marginalizeNum = 5

    def __str__(self):
        selfStr = ''
        for idx, t in enumerate(self.time_log):
            recv_to_drog_pos = slice(idx * 16 + 0, idx * 16 + 3)
            recv_to_drog_vel = slice(idx * 16 + 3, idx * 16 + 6)

            selfStr += str(idx) + ', ' + f'{t:.3f},  \n'
        return selfStr

    def copy(self, classToCopy):
        self.__dict__.update(copy.deepcopy(classToCopy.__dict__))

    @staticmethod
    def _as_rotmat(q_wr=None, R_wr=None):
        """
        Return a 3x3 rotation matrix R_wr mapping receiver-frame vectors into
        the stabilized/world frame.

        Accepts either:
          - R_wr directly, or
          - q_wr as an object with a known matrix conversion method.
        """
        if R_wr is not None:
            R = np.asarray(R_wr, dtype=float)
            if R.shape != (3, 3):
                raise ValueError(f'R_wr must have shape (3,3), got {R.shape}')
            return R

        if q_wr is None:
            return None

        # Common quaternion helper names
        for attr in ('rotmat_wr', 'toRotMat', 'toMat3', 'as_rotmat', 'as_matrix'):
            if hasattr(q_wr, attr):
                obj = getattr(q_wr, attr)
                R = obj() if callable(obj) else obj
                R = np.asarray(R, dtype=float)
                if R.shape != (3, 3):
                    raise ValueError(f'q_wr.{attr} did not produce shape (3,3), got {R.shape}')
                return R

        raise ValueError('q_wr was provided, but no recognized rotation-matrix conversion exists.')

    def addRecvMeas(self,
                    drgVec,
                    t=None,
                    q_wr=None,
                    R_wr=None):
        self.optComplete = False

        drgVec = np.asarray(drgVec, dtype=float).reshape(3)

        R_wr = self._as_rotmat(q_wr=q_wr, R_wr=R_wr)

        # If receiver attitude is provided, rotate receiver-frame measurement into world frame.
        # Otherwise preserve legacy behavior.
        if R_wr is not None:
            meas_vec = R_wr @ drgVec
        else:
            meas_vec = drgVec.copy()

        # Then the rest is basically your current addRecvMeas logic,
        # but use meas_vec instead of drgVec.
        if self.numMeas == 0:
            if meas_vec[2] > 200.0:
                return

            self.curr_meas = [meas_vec]
            self.meas = self.curr_meas

            self.r_T_d[0, :] = meas_vec
            self.r_V_d = np.zeros((1, 3))
            self.r_A_d = np.zeros((1, 3))

            if t is None:
                epoch = datetime.datetime(1970, 1, 1)
                self.time_log[0] = (datetime.datetime.now() - epoch).total_seconds() - self.startTime
            else:
                self.time_log[0] = t - self.startTime
        else:
            if meas_vec[2] > 200.0:
                meas_vec *= self.meas[-1][2] / meas_vec[2]

            if t is None:
                epoch = datetime.datetime(1970, 1, 1)
                t = (datetime.datetime.now() - epoch).total_seconds() - self.startTime
            else:
                t = t - self.startTime

            delT = t - self.time_log[-1]

            if t <= self.time_log[-1]:
                self.reset()
                self.addRecvMeas(meas_vec, t=self.startTime + t)
                return

            if np.linalg.norm((meas_vec - self.r_T_d[-1]) / delT) > 100.0:
                self.popOldestMeas()
                return

            self.time_log = np.append(self.time_log, t)
            self.curr_meas = meas_vec
            self.meas.append(self.curr_meas)

            self.r_T_d = np.append(self.r_T_d, meas_vec[np.newaxis, :], axis=0)

            new_vel = self.r_V_d[-1].copy() if self.r_V_d.shape[0] > 0 else np.zeros((3,))
            new_acc = self.r_A_d[-1].copy() if self.r_A_d.shape[0] > 0 else np.zeros((3,))

            alpha_v = 0.15
            if delT > 1e-6 and self.r_T_d.shape[0] >= 2:
                fd_vel = (self.r_T_d[-1] - self.r_T_d[-2]) / delT
                new_vel = (1.0 - alpha_v) * new_vel + alpha_v * fd_vel

            self.r_V_d = np.append(self.r_V_d, new_vel[np.newaxis, :], axis=0)
            self.r_A_d = np.append(self.r_A_d, new_acc[np.newaxis, :], axis=0)

        self.numMeas += 1

    def create_Q(self):
        N_y = (self.numMeas - 1) * 12 + 3
        Q = np.eye(N_y, N_y)

        for idx in range(self.numMeas):
            base = idx * 12
            cam_T_d = self.meas[idx]

            Q[base + 0: base + 3, base + 0: base + 3] = (
                    1.0 / la.norm(cam_T_d) * np.diag(self.cam_cov)
            )

            if idx < self.numMeas - 1:
                Q[base + 3: base + 6, base + 3: base + 6] = np.diag(self.V_cov)
                Q[base + 6: base + 9, base + 6: base + 9] = np.diag(self.A_cov)
                Q[base + 9: base + 12, base + 9: base + 12] = np.diag(self.Adot_cov)

        return Q

    def create_y(self, states=None):
        N_y = (self.numMeas - 1) * 12 + 3

        if states is None:
            states_r_T_d = copy.copy(self.r_T_d)
            states_r_V_d = copy.copy(self.r_V_d)
            states_r_A_d = copy.copy(self.r_A_d)
        else:
            states_r_T_d, states_r_V_d, states_r_A_d = states

        y = np.zeros(N_y)

        for meas_num in range(self.numMeas):
            s_drgPos = states_r_T_d[meas_num]
            s_drgVel = states_r_V_d[meas_num]
            s_drgAcc = states_r_A_d[meas_num]

            meas_res, dyn_t, dyn_v, dyn_a = self.resFromTimestep(meas_num)
            y[meas_res] = self.meas[meas_num] - s_drgPos

            if meas_num < self.numMeas - 1:
                s_nextDrgPos = states_r_T_d[meas_num + 1]
                s_nextDrgVel = states_r_V_d[meas_num + 1]
                s_nextDrgAcc = states_r_A_d[meas_num + 1]

                deltT = self.time_log[meas_num + 1] - self.time_log[meas_num]

                # p_{k+1} = p_k + dt v_k + 0.5 dt^2 a_k
                y[dyn_t] = (
                        s_nextDrgPos
                        - s_drgPos
                        - deltT * s_drgVel
                        - 0.5 * (deltT ** 2) * s_drgAcc
                )

                # v_{k+1} = v_k + dt a_k
                y[dyn_v] = (
                        s_nextDrgVel
                        - s_drgVel
                        - deltT * s_drgAcc
                )

                # a_{k+1} = a_k
                y[dyn_a] = s_nextDrgAcc - s_drgAcc

        return y

    def reset(self):
        self.r_T_d = np.zeros((1, 3))
        self.r_V_d = np.zeros((1, 3))
        self.r_A_d = np.zeros((1, 3))
        self.time_log = np.zeros((1,))
        self.meas = np.zeros((1,))
        self.numMeas = 0

    def popOldestMeas(self):
        self.optComplete = False

        if self.numMeas > 1:
            self.numMeas -= 1

            self.time_log = self.time_log[1:]
            self.meas = self.meas[1:]

            self.r_T_d = self.r_T_d[1:]
            self.r_V_d = self.r_V_d[1:]
            self.r_A_d = self.r_A_d[1:]
        else:
            self.reset()

    def stateFromTimestep(self, meas_idx):
        # s: r_T_d, r_V_d, r_A_d
        base = meas_idx * 9
        return (
            slice(base + 0, base + 3),
            slice(base + 3, base + 6),
            slice(base + 6, base + 9),
        )

    def measFromTimestep(self, meas_idx):
        # m: r_T_d
        return slice(meas_idx * 3 + 0, meas_idx * 3 + 3)

    def resFromTimestep(self, meas_idx):
        # meas, pos_dyn, vel_dyn, acc_dyn
        base = meas_idx * 12
        return (
            slice(base + 0, base + 3),
            slice(base + 3, base + 6),
            slice(base + 6, base + 9),
            slice(base + 9, base + 12),
        )

    def create_L(self):
        N_y = self.numMeas * 12 - 9
        N_x = self.numMeas * 9
        L = np.zeros((N_y, N_x))

        for meas_num in range(self.numMeas):
            sID_r_T_d, sID_r_V_d, sID_r_A_d = self.stateFromTimestep(meas_num)
            meas_res, dyn_t, dyn_v, dyn_a = self.resFromTimestep(meas_num)

            # y_meas = meas - pos_k
            L[meas_res, sID_r_T_d] = -np.eye(3)

            if meas_num < self.numMeas - 1:
                sID_r_T_d_next, sID_r_V_d_next, sID_r_A_d_next = self.stateFromTimestep(meas_num + 1)
                deltT = self.time_log[meas_num + 1] - self.time_log[meas_num]

                # y_pos = pos_{k+1} - pos_k - dt vel_k - 0.5 dt^2 acc_k
                L[dyn_t, sID_r_T_d_next] = np.eye(3)
                L[dyn_t, sID_r_T_d] = -np.eye(3)
                L[dyn_t, sID_r_V_d] = -deltT * np.eye(3)
                L[dyn_t, sID_r_A_d] = -0.5 * (deltT ** 2) * np.eye(3)

                # y_vel = vel_{k+1} - vel_k - dt acc_k
                L[dyn_v, sID_r_V_d_next] = np.eye(3)
                L[dyn_v, sID_r_V_d] = -np.eye(3)
                L[dyn_v, sID_r_A_d] = -deltT * np.eye(3)

                # y_acc = acc_{k+1} - acc_k
                L[dyn_a, sID_r_A_d_next] = np.eye(3)
                L[dyn_a, sID_r_A_d] = -np.eye(3)

        return L

    def update_states(self, states):
        self.r_T_d, self.r_V_d, self.r_A_d = states

    def calc_next_states(self, delta_x):
        r_T_d = copy.deepcopy(self.r_T_d)
        r_V_d = copy.deepcopy(self.r_V_d)
        r_A_d = copy.deepcopy(self.r_A_d)

        for meas_num in range(self.numMeas):
            r_T_d_slice, r_V_d_slice, r_A_d_slice = self.stateFromTimestep(meas_num)

            r_T_d[meas_num] -= delta_x[r_T_d_slice]

            # Do not let the freshest tail node freely absorb all dynamics noise.
            if meas_num < self.numMeas - 1:
                r_V_d[meas_num] -= delta_x[r_V_d_slice]
                r_A_d[meas_num] -= delta_x[r_A_d_slice]

        # Tail stabilization: newest node inherits from prior stabilized node.
        if self.numMeas > 1:
            r_V_d[-1] = r_V_d[-2].copy()
            r_A_d[-1] = r_A_d[-2].copy()

        return [r_T_d, r_V_d, r_A_d]

    def opt(self, func=None):
        if self.time_log[-1] <= self.time_log[-2]:
            self.reset()
            return

        self.num_iters = 0
        prev_ratio = np.inf
        keep_going = True
        stop = False
        Q = self.create_Q()
        scale = 1.0

        next_states = [copy.deepcopy(self.r_T_d),
                       copy.deepcopy(self.r_V_d),
                       copy.deepcopy(self.r_A_d)]
        while keep_going:
            startProcTime = datetime.datetime.now()
            np.set_printoptions(precision=3, threshold=np.inf)
            # print(f'Pre: \n{self}')

            y = self.create_y()
            is_scale_good = False
            Qy = Q.dot(self.create_y())
            Qy_mag = Qy.T.dot(Qy)
            QL = Q.dot(self.create_L())
            # Qy = self.create_y()
            # QL = self.create_L()
            # Q = np.eye(len(y), len(y))

            startRes = la.norm(Qy)
            # print(f'Start-||y||: {startRes:.3f}')

            startInvTime = datetime.datetime.now()
            try:
                delta_x = la.pinv(QL).dot(Qy)
            except np.linalg.LinAlgError as e:
                print(f'SVD did not converge. Error Message: \n{e}')
                self.reset()
                return

            # print(f'Norm delX: {np.linalg.norm(delta_x)}')
            endInvTime = datetime.datetime.now()

            prev_ratio = np.inf

            while not is_scale_good:
                # print('Starting Estimate')

                next_states = self.calc_next_states(delta_x * scale)

                # print('Updated Estimate')
                new_Qy = Q.dot(self.create_y(next_states))

                pred_Qy = Qy - QL.dot(delta_x * scale)

                if np.abs(Qy_mag - pred_Qy.dot(pred_Qy)) > 0.00001:
                    ratio = (Qy_mag - new_Qy.dot(new_Qy)) / (Qy_mag - pred_Qy.dot(pred_Qy))
                else:
                    ratio = 1.0

                # print(f'Old y_mag: {np.linalg.norm(Qy):.4f}, New y_mag {np.linalg.norm(new_Qy):.4f}, Pred y_mag {np.linalg.norm(pred_Qy)}')

                if .2 < ratio < 5.0:
                    # print(f'Scale: {scale}, ratio: {ratio})')
                    is_scale_good = True
                    self.update_states(next_states)
                    self.y = self.create_y()
                    self.L = self.create_L()
                    # self.plotL(self.L)
                else:
                    scale /= 2.0
                    print(f'Scale: {scale}, ratio: {ratio}, y: {la.norm(new_Qy):.3f}')

                    if scale < 0.0001 or np.abs(ratio - 1.0) > np.abs(prev_ratio - 1.0):
                        is_scale_good = True
                        stop = True
                        # self.update_states(next_states, next_cameras)
                        self.y = self.create_y()
                        self.L = self.create_L()

                prev_ratio = copy.copy(ratio)

            self.num_iters += 1
            keep_going = la.norm(scale * delta_x) > 0.0001 and self.num_iters <= 10 and not stop
            # keep_going = False # Linear; achieve immediate convergence
            if func is not None:
                func(f'Optimize Data \nCurrent Residual: {la.norm(Q.dot(self.y)):.3f}')
            # print(f'Start Residual: {startRes:.3f}')
            # print(f'End Residual: {la.norm(Q.dot(self.y)):.3f}')
            # print(f'ShouldBeEnd Residual: {la.norm(new_Qy):.3f}')

            # print(f'Iteration: {self.num_iters}')
            # print(f'Size of del_x: {la.norm(delta_x * scale):.3f}')
            # print(f"Scale: 2^{np.log2(scale)}")
            # print(f'Time for Moore-Penrose Inversion: {(endInvTime - startInvTime).total_seconds():.3f}')
            # print(f'Time for processing total: {(startInvTime - startProcTime).total_seconds()}')
            # if self.numMeas > 2:
            # std = np.sqrt(np.diag(la.inv(L.T.dot(L))))
            # if self.haveAtLeastOneRecvMeas and self.haveAtLeastOneTankMeas:
            #     print(f"Std of RCam Location: {std[-6:-3]}")
            # else:
            #     print(f"Std of Cam Location: {std[-3:]}")
            # print("Size of L: ", QL.shape)
            # print("__________________________________________")
            # plt.spy(L)
            # plt.show()
        self.solution = SolutionData(self.numMeas)
        self.solution.ingest_xh(next_states)
        self.optComplete = True

        with open('./Caches/Interim.pkl', 'wb') as f:
            pkl.dump(self, f)

    def covariance(self):
        Q = self.create_Q()
        if self.L is not None:
            QL = Q.dot(self.L)
            return np.sqrt(np.diag(la.inv(QL.T.dot(QL))))
        else:
            return False

    def covarianceByVar(self):
        cov = self.covariance()
        r_T_d_cov = []
        r_V_d_cov = []
        r_A_d_cov = []

        for meas_num in range(self.numMeas):
            r_T_d_slice, r_V_d_slice, r_A_d_slice = self.stateFromTimestep(meas_num)

            r_T_d_cov.append(cov[r_T_d_slice])
            r_V_d_cov.append(cov[r_V_d_slice])
            r_A_d_cov.append(cov[r_A_d_slice])

        return [r_T_d_cov, r_V_d_cov, r_A_d_cov]

    def last_state_covariance(self):
        small_Q = self.create_Q()[-12:, -12:]
        small_L = self.create_L()[-12:, -9:]
        small_QL = small_Q.dot(small_L)
        cov = np.sqrt(np.diag(la.inv(small_QL.T.dot(small_QL))))
        return cov

    def predict(self,
                query_time,
                q_wr=None,
                R_wr=None,
                return_frame='receiver') -> PredictedState | None:
        if self.numMeas == 0:
            return None

        idx = -2 if self.numMeas >= 2 else -1

        if query_time is None:
            dt = 0.0
        else:
            query_time_rel = query_time - self.startTime
            dt = max(0.0, float(query_time_rel - self.time_log[idx]))

        # World/stabilized propagation
        w_T_d = (
                self.r_T_d[idx].copy()
                + dt * self.r_V_d[idx].copy()
                + 0.5 * (dt ** 2) * self.r_A_d[idx].copy()
        )
        w_V_d = self.r_V_d[idx].copy() + dt * self.r_A_d[idx].copy()
        w_A_d = self.r_A_d[idx].copy()

        out_T_d = w_T_d
        out_V_d = w_V_d
        out_A_d = w_A_d

        R_wr = self._as_rotmat(q_wr=q_wr, R_wr=R_wr)

        if return_frame.lower() in ('receiver', 'body', 'camera') and R_wr is not None:
            R_rw = R_wr.T
            out_T_d = R_rw @ w_T_d
            out_V_d = R_rw @ w_V_d
            out_A_d = R_rw @ w_A_d

        var_x = var_y = var_z = None
        try:
            cov = self.last_state_covariance()
            var_x = float(cov[0])
            var_y = float(cov[1])
            var_z = float(cov[2])
        except Exception:
            pass

        return PredictedState(
            query_time=query_time,
            state_time=self.startTime + float(self.time_log[idx]),
            dt=dt,
            r_T_d=out_T_d,
            r_V_d=out_V_d,
            r_A_d=out_A_d,
            var_x=var_x,
            var_y=var_y,
            var_z=var_z,
            is_extrapolated=(dt > 0.0),
        )

    def graphResults(self, gps=False, tspiFilename=None, tspiStartTime=None, tspiEndTime=None, true_r_V_d=None):

        t_log = self.time_log - self.time_log[0]

        r_T_d_meas = []
        r_T_d_meas_norm = []
        r_T_d_est = []
        r_T_d_est_norm = []

        for idx, meas in enumerate(self.meas):
            r_T_d_meas.append(meas)
            # r_T_d_est.append(self.r_T_d[idx])

        for r_T_d in r_T_d_meas:
            r_T_d_meas_norm.append(np.linalg.norm(r_T_d))

        r_T_d_cov, r_V_d_cov = self.covarianceByVar()
        est_Tnorm_low = []
        est_Tnorm_high = []
        est_Vnorm_low = []
        est_Vnorm_high = []
        for idx in range(self.numMeas):
            r_T_d_est_norm.append(np.linalg.norm(self.r_T_d[idx]))
            est_Tnorm_low.append(self.r_T_d[idx] - 3 * r_T_d_cov[idx])
            est_Tnorm_high.append(self.r_T_d[idx] + 3 * r_T_d_cov[idx])

            if idx < self.numMeas - 1:
                est_Vnorm_low.append(self.r_V_d[idx] - 3 * r_V_d_cov[idx])
                est_Vnorm_high.append(self.r_V_d[idx] + 3 * r_V_d_cov[idx])

        est_Tnorm_low = np.array(est_Tnorm_low)
        est_Tnorm_high = np.array(est_Tnorm_high)
        est_Vnorm_low = np.array(est_Vnorm_low)
        est_Vnorm_high = np.array(est_Vnorm_high)

        fig1, ax1 = plt.subplots(3)
        t = self.time_log
        r_T_d_meas = np.array(r_T_d_meas)
        for idx in range(3):
            ax1[idx].plot(t, r_T_d_meas[:, idx], label='Meas')
            ax1[idx].plot(t, self.r_T_d[:, idx], label='Est')

            ax1[idx].fill_between(t_log, est_Tnorm_low[:, idx], est_Tnorm_high[:, idx], alpha=0.3)

        plt.rcParams['text.usetex'] = True

        ax1[0].set_title(r'Factor-Graph Optimized Distance Estimations with 3$\sigma$')
        ax1[0].set_ylabel('Forward Distance(m)')
        ax1[1].set_ylabel('Lateral Distance(m)')
        ax1[2].set_ylabel('Vertical Distance(m)')
        ax1[2].set_xlabel('Scenario Time (s)')

        plt.legend()

        fig2, ax2 = plt.subplots(3)
        for idx in range(3):
            ax2[idx].plot(t, true_r_V_d[:, idx], label='True')
            ax2[idx].plot(t, self.r_V_d[:, idx], label='Est')

            ax2[idx].fill_between(t_log[:-1], est_Vnorm_low[:, idx], est_Vnorm_high[:, idx], alpha=0.3)

        ax2[0].set_title(r'Factor-Graph Optimized Velocity Estimations with 3$\sigma$')
        ax2[0].set_ylabel('Forward Velocity(m/s)')
        ax2[1].set_ylabel('Lateral Velocity(m/s)')
        ax2[2].set_ylabel('Vertical Velocity(m/s)')
        ax2[2].set_xlabel('Scenario Time (s)')
        plt.legend()

        plt.tight_layout(pad=0.5)
        # fig1.savefig("virtualSimResults.pdf", bbox_inches='tight')
        plt.show()


def utc_to_gps_time_of_week(utc_time):
    """Converts a UTC datetime object to GPS time of week (seconds)."""

    # Calculate the GPS epoch (January 6, 1980)
    gps_epoch = datetime.datetime(1980, 1, 6)

    # Correct for 18 second deviation
    gps_epoch += datetime.timedelta(0, -18)

    # Calculate the difference in seconds between the given UTC time and the GPS epoch
    time_difference = utc_time - gps_epoch

    # Calculate the GPS week number
    gps_week = int(time_difference.total_seconds() / 604800)  # 604800 seconds in a week

    # Calculate the GPS time of week (seconds)
    gps_tow = time_difference.total_seconds() % 604800

    return gps_week, gps_tow


def importNovatelData(filename: str, startStorageTow: float, endStorageTow: float) -> np.array:
    with open(filename, 'r') as f:
        string = f.read()

    stringList = string.split('\n')

    diff = []
    startTow = None
    for idx, strng in enumerate(stringList):
        subData = strng.split('\t')
        if idx > 2:

            try:
                tow = float(subData[0])
            except ValueError:
                return diff

            if startStorageTow <= tow <= endStorageTow:
                if startTow is None:
                    startTow = float(subData[0])

                diff.append([float(subData[0]), float(subData[4]), float(subData[6]), float(subData[8])])

    return diff
