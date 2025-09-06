import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
from numpy import linalg as la
import copy
import pickle as pkl

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

class SolutionData:
    def __init__(self, numMeas):
        self.numMeas = numMeas
        self.drg_pos = np.zeros((numMeas,3))
        self.drg_vel = np.zeros((numMeas,3))

        self.r_T_d = np.zeros((3,))
        self.r_V_d = np.zeros((3,))

        self.populated = False

    def ingest_xh(self, states) -> None :

        self.r_T_d, self.r_V_d = states

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

        self.r_T_d = np.zeros((1,3))
        self.r_V_d = np.zeros((1,3))

        self.meas = np.zeros((1,))
        self.curr_meas = np.zeros((1,))

        self.L = np.zeros((1,1))
        self.y = np.zeros(1, )

        self.init_guess = np.zeros(1, )
        self.init_residual = np.zeros((1,))
        self.num_iters = 0

        # self.cam_cov = np.array([0.6, 5.0, 5.0]) * 4.0 ** 2
        # self.V_cov = np.array([1.0, 1.0, 1.0]) * 2.0 ** 2
        # self.Vdot_cov = np.array([1.0, 1.0, 1.0]) * 1.0 ** 2
        self.cam_cov = np.array([0.6, 5.0, 5.0]) * (4.0 ** 2) / 20.0
        self.V_cov = np.array([1.0, 1.0, 1.0]) * (2.0 ** 2) / 20.0
        self.Vdot_cov = np.array([1.0, 1.0, 1.0]) * (2.0 ** 2) / 20.0

        self.numMeas = 0

        if startT is None:
            epoch = datetime.datetime(1970,1,1)
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

    def newRecvMeas(self, drgVec, t=None):
        self.optComplete = False
        # Realign measurements to:
        # 0: Drogue from (recv or tanker) Camera

        if self.numMeas == 0:

            self.curr_meas = [drgVec]

            self.meas = self.curr_meas

            self.r_T_d[0, :] = drgVec
            self.r_V_d = np.zeros((1,3))

            if t is None:
                epoch = datetime.datetime(1970, 1, 1)
                self.time_log[0] = (datetime.datetime.now() - epoch).total_seconds() - self.startTime
            else:
                self.time_log[0] = t - self.startTime

        else:
            self.curr_meas = drgVec

            if t is None:
                epoch = datetime.datetime(1970, 1, 1)
                t = (datetime.datetime.now() - epoch).total_seconds() - self.startTime
                self.time_log = np.append(self.time_log, t)
            else:
                t = t - self.startTime
                self.time_log = np.append(self.time_log, t)

            
            delT = self.time_log[-1] - self.time_log[-2]

            if self.time_log[-1] <= self.time_log[-2]:
                self.reset()
                self.newRecvMeas(drgVec, t)
                return

            self.meas.append(self.curr_meas)
            self.r_T_d = np.append(self.r_T_d, drgVec[np.newaxis, :], axis=0)  # Drg from Recv
            self.r_V_d = np.append(self.r_V_d, ((self.r_T_d[-1] - self.r_T_d[-2]) / delT)[np.newaxis, :], axis=0)

        self.numMeas += 1

    def create_Q(self):
        N_y = (self.numMeas - 1) * 9 + 3
        Q = np.eye(N_y,N_y)

        for idx in range(self.numMeas):
            iter = idx * 9
            cam_T_d = self.meas[idx]
            Q[iter + 0: iter + 3, iter + 0: iter + 3] = 1.0 / la.norm(cam_T_d) * np.diag(self.cam_cov)

            if idx < self.numMeas - 1:
                Q[iter + 3: iter + 6, iter + 3: iter + 6] = np.diag(self.V_cov)
                Q[iter + 6: iter + 9, iter + 6: iter + 9] = np.diag(self.Vdot_cov)

        return Q

    def create_y(self, states=None):
        '''

        :param states: If states exists, it should be a list of five nparrays: [r_T_d, r_V_d, r_T_t, r_V_t, r_Q_t]
        :param cameras: If cameras exists, it should be a list of two nparrays: [r_T_rc, t_T_tc]
        :return: residual vector, y
        '''

        # y is the residual based on each measurement and dynamic model. There are
        # three 3d measurements in each set (of the drogue, recv, and tanker). For every
        # consecutive pair of measurements, there is 2 additional 3d velocities that can be inferred
        # representing relative motion between drogue and receiver and relative motion between tanker and receiver.
        # We can finally assume 2 more 3d connections between previous and current velocities.
        # This implies 9 initial residual values, with 9+12=21 additional values for each measurement set.
        N_y = (self.numMeas - 1) * 9 + 3

        if states is None:
            states_r_T_d = copy.copy(self.r_T_d)
            states_r_V_d = copy.copy(self.r_V_d)
        else:
            states_r_T_d, states_r_V_d = states


        y = np.zeros(N_y)

        for meas_num in range(self.numMeas):

            iter_Base = 9 * meas_num

            # States ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # [r_T_rc, t_T_tc]
            s_drgPos = states_r_T_d[meas_num]
            s_drgVel = states_r_V_d[meas_num]

            # Direct Measurement Residuals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            delta_t, dyn_t, dyn_v = self.resFromTimestep(meas_num)

            y[delta_t] = self.meas[meas_num] - s_drgPos

            if meas_num < self.numMeas - 1:
                # Dynamics Residuals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                s_nextDrgPos = states_r_T_d[meas_num + 1]
                s_nextDrgVel = states_r_V_d[meas_num + 1]

                deltT = self.time_log[meas_num + 1] - self.time_log[meas_num]

                # Dynamics equations:
                y[dyn_t] = s_drgVel - (s_nextDrgPos - s_drgPos) / deltT
                y[dyn_v] = (s_nextDrgVel - s_drgVel) / deltT

        return y

    def reset(self):
        self.r_T_d = np.zeros((1, 3))
        self.r_V_d = np.zeros((1, 3))
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
        elif self.numMeas <= 1:
            self.reset()

    def stateFromTimestep(self, meas_idx):
        # s: r_T_d, r_V_d
        return slice(meas_idx * 6 + 0, meas_idx * 6 + 3), \
               slice(meas_idx * 6 + 3, meas_idx * 6 + 6)

    def measFromTimestep(self, meas_idx):
        # m: r_T_d
        return slice(meas_idx * 3 + 0, meas_idx * 3 + 3)

    def resFromTimestep(self, meas_idx):
        # res eq: t(n+1) - t(n), v(n) - (t(n+1) - t(n))/delT, (v(n+1) - v(n)) / delT
        # simply delta_t, dyn_t, dyn_v
        return slice(meas_idx * 9 + 0, meas_idx * 9 + 3), \
               slice(meas_idx * 9 + 3, meas_idx * 9 + 6), \
               slice(meas_idx * 9 + 6, meas_idx * 9 + 9)

    def create_L(self):

        L = np.zeros((self.numMeas * 9 - 6, self.numMeas * 6))

        for meas_num in range(self.numMeas):
            rowStartID = meas_num * 9
            colStartID = meas_num * 6

            ## MEASUREMENT ACCOUNTING
            # States ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            sID_r_T_d, sID_r_V_d = self.stateFromTimestep(meas_num)
            # Measurements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            mID_r_T_d = self.measFromTimestep(meas_num)
            # Residual Equations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            delta_t, dyn_t, dyn_v = self.resFromTimestep(meas_num)

            # y[delta_t] = self.meas[meas_num] - s_drgPos
            L[delta_t, sID_r_T_d ] = -np.eye(3)

            if meas_num < self.numMeas - 1:
                ## DYNAMICS ACCOUNTING
                sID_r_T_d_next, sID_r_V_d_next = self.stateFromTimestep(meas_num+1)
                deltT = self.time_log[meas_num + 1] - self.time_log[meas_num]

                # y[self.rID_r_V_d(meas_num)] = s_drgVel - (s_nextDrgPos - s_drgPos) / deltT
                L[dyn_t, sID_r_V_d] =       np.eye(3)
                L[dyn_t, sID_r_T_d] =       np.eye(3) / deltT
                L[dyn_t, sID_r_T_d_next] = -np.eye(3) / deltT

                # y[dyn_v] = (s_nextDrgVel - s_drgVel) / deltT
                L[dyn_v, sID_r_V_d_next] = np.eye(3) / deltT
                L[dyn_v, sID_r_V_d] = -np.eye(3) / deltT

        # plt.spy(L)
        # plt.show()

        return L

    def update_states(self, states):
        # [r_T_d, r_V_d]
        self.r_T_d, self.r_V_d = states

    def calc_next_states(self, delta_x):

        # [r_T_d, r_V_d, r_T_t, r_V_t, r_Q_t]
        r_T_d = copy.deepcopy(self.r_T_d)
        r_V_d = copy.deepcopy(self.r_V_d)

        for meas_num in range(self.numMeas):
            r_T_d_slice, r_V_d_slice = self.stateFromTimestep(meas_num)

            r_T_d[meas_num] -= delta_x[r_T_d_slice]

            if meas_num < self.numMeas - 1:
                r_V_d[meas_num] -= delta_x[r_V_d_slice]

        r_V_d[-1] = r_V_d[-2]
        # [r_T_d, r_V_d]
        return [r_T_d, r_V_d]


    def opt(self, func=None):
        if self.time_log[-1] <= self.time_log[-2]:
            self.reset()
            return

        self.num_iters = 0
        prev_ratio = np.inf
        keep_going = True
        stop = False
        Q = self.create_Q()
        # Q = np.eye(len(self.create_y()))
        scale = 1.0
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

        with open('Caches/Interim.pkl', 'wb') as f:
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

        for meas_num in range(self.numMeas):
            # s: r_T_d, r_T_t, r_V_d, r_V_t, r_Q_t
            r_T_d_slice, r_V_d_slice = self.stateFromTimestep(meas_num)

            r_T_d_cov.append(cov[r_T_d_slice])

            if meas_num < self.numMeas - 1:
                r_V_d_cov.append(cov[r_V_d_slice])

        return [r_T_d_cov, r_V_d_cov]

    def last_pos_covariance(self):
        small_Q = self.create_Q()[-6:,-6:]
        small_L = self.create_L()[-6:,-6:]
        small_QL = small_Q.dot(small_L)
        cov = np.sqrt(np.diag(la.inv(small_QL.T.dot(small_QL))))
        return cov


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

            if idx < self.numMeas-1:
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
            ax1[idx].plot(t, r_T_d_meas[:,idx], label='Meas')
            ax1[idx].plot(t, self.r_T_d[:,idx], label='Est')

            ax1[idx].fill_between(t_log, est_Tnorm_low[:,idx], est_Tnorm_high[:,idx], alpha=0.3)

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
            ax2[idx].plot(t, self.r_V_d[:,idx], label='Est')

            ax2[idx].fill_between(t_log[:-1], est_Vnorm_low[:,idx], est_Vnorm_high[:,idx], alpha=0.3)

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