# AttitudeInterpreter.py  (refactor: Pandas -> NumPy arrays)
import pandas as pd
import numpy as np
from os.path import join
from enum import Enum

class ControlMode(Enum):
    auto = 'auto'
    manual = 'manual'
    controller = 'controller'
    error = 'error'

class AttitudeReader:
    def __init__(self, csv_folder_path: str = None):
        # raw dfs (only used during load)
        self.spd_dict = None
        self.roll_dict = None
        self.cmd_dict = None

        # numpy caches
        self.spd_t = self.spd_v = None                  # ARSP.csv
        self.att_t = self.roll = self.desroll = None    # ATT.csv
        self.pitch = self.despitch = None
        self.cmd_t = self.c1 = self.c3 = self.c8 = None # RCOU.csv
        self.cmd_throttle_perc = None                   # pre-mapped throttle %

        self.offset = 0.0
        self.ready = False
        if csv_folder_path is not None:
            self.read_files(csv_folder_path)

    def read_files(self, csv_folder_path: str):
        try:
            self.spd_dict  = pd.read_csv(join(csv_folder_path, 'ARSP.csv'))
            self.roll_dict = pd.read_csv(join(csv_folder_path, 'ATT.csv'))
            self.cmd_dict  = pd.read_csv(join(csv_folder_path, 'RCOU.csv'))
            offset_dict    = pd.read_csv(join(csv_folder_path, '__TIME_OFFSET.csv'))
        except FileNotFoundError:
            return False

        # validate columns (same checks you had)
        if not {'timestamp', 'Airspeed'}.issubset(self.spd_dict.columns): return False
        if not {'timestamp', 'Roll', 'DesRoll', 'Pitch', 'DesPitch'}.issubset(self.roll_dict.columns): return False
        if not {'timestamp', 'C1', 'C3', 'C8'}.issubset(self.cmd_dict.columns): return False

        self.offset = float(offset_dict['offset'][0])

        # stable ascending time -> better for np.interp
        self.spd_dict  = self.spd_dict.sort_values('timestamp').reset_index(drop=True)
        self.roll_dict = self.roll_dict.sort_values('timestamp').reset_index(drop=True)
        self.cmd_dict  = self.cmd_dict.sort_values('timestamp').reset_index(drop=True)

        # ---- one-time conversion to NumPy (choose dtypes deliberately) ----
        # timestamps as float64 (interp domain), signals as float32 (fast + compact)
        self.spd_t = self.spd_dict['timestamp'].to_numpy(np.float64)
        self.spd_v = self.spd_dict['Airspeed' ].to_numpy(np.float32)

        self.att_t    = self.roll_dict['timestamp'].to_numpy(np.float64)
        self.roll     = self.roll_dict['Roll'    ].to_numpy(np.float32)
        self.desroll  = self.roll_dict['DesRoll' ].to_numpy(np.float32)
        self.pitch    = self.roll_dict['Pitch'   ].to_numpy(np.float32)
        self.despitch = self.roll_dict['DesPitch'].to_numpy(np.float32)

        self.cmd_t = self.cmd_dict['timestamp'].to_numpy(np.float64)
        self.c1    = self.cmd_dict['C1'].to_numpy(np.float32)
        self.c3    = self.cmd_dict['C3'].to_numpy(np.float32)  # throttle pwm
        self.c8    = self.cmd_dict['C8'].to_numpy(np.float32)  # mode pwm

        # pre-map throttle → percent now so per-frame work is only one interp
        self.cmd_throttle_perc = self.throttle_pwm_to_perc(self.c3).astype(np.float32)

        # free dataframes to reduce memory/GC churn
        self.spd_dict = self.roll_dict = self.cmd_dict = None

        self.ready = True
        return True

    def get_attitude_at(self, query_time):
        if not self.ready:
            return 180.0, 0.0, 180.0, 0.0, 0.0, False

        t = float(query_time) + self.offset

        # print(f'Query_time: {query_time}\nOffset: {self.offset}\nt: {t}\nZero: {self.att_t[0]}\nMax: {self.att_t[-1]}\n\n')
        
        # fast O(1) bound checks using NumPy arrays
        if t < self.att_t[0] or t > self.att_t[-1]:
            return 180.0, 0.0, 180.0, 0.0, 0.0, False

        # all-NumPy interpolation (x arrays are strictly ascending)
        spd        = np.interp(t, self.spd_t, self.spd_v)
        roll       = np.interp(t, self.att_t, self.roll)
        cmd_roll   = np.interp(t, self.att_t, self.desroll)
        pitch      = np.interp(t, self.att_t, self.pitch)
        cmd_pitch  = np.interp(t, self.att_t, self.despitch)
        thr_perc   = np.interp(t, self.cmd_t, self.cmd_throttle_perc)  # already mapped to %

        # mode_pwm   = np.interp(t, self.cmd_t, self.c8)
        mode       = self.ch8_pwm_to_mode(np.interp(t, self.cmd_t, self.c8))

        return roll, cmd_roll, pitch, cmd_pitch, float(thr_perc), mode

    @staticmethod
    def ch8_pwm_to_mode(ch8):
        if (950 < ch8 < 1250):
            return ControlMode.controller
        if (1250 <= ch8 < 1750):
            return ControlMode.auto
        if (1750 <= ch8 < 2050):
            return ControlMode.manual
        else:
            return ControlMode.error


    @staticmethod
    def throttle_pwm_to_perc(throttle_pwm: np.ndarray) -> np.ndarray:
        # same mapping, vectorized
        return (throttle_pwm - 1300.0) / (1880.0 - 1330.0) * 100.0
