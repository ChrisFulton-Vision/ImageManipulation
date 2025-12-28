# AttitudeInterpreter.py  (refactor: Pandas -> NumPy arrays)
from support.io.my_logging import LOG

import pandas as pd
import numpy as np
from os.path import join
from enum import Enum

class ControlMode(Enum):
    auto       = 'auto'
    manual     = 'manual'
    controller = 'controller'
    error      = 'error'

class AttitudeReader:
    def __init__(self, csv_folder_path: str = None):

        # raw dfs (only used during load)
        self.spd_dict = None
        self.alt_dict = None
        self.roll_dict = None
        self.cmd_dict = None

        # numpy caches
        self.spd_t = self.spd_v = None                  # ARSP.csv
        self.att_t = self.roll = self.desroll = None    # ATT.csv
        self.alt_t = self.alt = None                    # BARO.csv
        self.pitch = self.despitch = None               # ATT.csv
        self.cmd_t = self.c3 = self.c8 = None           # RCOU.csv
        self.cmd_throttle_perc = None                   # pre-mapped throttle %

        self.offset = 0.0
        self.ready = False
        if csv_folder_path is not None:
            self.read_files(csv_folder_path)

    def read_files(self, csv_folder_path: str):

        try:
            self.spd_dict  = pd.read_csv(join(csv_folder_path, 'ARSP.csv'))
            self.alt_dict  = pd.read_csv(join(csv_folder_path, 'BARO.csv'))
            self.roll_dict = pd.read_csv(join(csv_folder_path, 'ATT.csv'))
            self.cmd_dict  = pd.read_csv(join(csv_folder_path, 'RCOU.csv'))
        except FileNotFoundError:
            LOG.info("Error. Aircraft Log datafile not found")
            return False

        # validate columns (same checks you had)
        if not {'timestamp', 'Airspeed'}.issubset(self.spd_dict.columns):
            print("ARSP.csv file not in expected format.")
            return False
        if not {'timestamp', 'Alt'}.issubset(self.alt_dict.columns):
            print("BARO.csv file not in expected format.")
            return False
        if not {'timestamp', 'Roll', 'DesRoll', 'Pitch', 'DesPitch'}.issubset(self.roll_dict.columns):
            print("ATT.csv file not in expected format.")
            return False
        if not {'timestamp', 'C1', 'C3', 'C8'}.issubset(self.cmd_dict.columns):
            print("RCOU.csv file not in expected format.")
            return False

        # stable ascending time -> better for np.interp
        self.spd_dict = self.spd_dict.sort_values('timestamp').reset_index(drop=True)
        self.alt_dict = self.alt_dict.sort_values('timestamp').reset_index(drop=True)
        self.roll_dict = self.roll_dict.sort_values('timestamp').reset_index(drop=True)
        self.cmd_dict = self.cmd_dict.sort_values('timestamp').reset_index(drop=True)

        # --- Read or synthesize time offset as a DataFrame consistently ---
        try:
            offset_df = pd.read_csv(join(csv_folder_path, '__TIME_OFFSET.csv'))
            # Be forgiving about column naming
            if 'offset' not in offset_df.columns:
                # Try common alternatives; add your own as needed
                for cand in ('time_offset', 'Offset', 'OFFSET'):
                    if cand in offset_df.columns:
                        offset_df = offset_df.rename(columns={cand: 'offset'})
                        break
            if 'offset' not in offset_df.columns:
                raise ValueError("__TIME_OFFSET.csv missing required 'offset' column")
        except FileNotFoundError:
            print("No __TIME_OFFSET.csv found; defaulting offset to first ARSP timestamp.")
            t0 = float(self.spd_dict['timestamp'].iloc[0])
            offset_df = pd.DataFrame({'offset': [t0]})
        except Exception as e:
            print("Unexpected error while reading __TIME_OFFSET.csv:\n", e)
            return False

        self.offset = float(offset_df['offset'][0])

        # ---- one-time conversion to NumPy (choose dtypes deliberately) ----
        # timestamps as float64 (interp domain), signals as float32 (fast + compact)
        self.spd_t = self.spd_dict['timestamp'].to_numpy(np.float64)
        self.spd_v = self.spd_dict['Airspeed' ].to_numpy(np.float32)

        self.alt_t = self.alt_dict['timestamp'].to_numpy(np.float64)
        self.alt = self.alt_dict['Alt'].to_numpy(np.float32)

        self.att_t    = self.roll_dict['timestamp'].to_numpy(np.float64)
        self.roll     = self.roll_dict['Roll'    ].to_numpy(np.float32)
        self.desroll  = self.roll_dict['DesRoll' ].to_numpy(np.float32)
        self.pitch    = self.roll_dict['Pitch'   ].to_numpy(np.float32)
        self.despitch = self.roll_dict['DesPitch'].to_numpy(np.float32)

        self.cmd_t = self.cmd_dict['timestamp'].to_numpy(np.float64)
        # self.c1 = self.cmd_dict['C1'].to_numpy(np.float32)
        self.c3 = self.cmd_dict['C3'].to_numpy(np.float32)  # throttle pwm
        self.c8 = self.cmd_dict['C8'].to_numpy(np.float32)  # mode pwm

        # pre-map throttle → percent now so per-frame work is only one interp
        self.cmd_throttle_perc = self.throttle_pwm_to_perc(self.c3).astype(np.float32)

        # free dataframes to reduce memory/GC churn
        self.spd_dict = self.roll_dict = self.cmd_dict = None

        self.ready = True
        return True

    def get_attitude_at(self, query_time):

        t = float(query_time) + self.offset

        # print(f'Query_time: {query_time}')
        # print(f'Offset: {self.offset}')
        # print(f't: {t}\nZero: {self.att_t[0]}\nMax: {self.att_t[-1]}\n\n')

        if not self.ready:
            return 180.0, 0.0, 0.0, 180.0, 0.0, 0.0, 0.0, False


        # fast O(1) bound checks using NumPy arrays
        if t < self.att_t[0] or t > self.att_t[-1]:
            return 180.0, 0.0, 0.0, 180.0, 0.0, 0.0, 0.0, False

        # all-NumPy interpolation (x arrays are strictly ascending)
        spd        = np.interp(t, self.spd_t, self.spd_v)
        alt        = np.interp(t, self.alt_t, self.alt)
        roll       = np.interp(t, self.att_t, self.roll)
        cmd_roll   = np.interp(t, self.att_t, self.desroll)
        pitch      = np.interp(t, self.att_t, self.pitch)
        cmd_pitch  = np.interp(t, self.att_t, self.despitch)
        thr_perc   = np.interp(t, self.cmd_t, self.cmd_throttle_perc)  # already mapped to %

        # mode_pwm   = np.interp(t, self.cmd_t, self.c8)
        mode       = self.ch8_pwm_to_mode(np.interp(t, self.cmd_t, self.c8))

        return spd, alt, roll, cmd_roll, pitch, cmd_pitch, float(thr_perc), mode

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
        MIN_THROTTLE = 1000.0  # 1300
        MAX_THROTTLE = 1935.0  # 1880
        return (throttle_pwm - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE) * 100.0
