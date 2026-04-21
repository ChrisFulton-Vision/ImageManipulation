# AttitudeInterpreter.py  (refactor: Pandas -> NumPy arrays)
from support.io.my_logging import LOG

import pandas as pd
import numpy as np
from os.path import join
from enum import Enum
from dataclasses import dataclass

class ControlMode(Enum):
    auto       = 'auto'
    manual     = 'manual'
    controller = 'controller'
    error      = 'error'

@dataclass(frozen=True)
class AttitudeSample:
    valid: bool
    time_s: float
    speed_mps: float
    altitude_m: float
    roll_deg: float
    cmd_roll_deg: float
    pitch_deg: float
    cmd_pitch_deg: float
    yaw_deg: float
    cmd_yaw_deg: float
    throttle_pct: float
    mode: ControlMode

    # --- GPS / minimap support ---
    gps_valid: bool = False
    lat_deg: float = 0.0
    lng_deg: float = 0.0
    gps_alt_m: float = 0.0
    gps_speed_mps: float = 0.0
    gps_ground_course_deg: float = 0.0
    gps_yaw_deg: float = 0.0

    # Local projected coordinates for minimap use.
    # These are NOT global meters, just a locally consistent flat projection.
    map_x: float = 0.0
    map_y: float = 0.0

    @property
    def rpy_deg(self) -> tuple[float, float, float]:
        return self.roll_deg, self.pitch_deg, self.yaw_deg

    def rotmat_wr(self) -> np.ndarray | None:
        if not self.valid:
            return None

        rr = np.deg2rad(self.roll_deg)
        rp = np.deg2rad(self.pitch_deg)
        ry = np.deg2rad(self.yaw_deg)

        cr, sr = np.cos(rr), np.sin(rr)
        cp, sp = np.cos(rp), np.sin(rp)
        cy, sy = np.cos(ry), np.sin(ry)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr,  cr],
        ], dtype=float)

        Ry = np.array([
            [ cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ], dtype=float)

        Rz = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        return Rz @ Ry @ Rx


class AttitudeReader:
    def __init__(self, csv_folder_path: str = None):

        # raw dfs (only used during load)
        self.spd_dict = None
        self.alt_dict = None
        self.roll_dict = None
        self.cmd_dict = None
        self.gps_dict = None

        # numpy caches
        self.spd_t = self.spd_v = None                    # ARSP.csv
        self.att_t = self.roll = self.desroll = None      # ATT.csv
        self.alt_t = self.alt = None                      # BARO.csv
        self.pitch = self.despitch = None                 # ATT.csv
        self.yaw = self.desyaw = None                     # ATT.csv
        self.cmd_t = self.c4 = self.c10 = None             # RCOU.csv
        self.cmd_throttle_perc = None                     # pre-mapped throttle %

        # GPS.csv
        self.gps_t = None
        self.gps_lat = None
        self.gps_lng = None
        self.gps_alt = None
        self.gps_spd = None
        self.gps_gc = None
        self.gps_yaw = None

        # Precomputed local-map coordinates (for minimap)
        self.gps_map_x = None
        self.gps_map_y = None
        self.gps_lat0_deg = 0.0
        self.gps_lng0_deg = 0.0
        self.gps_cos_lat0 = 1.0

        # Bounds in projected local map coordinates
        self.map_x_min = 0.0
        self.map_x_max = 0.0
        self.map_y_min = 0.0
        self.map_y_max = 0.0

        # Bounds in geodetic coordinates
        self.lat_min = 0.0
        self.lat_max = 0.0
        self.lng_min = 0.0
        self.lng_max = 0.0

        self.has_gps = False

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

        # GPS is optional for now
        try:
            self.gps_dict = pd.read_csv(join(csv_folder_path, 'GPS.csv'))
        except FileNotFoundError:
            self.gps_dict = None

        # validate columns
        if not {'timestamp', 'Airspeed'}.issubset(self.spd_dict.columns):
            print("ARSP.csv file not in expected format.")
            return False
        if not {'timestamp', 'Alt'}.issubset(self.alt_dict.columns):
            print("BARO.csv file not in expected format.")
            return False
        if not {'timestamp', 'Roll', 'DesRoll', 'Pitch', 'DesPitch', 'Yaw', 'DesYaw'}.issubset(self.roll_dict.columns):
            print("ATT.csv file not in expected format.")
            return False
        if not {'timestamp', 'C1', 'C5', 'C10'}.issubset(self.cmd_dict.columns):
            print("RCOU.csv file not in expected format.")
            return False

        if self.gps_dict is not None:
            if not {'timestamp', 'Lat', 'Lng'}.issubset(self.gps_dict.columns):
                print("GPS.csv file not in expected format. Ignoring GPS.")
                self.gps_dict = None

        # stable ascending time -> better for np.interp
        self.spd_dict = self.spd_dict.sort_values('timestamp').reset_index(drop=True)
        self.alt_dict = self.alt_dict.sort_values('timestamp').reset_index(drop=True)
        self.roll_dict = self.roll_dict.sort_values('timestamp').reset_index(drop=True)
        self.cmd_dict = self.cmd_dict.sort_values('timestamp').reset_index(drop=True)

        if self.gps_dict is not None:
            self.gps_dict = self.gps_dict.sort_values('timestamp').reset_index(drop=True)

        # --- Read or synthesize time offset as a DataFrame consistently ---
        try:
            offset_df = pd.read_csv(join(csv_folder_path, '__TIME_OFFSET.csv'))
            if 'offset' not in offset_df.columns:
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

        # ---- one-time conversion to NumPy ----
        self.spd_t = self.spd_dict['timestamp'].to_numpy(np.float64)
        self.spd_v = self.spd_dict['Airspeed'].to_numpy(np.float32)

        self.alt_t = self.alt_dict['timestamp'].to_numpy(np.float64)
        self.alt = self.alt_dict['Alt'].to_numpy(np.float32)

        self.att_t    = self.roll_dict['timestamp'].to_numpy(np.float64)
        self.roll     = self.roll_dict['Roll'].to_numpy(np.float32)
        self.desroll  = self.roll_dict['DesRoll'].to_numpy(np.float32)
        self.pitch    = self.roll_dict['Pitch'].to_numpy(np.float32)
        self.despitch = self.roll_dict['DesPitch'].to_numpy(np.float32)
        self.yaw      = self.roll_dict['Yaw'].to_numpy(np.float32)
        self.desyaw   = self.roll_dict['DesYaw'].to_numpy(np.float32)

        self.cmd_t = self.cmd_dict['timestamp'].to_numpy(np.float64)
        self.c5 = self.cmd_dict['C5'].to_numpy(np.float32)  # throttle pwm
        self.c10 = self.cmd_dict['C10'].to_numpy(np.float32)  # mode pwm
        self.cmd_throttle_perc = self.throttle_pwm_to_perc(self.c5).astype(np.float32)

        # --- GPS handling ---
        self.has_gps = False
        if self.gps_dict is not None and len(self.gps_dict) > 0:
            gps_t = self.gps_dict['timestamp'].to_numpy(np.float64)
            gps_lat = self.gps_dict['Lat'].to_numpy(np.float64)
            gps_lng = self.gps_dict['Lng'].to_numpy(np.float64)

            # Optional fields
            if 'Alt' in self.gps_dict.columns:
                gps_alt = self.gps_dict['Alt'].to_numpy(np.float32)
            else:
                gps_alt = np.zeros(len(self.gps_dict), dtype=np.float32)

            if 'Spd' in self.gps_dict.columns:
                gps_spd = self.gps_dict['Spd'].to_numpy(np.float32)
            else:
                gps_spd = np.zeros(len(self.gps_dict), dtype=np.float32)

            if 'GCrs' in self.gps_dict.columns:
                gps_gc = self.gps_dict['GCrs'].to_numpy(np.float32)
            else:
                gps_gc = np.zeros(len(self.gps_dict), dtype=np.float32)

            if 'Yaw' in self.gps_dict.columns:
                gps_yaw = self.gps_dict['Yaw'].to_numpy(np.float32)
            else:
                gps_yaw = np.zeros(len(self.gps_dict), dtype=np.float32)

            valid = (
                np.isfinite(gps_t) &
                np.isfinite(gps_lat) &
                np.isfinite(gps_lng)
            )
            valid &= (np.abs(gps_lat) > 1e-12) | (np.abs(gps_lng) > 1e-12)

            gps_t = gps_t[valid]
            gps_lat = gps_lat[valid]
            gps_lng = gps_lng[valid]
            gps_alt = gps_alt[valid]
            gps_spd = gps_spd[valid]
            gps_gc = gps_gc[valid]
            gps_yaw = gps_yaw[valid]

            if len(gps_t) >= 2:
                self.gps_t = gps_t
                self.gps_lat = gps_lat
                self.gps_lng = gps_lng
                self.gps_alt = gps_alt
                self.gps_spd = gps_spd
                self.gps_gc = gps_gc
                self.gps_yaw = gps_yaw

                self.lat_min = float(np.min(self.gps_lat))
                self.lat_max = float(np.max(self.gps_lat))
                self.lng_min = float(np.min(self.gps_lng))
                self.lng_max = float(np.max(self.gps_lng))

                self.gps_lat0_deg = float(np.mean(self.gps_lat))
                self.gps_lng0_deg = float(np.mean(self.gps_lng))
                self.gps_cos_lat0 = float(np.cos(np.deg2rad(self.gps_lat0_deg)))

                self.gps_map_x = (self.gps_lng - self.gps_lng0_deg) * self.gps_cos_lat0
                self.gps_map_y = (self.gps_lat - self.gps_lat0_deg)

                self.map_x_min = float(np.min(self.gps_map_x))
                self.map_x_max = float(np.max(self.gps_map_x))
                self.map_y_min = float(np.min(self.gps_map_y))
                self.map_y_max = float(np.max(self.gps_map_y))

                self.has_gps = True

        # free dataframes to reduce memory/GC churn
        self.spd_dict = None
        self.alt_dict = None
        self.roll_dict = None
        self.cmd_dict = None
        self.gps_dict = None

        self.ready = True
        return True

    def get_attitude_at(self, query_time) -> AttitudeSample:
        t = float(query_time) + self.offset

        if not self.ready:
            return AttitudeSample(
                valid=False,
                time_s=t,
                speed_mps=180.0,
                altitude_m=0.0,
                roll_deg=0.0,
                cmd_roll_deg=180.0,
                pitch_deg=0.0,
                cmd_pitch_deg=0.0,
                yaw_deg=0.0,
                cmd_yaw_deg=0.0,
                throttle_pct=0.0,
                mode=ControlMode.error,
                gps_valid=False,
            )

        if t < self.att_t[0] or t > self.att_t[-1]:
            return AttitudeSample(
                valid=False,
                time_s=t,
                speed_mps=180.0,
                altitude_m=0.0,
                roll_deg=0.0,
                cmd_roll_deg=180.0,
                pitch_deg=0.0,
                cmd_pitch_deg=0.0,
                yaw_deg=0.0,
                cmd_yaw_deg=0.0,
                throttle_pct=0.0,
                mode=ControlMode.error,
                gps_valid=False,
            )

        # all-NumPy interpolation
        spd = np.interp(t, self.spd_t, self.spd_v)
        alt = np.interp(t, self.alt_t, self.alt)
        roll = np.interp(t, self.att_t, self.roll)
        cmd_roll = np.interp(t, self.att_t, self.desroll)
        pitch = np.interp(t, self.att_t, self.pitch)
        cmd_pitch = np.interp(t, self.att_t, self.despitch)
        yaw = np.interp(t, self.att_t, self.yaw)
        cmd_yaw = np.interp(t, self.att_t, self.desyaw)
        thr_perc = np.interp(t, self.cmd_t, self.cmd_throttle_perc)
        mode = self.ch10_pwm_to_mode(np.interp(t, self.cmd_t, self.c10))

        gps_valid = False
        lat_deg = 0.0
        lng_deg = 0.0
        gps_alt_m = 0.0
        gps_speed_mps = 0.0
        gps_ground_course_deg = 0.0
        gps_yaw_deg = 0.0
        map_x = 0.0
        map_y = 0.0

        if self.has_gps and self.gps_t is not None:
            if self.gps_t[0] <= t <= self.gps_t[-1]:
                lat_deg = float(np.interp(t, self.gps_t, self.gps_lat))
                lng_deg = float(np.interp(t, self.gps_t, self.gps_lng))
                gps_alt_m = float(np.interp(t, self.gps_t, self.gps_alt))
                gps_speed_mps = float(np.interp(t, self.gps_t, self.gps_spd))
                gps_ground_course_deg = float(np.interp(t, self.gps_t, self.gps_gc))
                gps_yaw_deg = float(np.interp(t, self.gps_t, self.gps_yaw))

                map_x = float((lng_deg - self.gps_lng0_deg) * self.gps_cos_lat0)
                map_y = float(lat_deg - self.gps_lat0_deg)
                gps_valid = True

        return AttitudeSample(
            valid=True,
            time_s=float(t),
            speed_mps=float(spd),
            altitude_m=float(alt),
            roll_deg=float(roll),
            cmd_roll_deg=float(cmd_roll),
            pitch_deg=float(pitch),
            cmd_pitch_deg=float(cmd_pitch),
            yaw_deg=float(yaw),
            cmd_yaw_deg=float(cmd_yaw),
            throttle_pct=float(thr_perc),
            mode=mode,
            gps_valid=gps_valid,
            lat_deg=lat_deg,
            lng_deg=lng_deg,
            gps_alt_m=gps_alt_m,
            gps_speed_mps=gps_speed_mps,
            gps_ground_course_deg=gps_ground_course_deg,
            gps_yaw_deg=gps_yaw_deg,
            map_x=map_x,
            map_y=map_y,
        )

    @staticmethod
    def ch10_pwm_to_mode(ch8):
        if 950 < ch8 < 1250:
            return ControlMode.manual
        if 1250 <= ch8 < 1750:
            return ControlMode.auto
        if 1750 <= ch8 < 2050:
            return ControlMode.controller
        return ControlMode.error

    @staticmethod
    def throttle_pwm_to_perc(throttle_pwm: np.ndarray) -> np.ndarray:
        MIN_THROTTLE = 1000.0
        MAX_THROTTLE = 1935.0
        return (throttle_pwm - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE) * 100.0