# AttitudeInterpreter.py
from support.io.my_logging import LOG
from support.io.image_time_reader import _parse_log_timestamp_utc_seconds
from support.core.enums import ControlMode

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from os.path import join
from dataclasses import dataclass

_EARTH_RADIUS_M = 6378137.0

# Hard-coded runway corners for initial integration.
# Order should follow the perimeter so HUD drawing can close the quadrilateral.
_RUNWAY_CORNERS_LLA = (
    (39.344333, -86.009638, 219.0),  #NW
    (39.344341, -86.009465, 219.0),  #NE
    (39.341582, -86.009378, 214.0),  #SE
    (39.341578, -86.009557, 214.0),  #SW
)

# Camera origin relative to the aircraft/body origin in body axes [forward, right, down], meters.
# Positive right moves the camera to starboard.
CAMERA_LEVER_ARM_BODY_M = (0.0, 0.0, 0.0)

# Camera angular offset relative to the aircraft/body axes in [roll, pitch, yaw] degrees.
CAMERA_RPY_OFFSET_DEG = (0.0, 0.0, 1.0)


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

    home_x: float = 0.0
    home_y: float = 0.0

    # Runway corners in aircraft/body coordinates [forward, right, down], meters.
    runway_corners_body_m: tuple[tuple[float, float, float], ...] | None = None

    # Runway corners in local minimap coordinates [map_x, map_y], meters.
    runway_corners_map_m: tuple[tuple[float, float], ...] | None = None

    # Axis-aligned runway bounds in local minimap coordinates [x_min, y_min, x_max, y_max], meters.
    runway_rect_map_m: tuple[float, float, float, float] | None = None

    @property
    def rpy_deg(self) -> tuple[float, float, float]:
        return self.roll_deg, self.pitch_deg, self.yaw_deg

    def rotmat_wr(self) -> NDArray | None:
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
            [cp,  0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ], dtype=float)

        Rz = np.array([
            [cy, -sy,  0.0],
            [sy,  cy,  0.0],
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
        self.cmd_t = self.c4 = self.c5 = self.c10 = None  # RCOU.csv
        self.cmd_throttle_perc = None                     # pre-mapped throttle %

        # GPS.csv
        self.gps_t = None
        self.gps_lat = None
        self.gps_lng = None
        self.gps_alt = None
        self.gps_spd = None
        self.gps_gc = None
        self.gps_yaw = None
        self.gps_gc_unwrapped = None
        self.gps_yaw_unwrapped = None

        # Precomputed local-map coordinates (for minimap)
        self.gps_map_x = None
        self.gps_map_y = None
        self.gps_home_x = None
        self.gps_home_y = None
        self.gps_east_m = None
        self.gps_north_m = None
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
        self.runway_corners_map_m = None
        self.runway_rect_map_m = None

        self.offset = 0.0
        self.ready = False
        if csv_folder_path is not None:
            self.read_files(csv_folder_path)

    def read_files(self,
                   csv_folder_path: str,
                   img_folder_path: str | None = None,
                   update_time_offset_func=None):

        def first_image_time_from_folder(path_like: str | None) -> float | None:
            if not path_like:
                return None

            from pathlib import Path

            path = Path(path_like)
            folder = path if path.is_dir() else path.parent
            try:
                log_path = next(folder.glob("*.log"))
            except StopIteration:
                return None

            try:
                with open(log_path, newline='') as file_obj:
                    for line in file_obj:
                        row = line.strip()
                        if not row:
                            continue
                        first_token = row.split()[0]
                        if first_token == '#':
                            continue
                        return _parse_log_timestamp_utc_seconds(first_token, 7)
            except OSError as exc:
                LOG.warning("Failed to read image log '%s': %s", log_path, exc)
            return None

        def read_from_csv(filename: str):
            try:
                return pd.read_csv(join(csv_folder_path, filename))
            except FileNotFoundError:
                LOG.info("Error. Aircraft Log datafile not found")
                raise FileNotFoundError
        try:
            self.spd_dict = read_from_csv('ARSP.csv')
            self.alt_dict = read_from_csv('BARO.csv')
            self.roll_dict = read_from_csv('ATT.csv')
            self.cmd_dict = read_from_csv('RCIN.csv')
            self.gps_dict = read_from_csv('GPS.csv')
        except FileNotFoundError:
            return False

        # validate columns
        if not {'timestamp', 'Airspeed'}.issubset(self.spd_dict.columns):
            LOG.warning("ARSP.csv file not in expected format.")
            return False
        if not {'timestamp', 'Alt'}.issubset(self.alt_dict.columns):
            LOG.warning("BARO.csv file not in expected format.")
            return False
        if not {'timestamp', 'Roll', 'DesRoll', 'Pitch', 'DesPitch', 'Yaw', 'DesYaw'}.issubset(self.roll_dict.columns):
            LOG.warning("ATT.csv file not in expected format.")
            return False
        if not {'timestamp', 'C1', 'C5', 'C10'}.issubset(self.cmd_dict.columns):
            LOG.warning("RCOU.csv file not in expected format.")
            return False
        if not {'timestamp', 'Lat', 'Lng'}.issubset(self.gps_dict.columns):
            LOG.warning("GPS.csv file not in expected format. Ignoring GPS.")
            return False

        # stable ascending time -> better for np.interp
        self.spd_dict = self.spd_dict.sort_values('timestamp').reset_index(drop=True)
        self.alt_dict = self.alt_dict.sort_values('timestamp').reset_index(drop=True)
        self.roll_dict = self.roll_dict.sort_values('timestamp').reset_index(drop=True)
        self.cmd_dict = self.cmd_dict.sort_values('timestamp').reset_index(drop=True)
        self.gps_dict = self.gps_dict.sort_values('timestamp').reset_index(drop=True)

        first_sensor_time = float(self.spd_dict['timestamp'].iloc[0])
        first_image_time = first_image_time_from_folder(img_folder_path)

        # With absolute image timestamps, offset maps image time to sensor/log time.
        should_persist_offset = False

        try:
            offset_df = read_from_csv('__TIME_OFFSET.csv')
            if 'offset' not in offset_df.columns:
                for cand in ('time_offset', 'Offset', 'OFFSET'):
                    if cand in offset_df.columns:
                        offset_df = offset_df.rename(columns={cand: 'offset'})
                        break
            if 'offset' not in offset_df.columns:
                raise ValueError("__TIME_OFFSET.csv missing required 'offset' column")

            raw_offset = float(offset_df['offset'][0])
            if first_image_time is not None:
                err_as_new = abs((first_image_time + raw_offset) - first_sensor_time)
                err_as_legacy = abs(raw_offset - first_sensor_time)
                if err_as_legacy + 1e-6 < err_as_new:
                    converted_offset = raw_offset - first_image_time
                    offset_df = pd.DataFrame({'offset': [converted_offset]})
                    should_persist_offset = True
                    LOG.info(
                        "Converted legacy __TIME_OFFSET %.6f to absolute-image offset %.6f.",
                        raw_offset,
                        converted_offset,
                    )
        except FileNotFoundError:
            offset_df = None

        except Exception as e:
            LOG.warning(f"Unexpected error while reading __TIME_OFFSET.csv:\n{e}")
            return False

        if offset_df is None:
            if first_image_time is not None:
                inferred_offset = first_sensor_time - first_image_time
                offset_df = pd.DataFrame({'offset': [inferred_offset]})
                should_persist_offset = True
                LOG.info(
                    "No __TIME_OFFSET.csv found; aligning first image time to first log time with offset %.6f.",
                    inferred_offset,
                )
            else:
                LOG.info("No __TIME_OFFSET.csv found and no image log available; defaulting offset to 0.0s.")
                offset_df = pd.DataFrame({'offset': [0.0]})

        self.offset = float(offset_df['offset'][0])
        if should_persist_offset and update_time_offset_func is not None:
            update_time_offset_func(self.offset, csv_folder_path)

        # ---- one-time conversion to NumPy ----
        def convert_to_numpy(conversion_dict_subset, np_type=np.float32) -> NDArray | None:
            if conversion_dict_subset is not None:
                return conversion_dict_subset.to_numpy(np_type)
            return None

        self.spd_t = convert_to_numpy(self.spd_dict['timestamp'], np.float64)
        self.spd_v = convert_to_numpy(self.spd_dict['Airspeed'])

        self.alt_t = convert_to_numpy(self.alt_dict['timestamp'], np.float64)
        self.alt = convert_to_numpy(self.alt_dict['Alt'])

        self.att_t = convert_to_numpy(self.roll_dict['timestamp'], np.float64)
        self.roll = convert_to_numpy(self.roll_dict['Roll'])
        self.desroll = convert_to_numpy(self.roll_dict['DesRoll'])
        self.pitch = convert_to_numpy(self.roll_dict['Pitch'])
        self.despitch = convert_to_numpy(self.roll_dict['DesPitch'])
        self.yaw = convert_to_numpy(self.roll_dict['Yaw'])
        self.desyaw = convert_to_numpy(self.roll_dict['DesYaw'])

        self.cmd_t = convert_to_numpy(self.cmd_dict['timestamp'], np.float64)
        self.c5 = convert_to_numpy(self.cmd_dict['C5'])  # throttle pwm
        self.c10 = convert_to_numpy(self.cmd_dict['C10'])  # mode pwm
        self.cmd_throttle_perc = self.throttle_pwm_to_perc(self.c5).astype(np.float32)

        # --- GPS handling ---
        self.has_gps = False
        if len(self.gps_dict) > 0:
            gps_t = convert_to_numpy(self.gps_dict['timestamp'], np.float64)
            gps_lat = convert_to_numpy(self.gps_dict['Lat'], np.float64)
            gps_lng = convert_to_numpy(self.gps_dict['Lng'], np.float64)

            # Optional fields
            if 'Alt' in self.gps_dict.columns:
                gps_alt = convert_to_numpy(self.gps_dict['Alt'])
            else:
                gps_alt = np.zeros(len(self.gps_dict), dtype=np.float32)

            if 'Spd' in self.gps_dict.columns:
                gps_spd = convert_to_numpy(self.gps_dict['Spd'])
            else:
                gps_spd = np.zeros(len(self.gps_dict), dtype=np.float32)

            if 'GCrs' in self.gps_dict.columns:
                gps_gc = convert_to_numpy(self.gps_dict['GCrs'])
            else:
                gps_gc = np.zeros(len(self.gps_dict), dtype=np.float32)

            if 'Yaw' in self.gps_dict.columns:
                gps_yaw = convert_to_numpy(self.gps_dict['Yaw'])
            else:
                gps_yaw = np.zeros(len(self.gps_dict), dtype=np.float32)

            valid = (
                np.isfinite(gps_t) &
                np.isfinite(gps_lat) &
                np.isfinite(gps_lng) &
                ((np.abs(gps_lat) > 1e-12) | (np.abs(gps_lng) > 1e-12))
            )

            gps_t = gps_t[valid]
            gps_lat = gps_lat[valid]
            gps_lng = gps_lng[valid]
            gps_alt = gps_alt[valid]
            gps_spd = gps_spd[valid]
            gps_gc = gps_gc[valid]
            gps_yaw = gps_yaw[valid]

            gps_t_unique, unique_idx = np.unique(gps_t, return_index=True)
            gps_t = gps_t_unique
            gps_lat = gps_lat[unique_idx]
            gps_lng = gps_lng[unique_idx]
            gps_alt = gps_alt[unique_idx]
            gps_spd = gps_spd[unique_idx]
            gps_gc = gps_gc[unique_idx]
            gps_yaw = gps_yaw[unique_idx]

            if len(gps_t) >= 2:
                self.gps_t = gps_t
                self.gps_lat = gps_lat
                self.gps_lng = gps_lng
                self.gps_alt = gps_alt
                self.gps_spd = gps_spd
                self.gps_gc = gps_gc
                self.gps_yaw = gps_yaw
                self.gps_gc_unwrapped = self._unwrap_angle_series_deg(gps_gc)
                self.gps_yaw_unwrapped = self._unwrap_angle_series_deg(gps_yaw)

                self.lat_min = float(np.min(self.gps_lat))
                self.lat_max = float(np.max(self.gps_lat))
                self.lng_min = float(np.min(self.gps_lng))
                self.lng_max = float(np.max(self.gps_lng))

                self.gps_lat0_deg = float(np.mean(self.gps_lat))
                self.gps_lng0_deg = float(np.mean(self.gps_lng))
                self.gps_cos_lat0 = float(np.cos(np.deg2rad(self.gps_lat0_deg)))

                deg_to_rad = np.deg2rad(1.0)
                self.gps_east_m = (
                    (self.gps_lng - self.gps_lng0_deg)
                    * deg_to_rad
                    * _EARTH_RADIUS_M
                    * self.gps_cos_lat0
                )
                self.gps_north_m = (
                    (self.gps_lat - self.gps_lat0_deg)
                    * deg_to_rad
                    * _EARTH_RADIUS_M
                )

                self.gps_map_x = self.gps_east_m
                self.gps_map_y = self.gps_north_m

                self.gps_home_x = self.gps_map_x[0]
                self.gps_home_y = self.gps_map_y[0]

                self.map_x_min = float(np.min(self.gps_map_x))
                self.map_x_max = float(np.max(self.gps_map_x))
                self.map_y_min = float(np.min(self.gps_map_y))
                self.map_y_max = float(np.max(self.gps_map_y))

                self.runway_corners_map_m = self._runway_corners_in_map_frame()
                self.runway_rect_map_m = self._runway_rect_in_map_frame(self.runway_corners_map_m)
                self.has_gps = True

        # free dataframes to reduce memory/GC churn
        self.spd_dict = None
        self.alt_dict = None
        self.roll_dict = None
        self.cmd_dict = None
        self.gps_dict = None

        self.ready = True
        return True

    @staticmethod
    def _wrap_angle_deg(angle_deg):
        return np.mod(angle_deg, 360.0)

    @staticmethod
    def _unwrap_angle_series_deg(angles_deg: NDArray) -> NDArray:
        return np.rad2deg(np.unwrap(np.deg2rad(angles_deg.astype(np.float64))))

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
                home_x=self.gps_home_x,
                home_y=self.gps_home_y
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
                home_x=self.gps_home_x,
                home_y=self.gps_home_y
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
                east_m = float(np.interp(t, self.gps_t, self.gps_east_m))
                north_m = float(np.interp(t, self.gps_t, self.gps_north_m))
                lat_deg = float(self.gps_lat0_deg + np.rad2deg(north_m / _EARTH_RADIUS_M))
                lng_deg = float(
                    self.gps_lng0_deg +
                    np.rad2deg(east_m / (_EARTH_RADIUS_M * max(self.gps_cos_lat0, 1e-12)))
                )
                gps_alt_m = float(np.interp(t, self.gps_t, self.gps_alt))
                gps_speed_mps = float(np.interp(t, self.gps_t, self.gps_spd))
                gps_ground_course_deg = float(
                    self._wrap_angle_deg(np.interp(t, self.gps_t, self.gps_gc_unwrapped))
                )
                gps_yaw_deg = float(
                    self._wrap_angle_deg(np.interp(t, self.gps_t, self.gps_yaw_unwrapped))
                )

                map_x = east_m
                map_y = north_m
                gps_valid = True

        runway_corners_body_m = None
        runway_corners_map_m = None
        runway_rect_map_m = None
        if gps_valid:
            runway_corners_body_m = self._runway_corners_in_body_frame(
                lat_deg=lat_deg,
                lng_deg=lng_deg,
                alt_m=gps_alt_m,
                roll_deg=float(roll),
                pitch_deg=float(pitch),
                yaw_deg=float(yaw),
            )
            runway_corners_map_m = self.runway_corners_map_m
            runway_rect_map_m = self.runway_rect_map_m

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
            runway_corners_body_m=runway_corners_body_m,
            runway_corners_map_m=runway_corners_map_m,
            runway_rect_map_m=runway_rect_map_m,
            home_x=self.gps_home_x,
            home_y=self.gps_home_y
        )

    @staticmethod
    def _lla_to_ned_delta_m(
        lat_deg: float,
        lng_deg: float,
        alt_m: float,
        ref_lat_deg: float,
        ref_lng_deg: float,
        ref_alt_m: float,
    ) -> NDArray:
        lat_rad = np.deg2rad(lat_deg)
        lng_rad = np.deg2rad(lng_deg)
        ref_lat_rad = np.deg2rad(ref_lat_deg)
        ref_lng_rad = np.deg2rad(ref_lng_deg)

        d_lat = lat_rad - ref_lat_rad
        d_lng = lng_rad - ref_lng_rad

        north_m = d_lat * _EARTH_RADIUS_M
        east_m = d_lng * _EARTH_RADIUS_M * np.cos(ref_lat_rad)
        down_m = ref_alt_m - alt_m
        return np.array([north_m, east_m, down_m], dtype=float)

    @staticmethod
    def _rotmat_wr_from_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float) -> NDArray:
        rr = np.deg2rad(roll_deg)
        rp = np.deg2rad(pitch_deg)
        ry = np.deg2rad(yaw_deg)

        cr, sr = np.cos(rr), np.sin(rr)
        cp, sp = np.cos(rp), np.sin(rp)
        cy, sy = np.cos(ry), np.sin(ry)

        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ], dtype=float)
        ry = np.array([
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ], dtype=float)
        rz = np.array([
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        return rz @ ry @ rx

    def _runway_corners_in_body_frame(
        self,
        lat_deg: float,
        lng_deg: float,
        alt_m: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> tuple[tuple[float, float, float], ...]:
        r_wr = self._rotmat_wr_from_rpy(roll_deg, pitch_deg, yaw_deg)
        r_rw = r_wr.T

        body_pts = []
        for corner_lat, corner_lng, corner_alt in _RUNWAY_CORNERS_LLA:
            runway_ned = self._lla_to_ned_delta_m(
                lat_deg=corner_lat,
                lng_deg=corner_lng,
                alt_m=corner_alt,
                ref_lat_deg=lat_deg,
                ref_lng_deg=lng_deg,
                ref_alt_m=alt_m,
            )
            runway_body = r_rw @ runway_ned
            body_pts.append(tuple(float(v) for v in runway_body))

        return tuple(body_pts)

    def _runway_corners_in_map_frame(self) -> tuple[tuple[float, float], ...] | None:
        if self.gps_t is None:
            return None

        corners = []
        deg_to_rad = np.pi / 180.0
        for corner_lat, corner_lng, _corner_alt in _RUNWAY_CORNERS_LLA:
            east_m = (
                (corner_lng - self.gps_lng0_deg)
                * deg_to_rad
                * _EARTH_RADIUS_M
                * self.gps_cos_lat0
            )
            north_m = (
                (corner_lat - self.gps_lat0_deg)
                * deg_to_rad
                * _EARTH_RADIUS_M
            )
            corners.append((float(east_m), float(north_m)))

        return tuple(corners)

    @staticmethod
    def _runway_rect_in_map_frame(
        runway_corners_map_m: tuple[tuple[float, float], ...] | None
    ) -> tuple[float, float, float, float] | None:
        if not runway_corners_map_m:
            return None

        corners = np.asarray(runway_corners_map_m, dtype=float)
        return (
            float(np.min(corners[:, 0])),
            float(np.min(corners[:, 1])),
            float(np.max(corners[:, 0])),
            float(np.max(corners[:, 1])),
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
    def throttle_pwm_to_perc(throttle_pwm: NDArray) -> NDArray:
        MIN_THROTTLE = 1000.0
        MAX_THROTTLE = 2000.0
        return (throttle_pwm - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE) * 100.0
