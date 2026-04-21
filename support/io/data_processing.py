from __future__ import annotations
import re
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np

import cv2
from concurrent.futures import ThreadPoolExecutor, wait

from support.core.pixel_kalmanFilter import KalmanFilter as PixelKalmanFilter
from support.io.my_logging import LOG


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


ProgressCb = Callable[[float, str], None]  # (overall_frac, text)
StatusCb = Callable[[str], None]
FinishCb = Callable[[str], None]

ProgressUiCb = Callable[[float, str], None]   # (overall_frac, text)
StatusUiCb   = Callable[[str], None]

@dataclass
class PnpQnpSweepPaths:
    img_dir: Path
    processed_dir: Path

    def yolo_csv_for_conf(self, conf: float) -> Path:
        return self.processed_dir / f"1_yolo_detections_conf{conf:.2f}.csv"

@dataclass
class YoloSweepParams:
    img_dir: Path
    out_csv_base: Path  # .../_ProcessedData/1_yolo_detections.csv (base name)
    conf_list_var: object  # Tk var or string; parsed by parse_conf_list()
    ckpt_every_var: object | None  # Tk var or int/str
    prefetch_var: object | None  # Tk var or int/str
    cam_to_log_time_offset: float


def _is_completed_row(row: dict, n_cls: int) -> bool:
    """Row is 'complete' if it has image_name, image_time and all feat_<cid>_x/y present (even if -1)."""
    if "image_name" not in row or "image_time" not in row:
        return False
    for cid in range(n_cls):
        if f"feat_{cid}_x" not in row or f"feat_{cid}_y" not in row:
            return False
    return True


def parse_conf_list(raw_var_or_str: str) -> list[float]:
    """
    Read the Data Processing confidence list from the GUI and return
    a list of floats.

    Any parse error or out-of-range value => fallback to [0.80].
    """
    default = [0.80]
    if raw_var_or_str is None:
        return default

    # Tk var?
    try:
        raw = (raw_var_or_str.get() or "").strip()
    except Exception:
        raw = str(raw_var_or_str).strip()

    try:
        parts = [p.strip() for p in raw.split(",")]
        vals = [float(p) for p in parts if p]

        # no valid numbers?
        if not vals:
            return default

        # ensure all are in [0,1]
        for v in vals:
            if not (0.0 <= v <= 1.0):
                return default

        return vals

    except Exception:
        return default


def _write_csv_atomic(out_csv: str, columns: list[str], completed_map: dict[str, dict]):
    """Write CSV atomically and sort by numeric portion of image_name."""
    import pandas as pd
    rows = list(completed_map.values())
    df = pd.DataFrame(rows, columns=columns)

    # --- Sort numerically by filename stem (e.g. 1.png, 2.png, 10.png) ---
    def _numeric_key(name: str) -> int:
        try:
            # extract first integer from filename; fall back to 0 if none
            return int(re.search(r"\d+", str(name)).group())
        except Exception:
            return 0

    df = df.sort_values(
        by="image_name",
        key=lambda col: col.map(_numeric_key),
        ignore_index=True,
    )

    tmp = out_csv + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)  # atomic replace


class DataProcessorRunner:
    def __init__(self) -> None:
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def reset_cancel(self) -> None:
        self.cancel_event.clear()

    ############### YOLO PROCESSING ####################

    def run_yolo_conf_sweep(
            self,
            *,
            yolo_session,
            calibration,
            ids_times_pairs: list[tuple[str, float | None]],
            params: YoloSweepParams,
            sweep_timer,  # your utils.SweepTimer instance
            fmt_mmss: Callable[[float], str],
            post_progress: ProgressCb,
            post_status: StatusCb,
            post_finish: FinishCb,
            undistort_points_px: Callable[[object, tuple[float, float]], tuple[float, float]],
    ) -> None:
        """
        Runs YOLO across all images for each conf in conf_list.
        Writes one CSV per confidence: ..._conf0.80.csv etc.
        """
        yolo_wall_start = time.perf_counter()
        img_dir = Path(params.img_dir)
        if not img_dir.exists():
            post_status("No valid image directory selected.")
            post_finish("Ready.")
            return

        proc_dir = img_dir / "_ProcessedData"
        proc_dir.mkdir(parents=True, exist_ok=True)

        out_csv = Path(params.out_csv_base)
        out_csv = out_csv if out_csv.is_absolute() else (img_dir / out_csv)

        # Freeze current session thresholds (carry over from main config)
        iou = float(yolo_session.iou)
        conf_list = parse_conf_list(params.conf_list_var)  # Tk var compatible :contentReference[oaicite:2]{index=2}

        post_status(
            "Running YOLO batch sweep: "
            + ", ".join(f"{c:.2f}" for c in conf_list)
            + f"  (iou={iou:.2f})"
        )

        pairs = list(ids_times_pairs)
        if not pairs:
            post_status("No images found in the selected folder.")
            post_finish("Ready.")
            return

        n_cls = int(yolo_session.num_classes)

        def _feat_cols(cid: int):
            if n_cls == 1:
                return [
                    f"feat_{cid}_x1_dist",
                    f"feat_{cid}_y1_dist",
                    f"feat_{cid}_x2_dist",
                    f"feat_{cid}_y2_dist",
                ]
            else:
                return [
                    f"feat_{cid}_x_distPX",
                    f"feat_{cid}_y_distPX",
                    f"feat_{cid}_x_undistPX",
                    f"feat_{cid}_y_undistPX",
                ]

        columns = ["image_name", "image_time"]
        for cid in range(n_cls):
            columns.extend(_feat_cols(cid))

        conf_n = len(conf_list)
        all_total_imgs = len(pairs)
        overall_total = max(1, conf_n * all_total_imgs)
        overall_done = 0

        sweep_timer.start()
        current_conf = None

        # Pre-grab calibration pieces once
        if calibration is not None:
            width = float(getattr(calibration, "width", 1.0))
            height = float(getattr(calibration, "height", 1.0))
        else:
            width = 1.0
            height = 1.0

        time_offset = float(params.cam_to_log_time_offset)
        time_map = {Path(p).name: (None if t is None else float(t) + time_offset) for (p, t) in pairs}

        for conf_i, conf in enumerate(conf_list, start=1):
            current_conf = float(conf)
            yolo_session.conf = current_conf

            out_csv_conf = str(out_csv).replace(".csv", f"_conf{current_conf:.2f}.csv")

            post_status(f"Preparing batch (conf={current_conf:.2f})…")

            # Resume from existing CSV
            completed_map: dict[str, dict] = {}
            if os.path.exists(out_csv_conf):
                try:
                    import pandas as pd
                    prev = pd.read_csv(out_csv_conf)
                    for col in columns:
                        if col not in prev.columns:
                            prev[col] = (-1.0 if col.startswith("feat_") else None)
                    need = set(columns)
                    for _, r in prev.iterrows():
                        rd = r.to_dict()
                        if need.issubset(rd.keys()):
                            completed_map[str(rd["image_name"])] = rd
                except Exception as e:
                    post_status(f"Existing CSV unreadable, starting fresh: {e}")

            overall_done += len(completed_map)

            work_items: list[tuple[str, str]] = []
            for p, _t in pairs:
                name = Path(p).name
                if name in completed_map:
                    continue
                work_items.append((p, name))

            total_todo = len(work_items)
            if total_todo == 0:
                try:
                    _write_csv_atomic(out_csv_conf, columns, completed_map)
                except Exception as e:
                    post_status(f"Failed to write CSV: {e}")

                eta_txt = fmt_mmss(sweep_timer.eta_from_fraction(overall_done / max(1, overall_total)))
                post_progress(
                    overall_done / float(max(1, overall_total)),
                    f"[{conf_i}/{conf_n}] Completed conf={current_conf:.2f}. Preparing next… • ETA {eta_txt}",
                )
                self.cancel_event.clear()
                continue

            # checkpoint config
            checkpoint_every = 0
            try:
                raw = params.ckpt_every_var.get() if params.ckpt_every_var is not None else 0
                checkpoint_every = max(0, int(str(raw).strip()))
            except Exception:
                checkpoint_every = 0
            processed_since_ckpt = 0

            # prefetch
            prefetch = 32
            try:
                raw = params.prefetch_var.get() if params.prefetch_var is not None else 32
                prefetch = max(2, int(str(raw).strip()))
            except Exception:
                prefetch = 32

            from os import cpu_count
            cpu_workers = max(2, min(prefetch, (cpu_count() or 4)))
            que: queue.Queue = queue.Queue(maxsize=prefetch)
            producers_done = threading.Event()

            yW, yH = yolo_session.yoloSize

            def _producer_job(path_str: str, name: str):
                if self.cancel_event.is_set():
                    return

                p = Path(path_str)
                if not p.exists():
                    rp = img_dir / p.name
                    if rp.exists():
                        p = rp

                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    item = (name, None, (0, 0))
                else:
                    H, W = img.shape[:2]
                    try:
                        tensor = yolo_session.preprocessImage(img)
                        item = (name, tensor, (W, H))
                    except Exception:
                        item = (name, None, (W, H))

                while not self.cancel_event.is_set():
                    try:
                        que.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue

            made = 0
            ex = ThreadPoolExecutor(max_workers=cpu_workers)
            try:
                futures = [ex.submit(_producer_job, p, name) for (p, name) in work_items]

                def _watch():
                    wait(futures)
                    producers_done.set()

                threading.Thread(target=_watch, daemon=True).start()

                while True:
                    if (self.cancel_event.is_set() or producers_done.is_set()) and que.empty():
                        break

                    try:
                        name, tensor, (W, H) = que.get(timeout=0.1)
                    except queue.Empty:
                        # heartbeat
                        eta_txt = fmt_mmss(sweep_timer.eta_from_fraction(overall_done / max(1, overall_total)))
                        post_progress(
                            overall_done / float(max(1, overall_total)),
                            f"[{conf_i}/{conf_n}] conf={current_conf:.2f} • Working… • ETA {eta_txt}",
                        )
                        continue

                    if tensor is not None:
                        centers, boxes, scores, classes, _dt = yolo_session.runOneSession(tensor)

                        rec = {c: -1.0 for cid in range(n_cls) for c in _feat_cols(cid)}
                        sx, sy = (W / float(yW)), (H / float(yH))

                        if n_cls == 1:
                            if boxes:
                                x1, y1, x2, y2 = boxes[0]
                                rec["feat_0_x1_dist"] = float(x1) * sx / width
                                rec["feat_0_y1_dist"] = float(y1) * sy / height
                                rec["feat_0_x2_dist"] = float(x2) * sx / width
                                rec["feat_0_y2_dist"] = float(y2) * sy / height
                        else:
                            for (cx, cy), cid in zip(centers, classes):
                                cidi = int(cid)
                                x = float(cx) * sx
                                y = float(cy) * sy
                                xp, yp = undistort_points_px(calibration, (x, y))
                                rec[f"feat_{cidi}_x_distPX"] = x
                                rec[f"feat_{cidi}_y_distPX"] = y
                                rec[f"feat_{cidi}_x_undistPX"] = xp
                                rec[f"feat_{cidi}_y_undistPX"] = yp

                        row = {"image_name": name, "image_time": time_map.get(name, None)}
                        row.update(rec)
                        completed_map[name] = row

                    made += 1
                    processed_since_ckpt += 1
                    overall_done += 1

                    overall_frac = float(overall_done) / float(max(1, overall_total))
                    eta_txt = fmt_mmss(sweep_timer.eta_from_fraction(overall_frac))
                    pct = int(overall_frac * 100.0 + 0.5)

                    post_progress(
                        overall_frac,
                        f"[{conf_i}/{conf_n}] conf={current_conf:.2f} • "
                        f"conf: {made}/{max(1, total_todo)} • overall: {overall_done}/{overall_total} ({pct}%) • ETA {eta_txt} • "
                        f"(total done: {len(completed_map)}/{len(pairs)})",
                    )

                    if checkpoint_every > 0 and (
                    not self.cancel_event.is_set()) and processed_since_ckpt >= checkpoint_every:
                        try:
                            _write_csv_atomic(out_csv_conf, columns, completed_map)
                            processed_since_ckpt = 0
                            post_status(f"Checkpoint saved ({len(completed_map)} rows)…")
                        except Exception as e:
                            post_status(f"Checkpoint save failed: {e}")

            finally:
                if self.cancel_event.is_set():
                    ex.shutdown(wait=False, cancel_futures=True)
                else:
                    ex.shutdown(wait=True)

                try:
                    _write_csv_atomic(out_csv_conf, columns, completed_map)
                    msg = ("Partial CSV written (resume later): " + out_csv_conf) if self.cancel_event.is_set() else (
                            "Done. CSV written: " + out_csv_conf
                    )
                except Exception as e:
                    msg = f"Failed to write CSV: {e}"

                post_status(msg)
                self.cancel_event.clear()
        yolo_wall_end = time.perf_counter()
        LOG.info(
            "YOLO sweep timing: total=%.2f s, images=%d, avg=%.3f ms/img",
            yolo_wall_end - yolo_wall_start,
            all_total_imgs * conf_n,
            1000.0 * (yolo_wall_end - yolo_wall_start) / max(1, all_total_imgs * conf_n),
        )

        post_finish("All confidence sweeps completed.")
        self.cancel_event.clear()

    ################## KF ########################
    def run_kalman_tracks_from_detection_csv(
            self,
            *,
            csv_path: str,
            calibration,
            out_csv: str | None = None,
            progress_cb: Callable[[int, int, str], None] | None = None,
            cancel_event=None,  # threading.Event | None
    ) -> str | None:
        """
        Pure version of your method.

        Returns out_csv path on success, or None on early failure.
        """

        if not os.path.exists(csv_path):
            LOG.error("run_kalman_tracks_from_detection_csv: missing CSV: %s", csv_path)
            return None

        import pandas as pd
        df = pd.read_csv(csv_path)
        if df.empty:
            LOG.warning("run_kalman_tracks_from_detection_csv: empty CSV: %s", csv_path)
            return None

        total_rows = len(df)

        # --- Discover feature ids from columns (feat_<id>_x_undistPX) ---
        feat_ids: list[int] = []
        for col in df.columns:
            m = re.match(r"feat_(\d+)_x_undistPX$", col)
            if m:
                fid = int(m.group(1))
                if fid not in feat_ids:
                    feat_ids.append(fid)
        feat_ids.sort()

        if not feat_ids:
            LOG.error("run_kalman_tracks_from_detection_csv: no feat_*_x columns in %s", csv_path)
            return None

        M = len(feat_ids)

        # --- Require calibration (hard-fail) ---
        if calibration is None or not getattr(calibration, "validCal", False):
            raise ValueError("No calibration (calibration missing or invalid).")

        try:
            _K = calibration.getCameraMatrix()
            width = float(calibration.width)
            height = float(calibration.height)
        except Exception as e:
            raise ValueError(f"No calibration (failed to access intrinsics): {e}")

        # --- Output path ---
        base = Path(csv_path)
        if out_csv is None:
            out_csv = str(base.with_name(base.stem + ".csv")).replace("1_yolo_detections", "2_kalman")

        # --- Pull time vector (numeric seconds) ---
        if "image_time" not in df.columns:
            LOG.error("run_kalman_tracks_from_detection_csv: missing image_time column in %s", csv_path)
            return None
        t_sec = df["image_time"].to_numpy(dtype=np.float64)  # (N,)

        # --- Build dense measurement matrices (N,M) ---
        x_cols = [f"feat_{fid}_x_undistPX" for fid in feat_ids]
        y_cols = [f"feat_{fid}_y_undistPX" for fid in feat_ids]

        for c in x_cols:
            if c not in df.columns:
                df[c] = np.nan
        for c in y_cols:
            if c not in df.columns:
                df[c] = np.nan

        Xraw = df[x_cols].to_numpy(dtype=np.float64, copy=False)  # (N,M)
        Yraw = df[y_cols].to_numpy(dtype=np.float64, copy=False)

        kf_start = time.perf_counter()

        inv_w = 1.0 / max(width, 1.0)
        inv_h = 1.0 / max(height, 1.0)
        Xmeas = Xraw * inv_w
        Ymeas = Yraw * inv_h

        valid = np.isfinite(Xmeas) & np.isfinite(Ymeas) & (Xmeas != -1.0) & (Ymeas != -1.0)
        valid_u8 = valid.astype(np.uint8, copy=False)

        # --- KF parameter seed ---
        kf0 = PixelKalmanFilter()
        kf0.set_image_size(width, height)
        kf0.set_sigma_meas_px(2.0, 2.0)
        kf0.set_max_pixel_jump_px(100.0)
        kf0.max_mahalanobis_sq = 9.21

        var_proc = float(kf0.var_proc)
        var_meas_x = float(kf0.var_meas_x)
        var_meas_y = float(kf0.var_meas_y)
        max_pixel_jump = float(kf0.max_pixel_jump)
        max_mahalanobis_sq = float(kf0.max_mahalanobis_sq)

        # --- NIS adaptation parameters ---
        nis_p95_target = 5.991
        nis_beta = 0.01
        nis_clip_lo = 0.25
        nis_clip_hi = 4.0
        min_var_meas = 1e-12

        min_used_frac_for_good = 0.10
        min_used_abs_for_good = 5

        burn_in_good_frames = 30
        accepted_feat_target = int(max(50, burn_in_good_frames * M * min_used_frac_for_good))
        accepted_feat_accum = 0

        freeze_r = True
        frozen = False
        var_meas_x_frozen = None
        var_meas_y_frozen = None

        drift_hi = 9.21
        drift_trigger_good_frames = 20
        drift_count = 0

        stable_hi = 5.991
        stable_trigger_good_frames = 30
        stable_count = 0

        # --- KF bank state ---
        X = np.zeros((M, 4), dtype=np.float64)
        P = np.zeros((M, 4, 4), dtype=np.float64)
        for j in range(M):
            P[j] = np.eye(4, dtype=np.float64) * 10.0
        last_t = np.zeros(M, dtype=np.float64)
        init = np.zeros(M, dtype=np.uint8)

        # --- Output buffers ---
        out_kf_x = np.full((total_rows, M), np.nan, dtype=np.float64)
        out_kf_y = np.full((total_rows, M), np.nan, dtype=np.float64)
        out_kf_vx = np.full((total_rows, M), np.nan, dtype=np.float64)
        out_kf_vy = np.full((total_rows, M), np.nan, dtype=np.float64)
        out_sig_px = np.full((total_rows, M), np.nan, dtype=np.float64)
        out_sig_py = np.full((total_rows, M), np.nan, dtype=np.float64)

        out_used = np.zeros((total_rows, M), dtype=np.uint8)
        out_nis = np.full((total_rows, M), np.nan, dtype=np.float64)

        out_used_rate = np.full(total_rows, np.nan, dtype=np.float64)
        out_nis_med_used = np.full(total_rows, np.nan, dtype=np.float64)
        out_nis_p95_used = np.full(total_rows, np.nan, dtype=np.float64)
        out_var_meas_x = np.full(total_rows, np.nan, dtype=np.float64)
        out_var_meas_y = np.full(total_rows, np.nan, dtype=np.float64)
        out_sig_meas_px = np.full(total_rows, np.nan, dtype=np.float64)
        out_sig_meas_py = np.full(total_rows, np.nan, dtype=np.float64)

        last_report_t = 0.0
        last_report_row = 0
        nis_out = np.empty(M, dtype=np.float64)

        for idx in range(total_rows):
            if cancel_event is not None and cancel_event.is_set():
                LOG.info("Kalman batch canceled at row %d/%d", idx, total_rows)
                break

            image_name = df.iloc[idx].get("image_name", "")

            if progress_cb is not None:
                now = time.monotonic()
                dt = now - last_report_t
                dr = (idx + 1) - last_report_row
                step_rows = max(1, total_rows // 100)
                if (idx == 0) or (idx == total_rows - 1) or (dt >= 0.1) or (dr >= step_rows):
                    try:
                        progress_cb(idx + 1, total_rows, str(image_name))
                    except Exception:
                        pass
                    last_report_t = now
                    last_report_row = (idx + 1)

            used_u8 = PixelKalmanFilter._kf_bank_step_inplace(
                float(t_sec[idx]),
                Xmeas[idx], Ymeas[idx], valid_u8[idx],
                X, P, last_t, init,
                var_proc, var_meas_x, var_meas_y,
                max_pixel_jump, max_mahalanobis_sq,
                nis_out
            )

            out_used[idx, :] = used_u8
            out_nis[idx, :] = nis_out

            used_bool = used_u8.astype(bool)
            used_count = int(used_bool.sum())
            used_rate = float(used_count) / float(max(1, M))
            out_used_rate[idx] = used_rate

            nis_used = nis_out[used_bool]
            good_frame = (used_count >= min_used_abs_for_good) and (used_rate >= min_used_frac_for_good) and (
                    nis_used.size > 0)

            if nis_used.size > 0:
                nis_med = float(np.median(nis_used))
                nis_p95 = float(np.percentile(nis_used, 95.0))
                out_nis_med_used[idx] = nis_med
                out_nis_p95_used[idx] = nis_p95
            else:
                nis_med = np.nan
                nis_p95 = np.nan

            if good_frame:
                accepted_feat_accum += used_count

            do_adapt = (not freeze_r) or (not frozen)
            if do_adapt and good_frame:
                ratio = nis_p95 / nis_p95_target
                ratio = max(nis_clip_lo, min(ratio, nis_clip_hi))
                scale = ratio ** nis_beta
                var_meas_x = max(min_var_meas, var_meas_x * scale)
                var_meas_y = max(min_var_meas, var_meas_y * scale)

            if freeze_r and (not frozen) and (accepted_feat_accum >= accepted_feat_target):
                frozen = True
                var_meas_x_frozen = float(var_meas_x)
                var_meas_y_frozen = float(var_meas_y)
                LOG.info(
                    "Freezing KF measurement noise after evidence: accepted_feat=%d target=%d "
                    "var_meas=(%.3e, %.3e) sigma_px=(%.3f, %.3f)",
                    accepted_feat_accum, accepted_feat_target,
                    var_meas_x_frozen, var_meas_y_frozen,
                    np.sqrt(var_meas_x_frozen) * width, np.sqrt(var_meas_y_frozen) * height,
                )

            if freeze_r and frozen and good_frame and (not np.isnan(nis_p95)):
                if nis_p95 > drift_hi:
                    drift_count += 1
                else:
                    drift_count = max(0, drift_count - 1)

                if drift_count >= drift_trigger_good_frames:
                    frozen = False
                    drift_count = 0
                    stable_count = 0
                    LOG.info(
                        "Unfreezing KF measurement noise due to sustained NIS drift: p95>%.3f for %d good frames",
                        drift_hi, drift_trigger_good_frames
                    )

            if freeze_r and (not frozen) and good_frame and (not np.isnan(nis_p95)):
                if nis_p95 <= stable_hi:
                    stable_count += 1
                else:
                    stable_count = max(0, stable_count - 1)

                if stable_count >= stable_trigger_good_frames:
                    frozen = True
                    stable_count = 0
                    var_meas_x_frozen = float(var_meas_x)
                    var_meas_y_frozen = float(var_meas_y)
                    LOG.info(
                        "Re-freezing KF measurement noise after stability: var_meas=(%.3e, %.3e) sigma_px=(%.3f, %.3f)",
                        var_meas_x_frozen, var_meas_y_frozen,
                        np.sqrt(var_meas_x_frozen) * width, np.sqrt(var_meas_y_frozen) * height,
                    )

            if freeze_r and frozen and (var_meas_x_frozen is not None):
                var_meas_x = var_meas_x_frozen
                var_meas_y = var_meas_y_frozen

            out_var_meas_x[idx] = var_meas_x
            out_var_meas_y[idx] = var_meas_y
            out_sig_meas_px[idx] = (np.sqrt(var_meas_x) * width)
            out_sig_meas_py[idx] = (np.sqrt(var_meas_y) * height)

            # write per-feature outputs
            for j in range(M):
                if init[j] == 0:
                    continue

                out_kf_x[idx, j] = X[j, 0]
                out_kf_y[idx, j] = X[j, 1]
                out_kf_vx[idx, j] = X[j, 2]
                out_kf_vy[idx, j] = X[j, 3]

                sig_px = float(width * np.sqrt(max(P[j, 0, 0], 0.0)))
                sig_py = float(height * np.sqrt(max(P[j, 1, 1], 0.0)))
                out_sig_px[idx, j] = sig_px
                out_sig_py[idx, j] = sig_py
        kf_end = time.perf_counter()
        LOG.info(
            "KF feature tracking timing: total=%.2f s, rows=%d, avg=%.3f ms/frame",
            kf_end - kf_start,
            total_rows,
            1000.0 * (kf_end - kf_start) / max(1, total_rows),
        )

        # --- Build output DataFrame ---
        import pandas as pd
        out_df = df.copy()

        new_cols = {
            "kf_used_rate": out_used_rate,
            "kf_nis_med_used": out_nis_med_used,
            "kf_nis_p95_used": out_nis_p95_used,
            "kf_var_meas_x": out_var_meas_x,
            "kf_var_meas_y": out_var_meas_y,
            "kf_sigma_meas_px": out_sig_meas_px,
            "kf_sigma_meas_py": out_sig_meas_py,
        }

        for j, fid in enumerate(feat_ids):
            new_cols[f"feat_{fid}_kf_x"] = out_kf_x[:, j]
            new_cols[f"feat_{fid}_kf_y"] = out_kf_y[:, j]
            new_cols[f"feat_{fid}_kf_vx"] = out_kf_vx[:, j]
            new_cols[f"feat_{fid}_kf_vy"] = out_kf_vy[:, j]
            new_cols[f"feat_{fid}_kf_sigma_px"] = np.clip(out_sig_px[:, j], 1e-6, None)
            new_cols[f"feat_{fid}_kf_sigma_py"] = np.clip(out_sig_py[:, j], 1e-6, None)
            new_cols[f"feat_{fid}_kf_used"] = out_used[:, j].astype(np.uint8)
            new_cols[f"feat_{fid}_kf_nis"] = out_nis[:, j]

        out_df = pd.concat([out_df, pd.DataFrame(new_cols)], axis=1)

        cols_to_drop = []
        for fid in feat_ids:
            cols_to_drop.append(f"feat_{fid}_x_distPX")
            cols_to_drop.append(f"feat_{fid}_y_distPX")
        out_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        out_df.to_csv(out_csv, index=False)
        LOG.info("Kalman tracks CSV written: %s", out_csv)
        return out_csv

    def run_kalman_conf_sweep(
            self,
            *,
            img_dir: Path,
            conf_list_var,
            calibration,
            progress_cb: Callable[[float, str], None],
            status_cb: Callable[[str], None],
    ) -> None:
        start_time = time.time()
        confs = parse_conf_list(conf_list_var)
        proc_dir = img_dir / "_ProcessedData"
        proc_dir.mkdir(parents=True, exist_ok=True)

        n_conf = max(1, len(confs))
        overall_done = 0
        overall_total = n_conf  # coarse: 1 unit per conf (we’ll also show per-row inside)

        for i, conf in enumerate(confs, start=1):
            if self.cancel_event.is_set():
                status_cb("Kalman sweep canceled.")
                return

            in_csv = proc_dir / f"1_yolo_detections_conf{conf:.2f}.csv"
            if not in_csv.exists():
                status_cb(f"[{i}/{n_conf}] missing YOLO CSV: {in_csv.name} (skipping)")
                overall_done += 1
                progress_cb(overall_done / overall_total, f"[{i}/{n_conf}] skipped conf={conf:.2f}")
                continue

            out_csv = proc_dir / f"2_kalman_conf{conf:.2f}.csv"

            status_cb(f"[{i}/{n_conf}] running KF (conf={conf:.2f})…")

            # per-row progress hook that also updates overall
            def _row_progress(r: int, n: int, name: str):
                # overall fraction = conf_index + within-conf fraction
                within = float(r) / float(max(1, n))
                frac = (float(i - 1) + within) / float(max(1, n_conf))
                progress_cb(frac, f"[{i}/{n_conf}] conf={conf:.2f} • {r}/{n} • {name}")

            self.run_kalman_tracks_from_detection_csv(
                csv_path=str(in_csv),
                calibration=calibration,
                out_csv=str(out_csv),
                progress_cb=_row_progress,
                cancel_event=self.cancel_event,
            )

            overall_done = i
            progress_cb(overall_done / overall_total, f"[{i}/{n_conf}] finished conf={conf:.2f}")

        status_cb("Kalman confidence sweep completed.")

    def run_pnp_qnp_conf_sweep(
            self,
            *,
            img_dir: Path,
            conf_list_var,
            run_pnp_qnp_from_detection_csv: Callable[..., object],
            post_progress: ProgressUiCb,
            post_status: StatusUiCb,
            sweep_timer,  # utils.SweepTimer()
            fmt_mmss: Callable[[float], str],
    ) -> None:
        """
        Runs PnP/QnP for each YOLO detection CSV across the confidence list.
        Uses the provided callable `run_pnp_qnp_from_detection_csv(csv_path, progress_cb=...)`.
        """

        img_dir = Path(img_dir)
        if not img_dir.exists():
            post_status("No valid image directory selected.")
            return

        processed_dir = img_dir / "_ProcessedData"
        processed_dir.mkdir(parents=True, exist_ok=True)

        paths = PnpQnpSweepPaths(img_dir=img_dir, processed_dir=processed_dir)

        conf_values = parse_conf_list(conf_list_var)
        total = len(conf_values)

        if total <= 0:
            post_status("No conf values provided.")
            return

        sweep_timer.start()

        for i, conf in enumerate(conf_values, start=1):
            if self.cancel_event.is_set():
                post_status("SolvePnP/QnP canceled.")
                return

            conf = float(conf)
            target = paths.yolo_csv_for_conf(conf)

            post_status(f"[{i}/{total}] Checking conf={conf:.2f}…")

            if not target.exists():
                overall = i / max(total, 1)
                eta_txt = fmt_mmss(sweep_timer.eta_from_fraction(overall))
                post_progress(overall, f"[{i}/{total}] Skipped (missing file) • ETA {eta_txt}")
                continue

            # -------- throttled per-row callback adapter --------
            last_report_t = 0.0
            last_report_row = 0

            def row_progress(done_rows: int, total_rows: int, img_name: str):
                nonlocal last_report_t, last_report_row

                now = time.monotonic()

                # Same throttling conditions you had
                if done_rows == 1 or done_rows == total_rows:
                    do_update = True
                else:
                    dt = now - last_report_t
                    dr = done_rows - last_report_row
                    step_rows = max(1, total_rows // 100)
                    do_update = (dt >= 0.1) or (dr >= step_rows)

                if not do_update:
                    return

                last_report_t = now
                last_report_row = done_rows

                base_frac = (i - 1) / max(total, 1)
                inner_frac = done_rows / max(total_rows, 1)
                overall = base_frac + inner_frac / max(total, 1)

                eta_txt = fmt_mmss(sweep_timer.eta_from_fraction(overall))

                post_progress(
                    overall,
                    f"[{i}/{total}] {os.path.basename(str(target))} – "
                    f"{done_rows}/{total_rows} images (last: {img_name}) • ETA {eta_txt}",
                )

                # Optional: allow cancel to take effect faster even if inner algorithm is slow
                if self.cancel_event.is_set():
                    return

            # -------- end adapter --------

            post_status(f"[{i}/{total}] Running SolvePnP/QnP on {target.name}")
            try:
                # Keep your existing algorithm in the GUI for now; we just call it.
                run_pnp_qnp_from_detection_csv(
                    str(target),
                    progress_cb=row_progress,
                    cancel_cb=self.cancel_event.is_set,
                )
            except Exception as e:
                post_status(f"Error on {target.name}: {e}")
                continue

            # advance progress to end-of-file if algorithm doesn't report final row
            overall = i / max(total, 1)
            eta_txt = fmt_mmss(sweep_timer.eta_from_fraction(overall))
            post_progress(overall, f"[{i}/{total}] Finished {target.name} • ETA {eta_txt}")

        post_status(f"Done! Processed {total} SolvePnP/QnP files.")

    def run_pnp_qnp_from_detection_csv(
            self,
            *,
            csv_path: str,
            calibration,
            truth_dict: dict,
            checkpoint_every: int = 0,
            out_pnp: str | None = None,
            out_qnp: str | None = None,
            progress_cb: Callable[[int, int, str], None] | None = None,
            cancel_event=None,
    ) -> None:
        """
        Reads a 1_yolo_detections_*.csv and produces:
          - 3_pnp_*.csv
          - 4_qnp_*.csv

        Inputs expected (per feature ID):
          feat_<id>_x_distPX, feat_<id>_y_distPX,
          feat_<id>_x_undistPX, feat_<id>_y_undistPX

        If cancel_event is provided, it must have .is_set().
        """

        t_pnp = 0.0
        t_qnp = 0.0
        t_qnp_kf = 0.0
        n_pose = 0

        import pandas as pd
        # -------------------- sanity --------------------
        if not getattr(calibration, "validCal", False):
            LOG.error("run_pnp_qnp_from_detection_csv: calibration is not valid.")
            return

        csv_path = str(csv_path)
        base = Path(csv_path)

        df = pd.read_csv(csv_path)
        total_rows = int(len(df))
        if total_rows <= 0:
            LOG.warning("run_pnp_qnp_from_detection_csv: %s is empty", csv_path)
            return

        if out_pnp is None:
            # base.stem is like "1_yolo_detections_XXXX"; 18 chars matches your prior convention
            out_pnp = str(base.with_name("3_pnp_" + base.stem[18:] + ".csv"))
        if out_qnp is None:
            out_qnp = str(base.with_name("4_qnp_" + base.stem[18:] + ".csv"))

        # ------------------------------------------------------------------
        # Optional Kalman-trust CSV (for KF-weighted QnP)
        # ------------------------------------------------------------------
        df_kf = None
        kalman_available = False
        kalman_csv = base.with_name("2_kalman_" + base.stem[18:] + ".csv")
        if kalman_csv.exists():
            try:
                df_kf = pd.read_csv(kalman_csv)
                if len(df_kf) == len(df):
                    kalman_available = True
                    LOG.info("run_pnp_qnp_from_detection_csv: using Kalman trust from %s", kalman_csv)
                else:
                    LOG.warning(
                        "run_pnp_qnp_from_detection_csv: %s has %d rows but %s has %d; disabling Kalman weighting",
                        kalman_csv, len(df_kf), csv_path, len(df),
                    )
            except Exception as e:
                LOG.warning(
                    "run_pnp_qnp_from_detection_csv: could not read Kalman CSV %s: %s; disabling Kalman weighting",
                    kalman_csv, e,
                )

        # --- Resume support ----------------------------------------------
        processed_pnp: set[str] = set()
        processed_qnp: set[str] = set()

        if os.path.exists(out_pnp):
            try:
                df_pnp = pd.read_csv(out_pnp)
                if "image_name" in df_pnp.columns:
                    processed_pnp = set(df_pnp["image_name"].astype(str).tolist())
                LOG.info("Existing PnP file %s with %d rows", out_pnp, len(processed_pnp))
            except Exception as e:
                LOG.warning("Could not read existing PnP file %s: %s; recomputing all rows", out_pnp, e)

        if os.path.exists(out_qnp):
            try:
                df_qnp = pd.read_csv(out_qnp)
                if "image_name" in df_qnp.columns:
                    processed_qnp = set(df_qnp["image_name"].astype(str).tolist())
                LOG.info("Existing QnP file %s with %d rows", out_qnp, len(processed_qnp))
            except Exception as e:
                LOG.warning("Could not read existing QnP file %s: %s; recomputing all rows", out_qnp, e)

        already_done = processed_pnp & processed_qnp
        if already_done:
            LOG.info("Will skip %d rows already in BOTH outputs", len(already_done))

        pnp_header_written = os.path.exists(out_pnp)
        qnp_header_written = os.path.exists(out_qnp)

        checkpoint_every = int(checkpoint_every or 0)
        LOG.info("Checkpoint cadence: every %d processed rows.", checkpoint_every)

        pnp_batch: list[dict] = []
        qnp_batch: list[dict] = []
        rows_since_ckpt = 0
        ckpt_idx = 0

        def _flush_and_log_checkpoint(image_name_str, pnp_resid=np.nan, qnp_resid=np.nan, qnp_kf_resid=np.nan):
            nonlocal ckpt_idx, rows_since_ckpt, pnp_header_written, qnp_header_written
            ckpt_idx += 1

            if pnp_batch:
                pd.DataFrame(pnp_batch).to_csv(
                    out_pnp,
                    mode="a" if pnp_header_written else "w",
                    index=False,
                    header=not pnp_header_written,
                )
                pnp_header_written = True
                pnp_batch.clear()

            if qnp_batch:
                pd.DataFrame(qnp_batch).to_csv(
                    out_qnp,
                    mode="a" if qnp_header_written else "w",
                    index=False,
                    header=not qnp_header_written,
                )
                qnp_header_written = True
                qnp_batch.clear()

            rows_since_ckpt = 0

            LOG.info(
                "Checkpoint %d: Reproj norms [%s]: PnP=%.3f, QnP=%.3f, QnP-KF=%s",
                ckpt_idx,
                image_name_str,
                float(pnp_resid) if np.isfinite(pnp_resid) else float("nan"),
                float(qnp_resid) if np.isfinite(qnp_resid) else float("nan"),
                f"{float(qnp_kf_resid):.3f}" if np.isfinite(qnp_kf_resid) else "nan",
            )

        # -------------------- feature id discovery (your header format) --------------------
        cols = set(df.columns)
        feat_ids: set[int] = set()
        for col in cols:
            if not col.startswith("feat_"):
                continue
            parts = col.split("_")
            # feat_<id>_x_distPX  OR feat_<id>_x_undistPX
            if len(parts) >= 4:
                try:
                    feat_ids.add(int(parts[1]))
                except Exception:
                    pass
        feat_ids = sorted(feat_ids)

        if not feat_ids:
            LOG.error("run_pnp_qnp_from_detection_csv: no feat_* columns found in %s", csv_path)
            return

        def _valid(v) -> bool:
            # -1.0 is your "no detection" sentinel
            try:
                if v is None:
                    return False
                fv = float(v)
                return np.isfinite(fv) and (fv > -0.5)
            except Exception:
                return False

        # ------------------------------------------------------------------
        # Reprojection residual metric (same as your current version)
        # ------------------------------------------------------------------
        _BIG_SIGMA = 1e6
        _MIN_SIGMA = 1e-6

        def _prep_sigma_2N(sigma_2N, N: int) -> np.ndarray | None:
            if sigma_2N is None:
                return None
            s = np.asarray(sigma_2N, dtype=np.float64).ravel()
            if s.size == N:
                s = np.repeat(s, 2)
            if s.size != 2 * N:
                return None
            bad = ~np.isfinite(s) | (s <= 0.0)
            if np.any(bad):
                s = s.copy()
                s[bad] = _BIG_SIGMA
            return np.maximum(s, _MIN_SIGMA)

        def _reproj_metrics_from_proj_meas(proj_xy: np.ndarray, meas_xy: np.ndarray, sigma_2N=None,
                                           mode: str = "chi") -> float:
            if proj_xy is None or meas_xy is None:
                return float("nan")
            if proj_xy.shape != meas_xy.shape or proj_xy.ndim != 2 or proj_xy.shape[1] != 2:
                return float("nan")

            r = (meas_xy.astype(np.float64) - proj_xy.astype(np.float64)).ravel()
            N = proj_xy.shape[0]
            if r.size != 2 * N:
                return float("nan")

            if mode == "rms_px":
                return float(np.sqrt(np.mean(r * r)))

            s = _prep_sigma_2N(sigma_2N, N)
            if s is None:
                return float(np.linalg.norm(r))

            rw = r / s
            if mode == "nrms":
                return float(np.sqrt(np.mean(rw * rw)))
            return float(np.sqrt(np.sum(rw * rw)))

        # imports local so data_processing stays “lighter” at import time
        import cv2
        from support.mathHelpers.quaternions import mat2quat  # adjust if your path differs
        from support.mathHelpers.quaternions import Quaternion as q  # must contain fromOpenCV_toAftr_rvec
        from support.mathHelpers.twoD_to_threeD import solveQnP  # adjust if your solveQnP lives elsewhere

        K = calibration.getCameraMatrix()
        D_full = calibration.getDistortion()

        q_aftr_from_cv = mat2quat(np.array([
            [0., 0., 1.],
            [-1., 0., 0.],
            [0., -1., 0.],
        ], dtype=float))

        resid_stats = {"pnp_unw": [], "pnp_wt": [], "qnp_unw": [], "qnp_kf": []}

        prev_qnp_q = prev_qnp_t = None
        prev_qnp_kf_q = prev_qnp_kf_t = None

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            if cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set():
                LOG.info("PnP/QnP canceled at row %d/%d", idx, total_rows)
                break

            image_name = row.get("image_name", "")
            image_time = row.get("image_time", np.nan)
            image_name_str = str(image_name)

            if progress_cb is not None:
                try:
                    progress_cb(idx, total_rows, image_name_str)
                except Exception:
                    pass

            if image_name_str in already_done:
                continue

            # --------------------------
            # Build point sets from your actual header format
            # Prefer undist if present; else use dist
            # --------------------------
            centers: list[list[float]] = []
            ids_use: list[int] = []
            use_ud = True

            ud_centers: list[list[float]] = []
            ud_ids: list[int] = []
            dist_centers: list[list[float]] = []
            dist_ids: list[int] = []

            for fid in feat_ids:
                ux = row.get(f"feat_{fid}_x_undistPX", None)
                uy = row.get(f"feat_{fid}_y_undistPX", None)
                if _valid(ux) and _valid(uy):
                    ud_centers.append([float(ux), float(uy)])
                    ud_ids.append(fid)
                    continue

                dx = row.get(f"feat_{fid}_x_distPX", None)
                dy = row.get(f"feat_{fid}_y_distPX", None)
                if _valid(dx) and _valid(dy):
                    dist_centers.append([float(dx), float(dy)])
                    dist_ids.append(fid)

            if len(ud_centers) >= 6:
                centers = ud_centers
                ids_use = ud_ids
                use_ud = True
            elif len(dist_centers) >= 6:
                centers = dist_centers
                ids_use = dist_ids
                use_ud = False
            else:
                # Not enough features -> emit NaNs
                row_pnp = {
                    "image_name": image_name, "image_time": image_time,
                    "pnp_qw": np.nan, "pnp_qx": np.nan, "pnp_qy": np.nan, "pnp_qz": np.nan,
                    "pnp_x": np.nan, "pnp_y": np.nan, "pnp_z": np.nan,
                }
                row_qnp = {
                    "image_name": image_name, "image_time": image_time,
                    "qnp_used_n": np.nan,
                    "qnp_qw": np.nan, "qnp_qx": np.nan, "qnp_qy": np.nan, "qnp_qz": np.nan,
                    "qnp_x": np.nan, "qnp_y": np.nan, "qnp_z": np.nan,
                    "qnp_s2": np.nan, "qnp_dof": np.nan, "qnp_sse_w": np.nan,
                    "qnp_sig_rx": np.nan, "qnp_sig_ry": np.nan, "qnp_sig_rz": np.nan,
                    "qnp_sig_tx": np.nan, "qnp_sig_ty": np.nan, "qnp_sig_tz": np.nan,
                    "qnp_kf_used_n": np.nan,
                    "qnp_kf_qw": np.nan, "qnp_kf_qx": np.nan, "qnp_kf_qy": np.nan, "qnp_kf_qz": np.nan,
                    "qnp_kf_x": np.nan, "qnp_kf_y": np.nan, "qnp_kf_z": np.nan,
                    "qnp_kf_s2": np.nan, "qnp_kf_dof": np.nan, "qnp_kf_sse_w": np.nan,
                    "qnp_kf_sig_rx": np.nan, "qnp_kf_sig_ry": np.nan, "qnp_kf_sig_rz": np.nan,
                    "qnp_kf_sig_tx": np.nan, "qnp_kf_sig_ty": np.nan, "qnp_kf_sig_tz": np.nan,
                }
                pnp_batch.append(row_pnp)
                qnp_batch.append(row_qnp)

                if checkpoint_every > 0:
                    rows_since_ckpt += 1
                    if rows_since_ckpt >= checkpoint_every:
                        _flush_and_log_checkpoint(image_name_str)
                continue

            # Map ids -> 3D truth points; drop unknown ids safely
            obj_pts = []
            img_pts = []
            kept_ids = []
            for fid, (u, v) in zip(ids_use, centers):
                rec = truth_dict[fid]

                # idsNamesLocs is commonly [id, name, x, y, z] -> take xyz
                # but tolerate it already being xyz
                if hasattr(rec, "__len__") and len(rec) >= 5:
                    xyz = rec[2:5]
                else:
                    xyz = rec

                obj_pts.append(xyz)
                img_pts.append([u, v])
                kept_ids.append(fid)

            if len(obj_pts) < 6:
                # emit NaNs (same as above)
                row_pnp = {
                    "image_name": image_name, "image_time": image_time,
                    "pnp_qw": np.nan, "pnp_qx": np.nan, "pnp_qy": np.nan, "pnp_qz": np.nan,
                    "pnp_x": np.nan, "pnp_y": np.nan, "pnp_z": np.nan,
                }
                row_qnp = {
                    "image_name": image_name, "image_time": image_time,
                    "qnp_used_n": np.nan,
                    "qnp_qw": np.nan, "qnp_qx": np.nan, "qnp_qy": np.nan, "qnp_qz": np.nan,
                    "qnp_x": np.nan, "qnp_y": np.nan, "qnp_z": np.nan,
                    "qnp_s2": np.nan, "qnp_dof": np.nan, "qnp_sse_w": np.nan,
                    "qnp_sig_rx": np.nan, "qnp_sig_ry": np.nan, "qnp_sig_rz": np.nan,
                    "qnp_sig_tx": np.nan, "qnp_sig_ty": np.nan, "qnp_sig_tz": np.nan,
                    "qnp_kf_used_n": np.nan,
                    "qnp_kf_qw": np.nan, "qnp_kf_qx": np.nan, "qnp_kf_qy": np.nan, "qnp_kf_qz": np.nan,
                    "qnp_kf_x": np.nan, "qnp_kf_y": np.nan, "qnp_kf_z": np.nan,
                    "qnp_kf_s2": np.nan, "qnp_kf_dof": np.nan, "qnp_kf_sse_w": np.nan,
                    "qnp_kf_sig_rx": np.nan, "qnp_kf_sig_ry": np.nan, "qnp_kf_sig_rz": np.nan,
                    "qnp_kf_sig_tx": np.nan, "qnp_kf_sig_ty": np.nan, "qnp_kf_sig_tz": np.nan,
                }
                pnp_batch.append(row_pnp)
                qnp_batch.append(row_qnp)

                if checkpoint_every > 0:
                    rows_since_ckpt += 1
                    if rows_since_ckpt >= checkpoint_every:
                        _flush_and_log_checkpoint(image_name_str)
                continue

            obj_pts = np.asarray(obj_pts, dtype=np.float32)
            img_pts = np.asarray(img_pts, dtype=np.float32)

            # ------------------------------------------------------------------
            # Kalman trust weights for this row (if available) - keyed by kept_ids
            # ------------------------------------------------------------------
            sigma_2N = None
            kf_img_pts = []
            width, height = calibration.width, calibration.height

            if kalman_available:
                row_kf = df_kf.iloc[idx - 1]
                sig_2N_list: list[float] = []
                any_valid = False

                for fid in kept_ids:
                    px_name = f"feat_{fid}_kf_x"
                    py_name = f"feat_{fid}_kf_y"
                    px_sig_name = f"feat_{fid}_kf_sigma_px"
                    py_sig_name = f"feat_{fid}_kf_sigma_py"

                    try:
                        sx = float(row_kf.get(px_sig_name, np.nan))
                    except Exception:
                        sx = float("nan")
                    try:
                        sy = float(row_kf.get(py_sig_name, np.nan))
                    except Exception:
                        sy = float("nan")

                    try:
                        kf_u, kf_v = float(row_kf.get(px_name, np.nan)), float(row_kf.get(py_name, np.nan))
                    except Exception:
                        kf_u, kf_v = np.nan, np.nan

                    kf_img_pts.append([float(kf_u * width), float(kf_v * height)])

                    if np.isfinite(sx) and sx > 0.0 and np.isfinite(sy) and sy > 0.0:
                        any_valid = True
                        sig_2N_list.extend([sx, sy])
                    else:
                        sig_2N_list.extend([np.nan, np.nan])

                if any_valid:
                    sigma_2N = np.asarray(
                        [_BIG_SIGMA if (not np.isfinite(s)) else max(float(s), _MIN_SIGMA) for s in sig_2N_list],
                        dtype=np.float64,
                    )

            kf_img_pts = np.asarray(kf_img_pts, dtype=np.float32)

            # ----------------- PnP (OpenCV, RANSAC) -----------------
            distCoeffs = np.zeros((5, 1), dtype=np.float32) if use_ud else D_full


            rvec = tvec = None
            quatPnP = vectPnP = None
            ret = False
            t0 = time.perf_counter()
            try:
                ret, rvec, tvec, _inliers = cv2.solvePnPRansac(
                    objectPoints=obj_pts,
                    imagePoints=img_pts,
                    cameraMatrix=K,
                    distCoeffs=distCoeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            except cv2.error as e:
                LOG.error("solvePnPRansac failed for %s: %s", image_name, e)
                ret = False
            t_pnp += time.perf_counter() - t0
            if ret:
                quatPnP, vectPnP = q.fromOpenCV_toAftr_rvec(rvec, tvec)
                row_pnp = {
                    "image_name": image_name, "image_time": image_time,
                    "pnp_qw": float(quatPnP.s),
                    "pnp_qx": float(quatPnP.vec[0]),
                    "pnp_qy": float(quatPnP.vec[1]),
                    "pnp_qz": float(quatPnP.vec[2]),
                    "pnp_x": float(vectPnP[0]),
                    "pnp_y": float(vectPnP[1]),
                    "pnp_z": float(vectPnP[2]),
                }
            else:
                row_pnp = {
                    "image_name": image_name, "image_time": image_time,
                    "pnp_qw": np.nan, "pnp_qx": np.nan, "pnp_qy": np.nan, "pnp_qz": np.nan,
                    "pnp_x": np.nan, "pnp_y": np.nan, "pnp_z": np.nan,
                }

            pnp_batch.append(row_pnp)

            # ----------------- QnP (unweighted + KF-weighted) -----------------
            pnp_resid = float("nan")
            qnp_resid = float("nan")
            qnp_kf_resid = float("nan")
            t0 = time.perf_counter()
            try:
                quatQ, vectQ, stats = solveQnP(
                    obj_pts,
                    img_pts,
                    calibration,
                    True,
                    None,
                    # user_seed_q=prev_qnp_q,
                    # user_seed_t=prev_qnp_t,
                    user_seed_q=None,
                    user_seed_t=None,
                )
                t_qnp += time.perf_counter() - t0

                prev_qnp_q, prev_qnp_t = quatQ, vectQ

                quatQ_kf = vectQ_kf = None
                kf_stats = None
                if sigma_2N is not None:
                    try:
                        t0 = time.perf_counter()
                        quatQ_kf, vectQ_kf, kf_stats = solveQnP(
                            obj_pts,
                            img_pts,
                            calibration,
                            True,
                            sigma_2N,
                            # user_seed_q=prev_qnp_kf_q,
                            # user_seed_t=prev_qnp_kf_t,
                            user_seed_q=None,
                            user_seed_t=None,
                        )
                        t_qnp_kf += time.perf_counter() - t0
                        prev_qnp_kf_q, prev_qnp_kf_t = quatQ_kf, vectQ_kf
                    except Exception as e_kf:
                        LOG.error("solveQnP (Kalman-weighted) failed for %s: %s", image_name, e_kf)
                        quatQ_kf = vectQ_kf = None
                        kf_stats = None

                # Convert to AFTR convention
                quatQ_aftr = q_aftr_from_cv * quatQ
                vectQ_aftr = q_aftr_from_cv * vectQ

                if quatQ_kf is not None and vectQ_kf is not None and kf_stats is not None:
                    quatQ_kf_aftr = q_aftr_from_cv * quatQ_kf
                    vectQ_kf_aftr = q_aftr_from_cv * vectQ_kf
                    kf_fields = {
                        "qnp_kf_qw": float(quatQ_kf_aftr.s),
                        "qnp_kf_qx": float(quatQ_kf_aftr.vec[0]),
                        "qnp_kf_qy": float(quatQ_kf_aftr.vec[1]),
                        "qnp_kf_qz": float(quatQ_kf_aftr.vec[2]),
                        "qnp_kf_x": float(vectQ_kf_aftr[0]),
                        "qnp_kf_y": float(vectQ_kf_aftr[1]),
                        "qnp_kf_z": float(vectQ_kf_aftr[2]),
                        "qnp_kf_used_n": int(kf_stats.N),
                        "qnp_kf_s2": float(kf_stats.s2),
                        "qnp_kf_dof": int(kf_stats.dof),
                        "qnp_kf_sse_w": float(kf_stats.sse_w),
                        "qnp_kf_sig_rx": float(np.sqrt(kf_stats.cov6[0, 0])),
                        "qnp_kf_sig_ry": float(np.sqrt(kf_stats.cov6[1, 1])),
                        "qnp_kf_sig_rz": float(np.sqrt(kf_stats.cov6[2, 2])),
                        "qnp_kf_sig_tx": float(np.sqrt(kf_stats.cov6[3, 3])),
                        "qnp_kf_sig_ty": float(np.sqrt(kf_stats.cov6[4, 4])),
                        "qnp_kf_sig_tz": float(np.sqrt(kf_stats.cov6[5, 5])),
                    }
                else:
                    kf_fields = {
                        "qnp_kf_qw": np.nan, "qnp_kf_qx": np.nan, "qnp_kf_qy": np.nan, "qnp_kf_qz": np.nan,
                        "qnp_kf_x": np.nan, "qnp_kf_y": np.nan, "qnp_kf_z": np.nan,
                        "qnp_kf_used_n": np.nan,
                        "qnp_kf_s2": np.nan, "qnp_kf_dof": np.nan, "qnp_kf_sse_w": np.nan,
                        "qnp_kf_sig_rx": np.nan, "qnp_kf_sig_ry": np.nan, "qnp_kf_sig_rz": np.nan,
                        "qnp_kf_sig_tx": np.nan, "qnp_kf_sig_ty": np.nan, "qnp_kf_sig_tz": np.nan,
                    }

                row_qnp = {
                    "image_name": image_name, "image_time": image_time,
                    "qnp_qw": float(quatQ_aftr.s),
                    "qnp_qx": float(quatQ_aftr.vec[0]),
                    "qnp_qy": float(quatQ_aftr.vec[1]),
                    "qnp_qz": float(quatQ_aftr.vec[2]),
                    "qnp_x": float(vectQ_aftr[0]),
                    "qnp_y": float(vectQ_aftr[1]),
                    "qnp_z": float(vectQ_aftr[2]),
                    "qnp_used_n": int(stats.N),
                    "qnp_s2": float(stats.s2),
                    "qnp_dof": int(stats.dof),
                    "qnp_sse_w": float(stats.sse_w),
                    "qnp_sig_rx": float(np.sqrt(stats.cov6[0, 0])),
                    "qnp_sig_ry": float(np.sqrt(stats.cov6[1, 1])),
                    "qnp_sig_rz": float(np.sqrt(stats.cov6[2, 2])),
                    "qnp_sig_tx": float(np.sqrt(stats.cov6[3, 3])),
                    "qnp_sig_ty": float(np.sqrt(stats.cov6[4, 4])),
                    "qnp_sig_tz": float(np.sqrt(stats.cov6[5, 5])),
                    **kf_fields,
                }

                # --- residuals (kept identical in spirit; uses your whitening helper) ---
                # pnp_resid uses unweighted
                if ret and rvec is not None and tvec is not None:
                    proj, _ = cv2.projectPoints(
                        obj_pts.astype(np.float32),
                        rvec.astype(np.float64),
                        tvec.astype(np.float64),
                        K.astype(np.float64),
                        distCoeffs.astype(np.float64) if distCoeffs is not None else None,
                    )
                    pnp_resid = _reproj_metrics_from_proj_meas(proj.reshape(-1, 2), img_pts.reshape(-1, 2), None, "chi")

                # qnp_resid (unweighted)
                X_cam = quatQ * obj_pts + vectQ
                Z = np.where(X_cam[:, 2] > 1e-6, X_cam[:, 2], 1e-6)
                u = calibration.fx * (X_cam[:, 0] / Z) + calibration.cx
                v = calibration.fy * (X_cam[:, 1] / Z) + calibration.cy
                qnp_resid = _reproj_metrics_from_proj_meas(np.column_stack([u, v]), img_pts.reshape(-1, 2), None, "chi")

                resid_stats["pnp_unw"].append(pnp_resid)
                resid_stats["qnp_unw"].append(qnp_resid)

                # qnp_kf_resid (weighted)
                if sigma_2N is not None and quatQ_kf is not None and vectQ_kf is not None:
                    X_cam_kf = quatQ_kf * obj_pts + vectQ_kf
                    Zk = np.where(X_cam_kf[:, 2] > 1e-6, X_cam_kf[:, 2], 1e-6)
                    uk = calibration.fx * (X_cam_kf[:, 0] / Zk) + calibration.cx
                    vk = calibration.fy * (X_cam_kf[:, 1] / Zk) + calibration.cy
                    qnp_kf_resid = _reproj_metrics_from_proj_meas(
                        np.column_stack([uk, vk]), img_pts.reshape(-1, 2), sigma_2N, "chi"
                    )
                    resid_stats["qnp_kf"].append(qnp_kf_resid)

                    # PnP weighted metric (for reporting only)
                    if ret and rvec is not None and tvec is not None:
                        proj, _ = cv2.projectPoints(
                            obj_pts.astype(np.float32),
                            rvec.astype(np.float64),
                            tvec.astype(np.float64),
                            K.astype(np.float64),
                            distCoeffs.astype(np.float64) if distCoeffs is not None else None,
                        )
                        pnp_resid_w = _reproj_metrics_from_proj_meas(proj.reshape(-1, 2), img_pts.reshape(-1, 2),
                                                                     sigma_2N, "chi")
                    else:
                        pnp_resid_w = float("nan")
                else:
                    pnp_resid_w = float("nan")
                resid_stats["pnp_wt"].append(pnp_resid_w)

            except Exception as e:
                LOG.error("solveQnP failed for %s: %s", image_name, e)
                row_qnp = {
                    "image_name": image_name, "image_time": image_time,
                    "qnp_used_n": np.nan,
                    "qnp_qw": np.nan, "qnp_qx": np.nan, "qnp_qy": np.nan, "qnp_qz": np.nan,
                    "qnp_x": np.nan, "qnp_y": np.nan, "qnp_z": np.nan,
                    "qnp_s2": np.nan, "qnp_dof": np.nan, "qnp_sse_w": np.nan,
                    "qnp_sig_rx": np.nan, "qnp_sig_ry": np.nan, "qnp_sig_rz": np.nan,
                    "qnp_sig_tx": np.nan, "qnp_sig_ty": np.nan, "qnp_sig_tz": np.nan,
                    "qnp_kf_used_n": np.nan,
                    "qnp_kf_qw": np.nan, "qnp_kf_qx": np.nan, "qnp_kf_qy": np.nan, "qnp_kf_qz": np.nan,
                    "qnp_kf_x": np.nan, "qnp_kf_y": np.nan, "qnp_kf_z": np.nan,
                    "qnp_kf_s2": np.nan, "qnp_kf_dof": np.nan, "qnp_kf_sse_w": np.nan,
                    "qnp_kf_sig_rx": np.nan, "qnp_kf_sig_ry": np.nan, "qnp_kf_sig_rz": np.nan,
                    "qnp_kf_sig_tx": np.nan, "qnp_kf_sig_ty": np.nan, "qnp_kf_sig_tz": np.nan,
                }

            qnp_batch.append(row_qnp)

            # Count exactly once per processed row
            if checkpoint_every > 0:
                rows_since_ckpt += 1
                if rows_since_ckpt >= checkpoint_every:
                    _flush_and_log_checkpoint(image_name_str, pnp_resid, qnp_resid, qnp_kf_resid)

            n_pose += 1

        # Final flush
        LOG.info("Final CSV update...")
        if pnp_batch:
            pd.DataFrame(pnp_batch).to_csv(
                out_pnp,
                mode="a" if pnp_header_written else "w",
                index=False,
                header=not pnp_header_written,
            )
        if qnp_batch:
            pd.DataFrame(qnp_batch).to_csv(
                out_qnp,
                mode="a" if qnp_header_written else "w",
                index=False,
                header=not qnp_header_written,
            )

        # Summaries
        def _summ(vals: list[float]) -> str:
            arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
            if arr.size == 0:
                return "n/a"
            return f"n={arr.size}, mean={arr.mean():.3f}, median={np.median(arr):.3f}, min={arr.min():.3f}, max={arr.max():.3f}"

        if resid_stats["pnp_unw"]:
            LOG.info("Reproj norm summary (unweighted PnP):  %s", _summ(resid_stats["pnp_unw"]))
        if resid_stats["pnp_wt"]:
            LOG.info("Reproj norm summary (weighted PnP):    %s", _summ(resid_stats["pnp_wt"]))
        if resid_stats["qnp_unw"]:
            LOG.info("Reproj norm summary (unweighted QnP):  %s", _summ(resid_stats["qnp_unw"]))
        if resid_stats["qnp_kf"]:
            LOG.info("Reproj norm summary (KF-weighted QnP): %s", _summ(resid_stats["qnp_kf"]))

        LOG.info("run_pnp_qnp_from_detection_csv: finished (%s -> %s, %s)", csv_path, out_pnp, out_qnp)

        if n_pose > 0:
            LOG.info(
                "Pose solve timing (avg per frame): "
                "PnP=%.3f ms | QnP=%.3f ms | QnP-KF=%.3f ms",
                1000.0 * t_pnp / n_pose,
                1000.0 * t_qnp / n_pose,
                1000.0 * t_qnp_kf / max(1, n_pose),
            )

