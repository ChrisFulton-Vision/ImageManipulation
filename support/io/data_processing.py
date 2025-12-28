from __future__ import annotations
import re
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import numpy as np

import cv2
from concurrent.futures import ThreadPoolExecutor, wait

from support.core.pixel_kalmanFilter import KalmanFilter as PixelKalmanFilter
from support.io.my_logging import LOG


ProgressCb = Callable[[float, str], None]   # (overall_frac, text)
StatusCb   = Callable[[str], None]
FinishCb   = Callable[[str], None]

@dataclass
class YoloSweepParams:
    img_dir: Path
    out_csv_base: Path                 # .../_ProcessedData/1_yolo_detections.csv (base name)
    conf_list_var: object              # Tk var or string; parsed by parse_conf_list()
    ckpt_every_var: object | None      # Tk var or int/str
    prefetch_var: object | None        # Tk var or int/str
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
        sweep_timer,                    # your utils.SweepTimer instance
        fmt_mmss: Callable[[float], str],
        post_progress: ProgressCb,
        post_status: StatusCb,
        post_finish: FinishCb,
        distort_points_px: Callable[[object, tuple[float, float]], tuple[float, float]],
    ) -> None:
        """
        Runs YOLO across all images for each conf in conf_list.
        Writes one CSV per confidence: ..._conf0.80.csv etc.
        """
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
                                xp, yp = distort_points_px(calibration, (x, y))
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

                    if checkpoint_every > 0 and (not self.cancel_event.is_set()) and processed_since_ckpt >= checkpoint_every:
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

        inv_w = 1.0 / max(width, 1.0)
        inv_h = 1.0 / max(height, 1.0)
        Xmeas = Xraw * inv_w
        Ymeas = Yraw * inv_h

        valid = np.isfinite(Xmeas) & np.isfinite(Ymeas) & (Xmeas != -1.0) & (Ymeas != -1.0)
        valid_u8 = valid.astype(np.uint8, copy=False)

        # --- KF parameter seed ---
        kf0 = PixelKalmanFilter()
        kf0.set_image_size(width, height)
        kf0.set_sigma_meas_px(1.0, 1.0)
        kf0.set_max_pixel_jump_px(500.0)
        kf0.max_mahalanobis_sq = 13.82

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