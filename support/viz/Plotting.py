import pandas as pd
import matplotlib.pyplot as plt
from support.io.my_logging import LOG
from pathlib import Path
import numpy as np

class Plotter:
    @staticmethod
    def plot(conf: float, img_dir: Path, show: bool = True, save: bool = False):

        dir_path = img_dir / Path(f"{conf:.2f}")

        plt.rcParams['figure.max_open_warning'] = 30
        plt.rcParams['figure.autolayout'] = True

        # Input CSVs
        try:
            pnp = pd.read_csv(img_dir / f"3_pnp_conf{conf:.2f}.csv")
        except FileNotFoundError:
            LOG.warning(fr'Tried to find ' + str(img_dir / f"3_pnp_conf{conf:.2f}.csv"))
            return
        try:
            qnp = pd.read_csv(img_dir / f"4_qnp_conf{conf:.2f}.csv")
        except FileNotFoundError:
            LOG.warning(fr'Tried to find ' + str(img_dir / f"4_qnp_conf{conf:.2f}.csv"))
            return

        # Optional KF CSV (for NIS diagnostics)
        kf = None
        try:
            kf = pd.read_csv(img_dir / f"2_kalman_conf{conf:.2f}.csv")
        except FileNotFoundError:
            kf = None

        # Merge on image name + time
        merged = pd.merge(
            pnp,
            qnp,
            on=["image_name", "image_time"],
            how="inner",
            suffixes=("_pnp", "_qnp"),
        )

        merged_kf = None
        if kf is not None:
            merged_kf = pd.merge(
                merged,
                kf,
                on=["image_name", "image_time"],
                how="left",
            )
        else:
            merged_kf = merged

        t = merged["image_time"]

        def _pick_time_col(df, preferred=("image_time", "img_time", "time", "t")):
            for c in preferred:
                if c in df.columns:
                    return c
            raise KeyError(f"No time column found. Tried: {preferred}. Have: {list(df.columns)[:20]}...")

        def _merge_kf_used_rate_into_merged(merged: pd.DataFrame, kf_csv_path) -> pd.DataFrame:
            kf = pd.read_csv(kf_csv_path)

            # Find time columns
            t_merged = _pick_time_col(merged)
            t_kf = _pick_time_col(kf)

            if "kf_used_rate" not in kf.columns:
                raise KeyError(
                    f"'kf_used_rate' not found in KF CSV. "
                    f"Available columns include: {list(kf.columns)[:30]}..."
                )

            kf_small = kf[[t_kf, "kf_used_rate"]].copy()

            # Ensure numeric + sorted for merge_asof
            merged2 = merged.copy()
            merged2[t_merged] = pd.to_numeric(merged2[t_merged], errors="coerce")
            kf_small[t_kf] = pd.to_numeric(kf_small[t_kf], errors="coerce")
            merged2 = merged2.sort_values(t_merged)
            kf_small = kf_small.sort_values(t_kf)

            # Rename KF time column to match merged
            if t_kf != t_merged:
                kf_small = kf_small.rename(columns={t_kf: t_merged})

            # Merge (nearest time match)
            merged2 = pd.merge_asof(
                merged2,
                kf_small,
                on=t_merged,
                direction="nearest",
                tolerance=1e-3,  # adjust if needed (see note below)
            )

            return merged2

        # Detect if KF-weighted QnP is available
        has_kf_pos = all(f"qnp_kf_{c}" in merged.columns for c in ["x", "y", "z"])
        has_kf_quat = all(f"qnp_kf_{c}" in merged.columns for c in ["qw", "qx", "qy", "qz"])
        has_kf = has_kf_pos and has_kf_quat

        # -------------------------------
        # Position comparison: PnP vs QnP vs QnP-KF
        # -------------------------------
        for comp in ["x", "y", "z"]:
            plt.figure()
            plt.plot(t, merged[f"pnp_{comp}"], label="PnP")
            plt.plot(t, merged[f"qnp_{comp}"], label="QnP")
            if has_kf_pos:
                plt.plot(t, merged[f"qnp_kf_{comp}"], label="QnP (KF-weighted)")

            plt.xlabel("Time [s]")
            plt.ylabel(f"{comp.upper()} position")
            # plt.title(f"{comp.upper()} Position vs Time")
            plt.legend()
            plt.grid(True)

            if save:
                if not Path.is_dir(dir_path):
                    Path.mkdir(dir_path)
                plt.savefig(dir_path / Path(f"1_{comp.upper()}_Position_vs_Time.pdf"))
                if not show:
                    plt.close()

        # -------------------------------
        # Quaternion components: PnP vs QnP vs QnP-KF
        # -------------------------------
        for comp in ["qw", "qx", "qy", "qz"]:
            plt.figure()
            plt.plot(t, merged[f"pnp_{comp}"], label="PnP")
            plt.plot(t, merged[f"qnp_{comp}"], label="QnP")
            if has_kf_quat:
                plt.plot(t, merged[f"qnp_kf_{comp}"], label="QnP (KF-weighted)")

            plt.xlabel("Time [s]")
            # plt.ylabel(comp.upper())
            # plt.title(f"Quaternion component {comp.upper()} vs Time")
            plt.legend()
            plt.grid(True)

            if save:
                plt.savefig(dir_path / Path(f"2_{comp.upper()}_Quaternion_comp_vs_Time.pdf"))
                if not show:
                    plt.close()

        # -------------------------------
        # Position differences vs PnP
        # -------------------------------
        plt.figure()
        dt = merged[[f"qnp_{c}" for c in "xyz"]].values - \
             merged[[f"pnp_{c}" for c in "xyz"]].values
        plt.plot(
            t,
            np.linalg.norm(dt,axis=1),
            label=f"||Δt|| (QnP − PnP)",
        )
        if has_kf_pos:
            dt_kf = merged[[f"qnp_kf_{c}" for c in "xyz"]].values - \
                 merged[[f"pnp_{c}" for c in "xyz"]].values
            plt.plot(
                t,
                np.linalg.norm(dt_kf,axis=1),
                linestyle="--",
                label=f"||Δt|| (QnP-KF - PnP)",
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Position difference")
        # plt.title("Position Difference vs Time (relative to PnP)")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(dir_path / Path(f"3_Position_Diff_vs_time.pdf"))
            if not show:
                plt.close()

        # -------------------------------
        # Quaternion component differences vs PnP
        # -------------------------------
        def quat_conj(q):
            return np.column_stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]])

        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1.T
            w2, x2, y2, z2 = q2.T
            return np.column_stack([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])

        def calc_theta(q1, q2):
            q_rel = quat_mul(q1, quat_conj(q2))
            qw = np.clip(np.abs(q_rel[:, 0]), -1.0, 1.0)
            return np.rad2deg(2.0 * np.arccos(qw))

        q_qnp = merged[[f"qnp_q{c}" for c in "wxyz"]].values
        q_pnp = merged[[f"pnp_q{c}" for c in "wxyz"]].values
        plt.figure()
        q_rel = calc_theta(q_qnp, q_pnp)
        plt.plot(
            t,
            q_rel,
            label=f"Δθ (QnP - PnP)",
        )
        if has_kf_quat:
            qKF_pnp = merged[[f"qnp_kf_q{c}" for c in "wxyz"]].values
            qKF_rel = calc_theta(qKF_pnp, q_pnp)
            plt.plot(
                t,
                qKF_rel,
                linestyle="--",
                label=f"Δθ (QnP-KF - PnP)",
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Quaternion angular difference (deg)")
        # plt.title("Quaternion Component Difference vs Time (relative to PnP)")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(dir_path / Path(f"4_Quaternion_Diff_vs_time.pdf"))
            if not show:
                plt.close()

        # -------------------------------
        # Optional: QnP-KF - QnP comparison
        # -------------------------------
        if has_kf:
            # Position
            plt.figure()
            for comp in ["x", "y", "z"]:
                plt.plot(
                    t,
                    merged[f"qnp_kf_{comp}"] - merged[f"qnp_{comp}"],
                    label=f"{comp.upper()} (QnP-KF - QnP)",
                )
            plt.xlabel("Time [s]")
            plt.ylabel("Position difference")
            # plt.title("Position Difference: QnP-KF - QnP")
            plt.legend()
            plt.grid(True)

            if save:
                plt.savefig(dir_path / Path(f"5_Position_Diff_QnpW_v_QnpU.pdf"))
                if not show:
                    plt.close()

            # Quaternion
            plt.figure()
            for comp in ["qw", "qx", "qy", "qz"]:
                plt.plot(
                    t,
                    merged[f"qnp_kf_{comp}"] - merged[f"qnp_{comp}"],
                    label=f"{comp.upper()} (QnP-KF - QnP)",
                )
            plt.xlabel("Time [s]")
            plt.ylabel("Quaternion component difference")
            # plt.title("Quaternion Component Difference: QnP-KF - QnP")
            plt.legend()
            plt.grid(True)

            if save:
                plt.savefig(dir_path / Path(f"6_Quat_Diff_QnpW_v_QnpU.pdf"))
                if not show:
                    plt.close()

        # ============================================================
        # NEW: SolveQnP covariance / uncertainty diagnostics (QnP vs QnP-KF)
        # ============================================================

        # Availability checks
        have_qnp_stats = all(c in merged.columns for c in ["qnp_used_n", "qnp_s2", "qnp_dof", "qnp_sse_w"])
        have_qnp_sig = all(c in merged.columns for c in [
            "qnp_sig_rx", "qnp_sig_ry", "qnp_sig_rz", "qnp_sig_tx", "qnp_sig_ty", "qnp_sig_tz"
        ])

        have_qnp_kf_stats = all(
            c in merged.columns for c in ["qnp_kf_used_n", "qnp_kf_s2", "qnp_kf_dof", "qnp_kf_sse_w"])
        have_qnp_kf_sig = all(c in merged.columns for c in [
            "qnp_kf_sig_rx", "qnp_kf_sig_ry", "qnp_kf_sig_rz", "qnp_kf_sig_tx", "qnp_kf_sig_ty", "qnp_kf_sig_tz"
        ])

        # s2 over time (key “whitened residual sanity” plot)
        if have_qnp_stats or have_qnp_kf_stats:
            plt.figure()
            if have_qnp_stats:
                plt.plot(t, merged["qnp_s2"], label="QnP s2")
            if have_qnp_kf_stats:
                plt.plot(t, merged["qnp_kf_s2"], label="QnP-KF s2")
            plt.axhline(1.0, linestyle="--", label="target ~1 (whitened)")
            plt.xlabel("Time [s]")
            plt.ylabel("s2 = SSE_w / dof")
            # plt.title("SolveQnP Residual Scale (s2) vs Time")
            plt.grid(True)
            plt.legend()

            if save:
                plt.savefig(dir_path / Path(f"7_QnP_Residual_vs_time.pdf"))
                if not show:
                    plt.close()

        # Bring kf_used_rate from merged_kf into merged (time-aligned)
        if ("kf_used_rate" not in merged.columns) and ("kf_used_rate" in merged_kf.columns):
            # choose time column names (adjust if yours differs)
            t_col_merged = "image_time" if "image_time" in merged.columns else "img_time"
            t_col_kf = "image_time" if "image_time" in merged_kf.columns else "img_time"

            kf_small = merged_kf[[t_col_kf, "kf_used_rate"]].copy()

            # ensure numeric + sorted
            merged = merged.sort_values(t_col_merged).copy()
            kf_small = kf_small.sort_values(t_col_kf).copy()
            merged[t_col_merged] = pd.to_numeric(merged[t_col_merged], errors="coerce")
            kf_small[t_col_kf] = pd.to_numeric(kf_small[t_col_kf], errors="coerce")

            # rename time column to match merged
            if t_col_kf != t_col_merged:
                kf_small = kf_small.rename(columns={t_col_kf: t_col_merged})

            merged = pd.merge_asof(
                merged,
                kf_small,
                on=t_col_merged,
                direction="nearest",
                tolerance=2e-2,  # 20 ms; tighten if possible
            )

            # used_n over time + KF rejected count (derived from kf_used_rate)
            if ("qnp_used_n" in merged.columns) and ("kf_used_rate" in merged.columns):
                plt.figure()

                n_used = merged["qnp_used_n"].to_numpy(dtype=float)
                kf_used_rate = merged["kf_used_rate"].to_numpy(dtype=float)

                # rejected fraction in [0,1]
                n_kf_rejected_eff = n_used * kf_used_rate

                plt.plot(t, n_used, label="QnP used_n")
                plt.plot(t, n_kf_rejected_eff, linestyle="--", label="QnP-KF used_n")

                plt.xlabel("Time [s]")
                plt.ylabel("N features")
                plt.grid(True)
                plt.legend()

                if save:
                    plt.savefig(dir_path / Path("8_KF_Features_used.pdf"))
                    if not show:
                        plt.close()

        # per-parameter 1-sigma time series
        if have_qnp_sig or have_qnp_kf_sig:
            # Rotation sigmas
            plt.figure()
            if have_qnp_sig:
                plt.plot(t, merged["qnp_sig_rx"], label="QnP sig_rx [rad]")
                plt.plot(t, merged["qnp_sig_ry"], label="QnP sig_ry [rad]")
                plt.plot(t, merged["qnp_sig_rz"], label="QnP sig_rz [rad]")
            if have_qnp_kf_sig:
                plt.plot(t, merged["qnp_kf_sig_rx"], linestyle="--", label="QnP-KF sig_rx [rad]")
                plt.plot(t, merged["qnp_kf_sig_ry"], linestyle="--", label="QnP-KF sig_ry [rad]")
                plt.plot(t, merged["qnp_kf_sig_rz"], linestyle="--", label="QnP-KF sig_rz [rad]")
            plt.xlabel("Time [s]")
            plt.ylabel("Rotation 1σ [rad]")
            # plt.title("SolveQnP Rotation Uncertainty (Rodrigues tangent) vs Time")
            plt.grid(True)
            plt.legend()

            if save:
                plt.savefig(dir_path / Path(f"9_QnP_Rot_Uncertainty_vs_time.pdf"))
                if not show:
                    plt.close()

            # Translation sigmas
            plt.figure()
            if have_qnp_sig:
                plt.plot(t, merged["qnp_sig_tx"], label="QnP sig_tx")
                plt.plot(t, merged["qnp_sig_ty"], label="QnP sig_ty")
                plt.plot(t, merged["qnp_sig_tz"], label="QnP sig_tz")
            if have_qnp_kf_sig:
                plt.plot(t, merged["qnp_kf_sig_tx"], linestyle="--", label="QnP-KF sig_tx")
                plt.plot(t, merged["qnp_kf_sig_ty"], linestyle="--", label="QnP-KF sig_ty")
                plt.plot(t, merged["qnp_kf_sig_tz"], linestyle="--", label="QnP-KF sig_tz")
            plt.xlabel("Time [s]")
            plt.ylabel("Translation 1σ [m]")
            # plt.title("SolveQnP Translation Uncertainty vs Time")
            plt.grid(True)
            plt.legend()

            if save:
                plt.savefig(dir_path / Path(f"10_QnP_Trans_Uncertainty_vs_time.pdf"))
                if not show:
                    plt.close()

        # combined magnitudes computed on the fly (no extra CSV columns)
        if have_qnp_sig or have_qnp_kf_sig:
            plt.figure()

            def _mag3(ax, ay, az):
                return np.sqrt(np.maximum(ax, 0.0) ** 2 + np.maximum(ay, 0.0) ** 2 + np.maximum(az, 0.0) ** 2)

            if have_qnp_sig:
                sig_r_mag = _mag3(merged["qnp_sig_rx"], merged["qnp_sig_ry"], merged["qnp_sig_rz"])
                sig_t_mag = _mag3(merged["qnp_sig_tx"], merged["qnp_sig_ty"], merged["qnp_sig_tz"])
                plt.plot(t, sig_r_mag, label="QnP ||sig_r|| [rad]")
                plt.plot(t, sig_t_mag, label="QnP ||sig_t|| [m]")

            if have_qnp_kf_sig:
                sig_r_mag_kf = _mag3(merged["qnp_kf_sig_rx"], merged["qnp_kf_sig_ry"], merged["qnp_kf_sig_rz"])
                sig_t_mag_kf = _mag3(merged["qnp_kf_sig_tx"], merged["qnp_kf_sig_ty"], merged["qnp_kf_sig_tz"])
                plt.plot(t, sig_r_mag_kf, linestyle="--", label="QnP-KF ||sig_r|| [rad]")
                plt.plot(t, sig_t_mag_kf, linestyle="--", label="QnP-KF ||sig_t|| [m]")

            plt.xlabel("Time [s]")
            plt.ylabel("Magnitude")
            # plt.title("SolveQnP Combined Uncertainty Magnitudes vs Time")
            plt.grid(True)
            plt.legend()

            if save:
                plt.savefig(dir_path / Path(f"11_QnP_Combined_Uncertainty_vs_time.pdf"))
                if not show:
                    plt.close()

            # Ratio plots (KF-weighted / unweighted) — nice for “improvement factor” story
            if have_qnp_sig and have_qnp_kf_sig:
                eps = 1e-12
                plt.figure()
                plt.plot(t, (sig_t_mag_kf + eps) / (sig_t_mag + eps), label="||sig_t||_KF / ||sig_t||")
                plt.plot(t, (sig_r_mag_kf + eps) / (sig_r_mag + eps), label="||sig_r||_KF / ||sig_r||")
                plt.axhline(1.0, linestyle="--", label="=1")
                plt.xlabel("Time [s]")
                plt.ylabel("Ratio")
                # plt.title("Uncertainty Ratio: KF-weighted vs Unweighted")
                plt.grid(True)
                plt.legend()

            if save:
                plt.savefig(dir_path / Path(f"12_UncertaintyRatio_QnpW_v_QnpU.pdf"))
                if not show:
                    plt.close()

        # -------------------------------
        # KF/NIS diagnostics (if present)
        # -------------------------------
        if merged_kf is not None:
            has_used_rate = "kf_used_rate" in merged_kf.columns
            has_nis_med = "kf_nis_med_used" in merged_kf.columns
            has_nis_p95 = "kf_nis_p95_used" in merged_kf.columns
            has_sig_px = "kf_sigma_meas_px" in merged_kf.columns
            has_sig_py = "kf_sigma_meas_py" in merged_kf.columns

            if has_used_rate:
                plt.figure()
                plt.plot(t, merged_kf["kf_used_rate"], label="KF used rate")
                plt.xlabel("Time [s]")
                plt.ylabel("Fraction used")
                # plt.title("KF Accepted Measurement Rate vs Time")
                plt.grid(True)
                plt.legend()

                if save:
                    plt.savefig(dir_path / Path(f"12_KF_Used_vs_time.pdf"))
                    if not show:
                        plt.close()

            if has_nis_med or has_nis_p95:
                plt.figure()
                if has_nis_med:
                    plt.plot(t, merged_kf["kf_nis_med_used"], label="NIS median (accepted)")
                if has_nis_p95:
                    plt.plot(t, merged_kf["kf_nis_p95_used"], label="NIS p95 (accepted)")
                plt.axhline(2.0, linestyle="--", label="E[NIS]=2")
                plt.axhline(5.991, linestyle="--", label="chi2_2 95% (5.991)")
                plt.axhline(9.21, linestyle="--", label="chi2_2 99% (9.21)")
                plt.xlabel("Time [s]")
                plt.ylabel("NIS")
                # plt.title("KF Normalized Innovation Squared (Accepted) vs Time")
                plt.grid(True)
                plt.legend()

                if save:
                    plt.savefig(dir_path / Path(f"13_KF_NIS_vs_time.pdf"))
                    if not show:
                        plt.close()

            if has_sig_px or has_sig_py:
                plt.figure()
                if has_sig_px:
                    plt.plot(t, merged_kf["kf_sigma_meas_px"], label="sigma_meas_x [px]")
                if has_sig_py:
                    plt.plot(t, merged_kf["kf_sigma_meas_py"], label="sigma_meas_y [px]")
                plt.xlabel("Time [s]")
                plt.ylabel("Sigma [px]")
                # plt.title("Estimated Measurement Noise Sigma vs Time")
                plt.grid(True)
                plt.legend()

                if save:
                    plt.savefig(dir_path / Path(f"14_Est_meas_noise_vs_time.pdf"))
                    if not show:
                        plt.close()

            nis_cols = [c for c in merged_kf.columns if c.endswith("_kf_nis")]
            used_cols = [c for c in merged_kf.columns if c.endswith("_kf_used")]

            if len(nis_cols) > 0 and len(used_cols) > 0:
                nis_vals = []
                for nis_c in nis_cols:
                    used_c = nis_c.replace("_kf_nis", "_kf_used")
                    if used_c not in merged_kf.columns:
                        continue
                    good = (merged_kf[used_c].to_numpy(dtype=float) > 0.5)
                    v = merged_kf[nis_c].to_numpy(dtype=float)
                    v = v[good & (v > 0.0)]
                    if v.size > 0:
                        nis_vals.append(v)

                if len(nis_vals) > 0:
                    nis_all = np.concatenate(nis_vals)

                    plt.figure()
                    plt.hist(nis_all, bins=60)
                    plt.axvline(2.0, linestyle="--", label="E[NIS]=2")
                    plt.axvline(5.991, linestyle="--", label="chi2_2 95%")
                    plt.axvline(9.21, linestyle="--", label="chi2_2 99%")
                    plt.xlabel("NIS")
                    plt.ylabel("Count")
                    # plt.title("Histogram of Accepted NIS (All Features)")
                    plt.grid(True)
                    plt.legend()

                    if save:
                        plt.savefig(dir_path / Path(f"15_NIS_Accepted_Histogram_vs_time.pdf"))
                        if not show:
                            plt.close()

            # ============================================================
            # NEW: "Brittleness / pollution" story plots
            #   - Confidence gain (σ_unw / σ_kf)
            #   - Link gain to KF rejection + innovation inconsistency
            #   - Heatmap of per-feature NIS/sigma with gain overlay
            # ============================================================

            # Helper: ensure output dir exists if saving
            def _ensure_dir():
                if save:
                    if not Path.is_dir(dir_path):
                        Path.mkdir(dir_path, parents=True, exist_ok=True)

            # We need QnP sigmas + QnP-KF sigmas to form confidence gain
            have_rot_sig = all(c in merged_kf.columns for c in ["qnp_sig_rx", "qnp_sig_ry", "qnp_sig_rz"])
            have_rot_sig_kf = all(c in merged_kf.columns for c in ["qnp_kf_sig_rx", "qnp_kf_sig_ry", "qnp_kf_sig_rz"])

            # KF rejection + innovation inconsistency summaries (frame-level)
            has_used_rate = "kf_used_rate" in merged_kf.columns
            has_nis_p95 = "kf_nis_p95_used" in merged_kf.columns
            has_nis_med = "kf_nis_med_used" in merged_kf.columns

            if have_rot_sig and have_rot_sig_kf:
                eps = 1e-12

                # Choose which rotational sigma to tell the story with:
                # - rz is often most interpretable
                # - or max(rx,ry,rz) is a robust "worst-axis" indicator
                sig_unw_rz = merged_kf["qnp_sig_rz"].to_numpy(dtype=float)
                sig_kf_rz = merged_kf["qnp_kf_sig_rz"].to_numpy(dtype=float)

                # Confidence gain: how much more "confident" the KF-weighted solve is
                conf_gain_rz = (sig_unw_rz + eps) / (sig_kf_rz + eps)

                # Also useful: a magnitude gain (optional)
                sig_unw_rmag = np.sqrt(
                    merged_kf["qnp_sig_rx"].to_numpy(dtype=float) ** 2 +
                    merged_kf["qnp_sig_ry"].to_numpy(dtype=float) ** 2 +
                    merged_kf["qnp_sig_rz"].to_numpy(dtype=float) ** 2
                )
                sig_kf_rmag = np.sqrt(
                    merged_kf["qnp_kf_sig_rx"].to_numpy(dtype=float) ** 2 +
                    merged_kf["qnp_kf_sig_ry"].to_numpy(dtype=float) ** 2 +
                    merged_kf["qnp_kf_sig_rz"].to_numpy(dtype=float) ** 2
                )
                conf_gain_rmag = (sig_unw_rmag + eps) / (sig_kf_rmag + eps)

                # KF rejected fraction (proxy for polluted measurement field)
                if has_used_rate:
                    used_rate = merged_kf["kf_used_rate"].to_numpy(dtype=float)
                    rej_frac = 1.0 - used_rate
                else:
                    rej_frac = None

                # ============================================================
                # Fig A: Confidence gain vs time + overlays (rejection + NIS)
                # ============================================================
                plt.figure()
                plt.plot(t, conf_gain_rz, label="Confidence gain (σ_unw_rz / σ_kf_rz)")
                plt.plot(t, conf_gain_rmag, linestyle="--", label="Confidence gain (||σ_r|| unw / KF)")
                plt.axhline(1.0, linestyle="--", label="= 1 (no advantage)")
                plt.xlabel("Time [s]")
                plt.ylabel("Confidence gain (ratio)")
                plt.grid(True)

                # Overlays on a right axis, if available
                ax1 = plt.gca()
                ax2 = ax1.twinx()

                overlay_handles = []
                overlay_labels = []

                if rej_frac is not None:
                    h = ax2.plot(t, rej_frac, linestyle=":", label="KF rejected fraction (1 - used_rate)")
                    overlay_handles += h
                    overlay_labels += ["KF rejected fraction (1 - used_rate)"]

                if has_nis_p95:
                    h = ax2.plot(t, merged_kf["kf_nis_p95_used"], linestyle="-.", label="KF NIS p95 (accepted)")
                    overlay_handles += h
                    overlay_labels += ["KF NIS p95 (accepted)"]
                elif has_nis_med:
                    h = ax2.plot(t, merged_kf["kf_nis_med_used"], linestyle="-.", label="KF NIS median (accepted)")
                    overlay_handles += h
                    overlay_labels += ["KF NIS median (accepted)"]

                ax2.set_ylabel("KF diagnostics")

                # Combine legends cleanly
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="upper left")

                if save:
                    _ensure_dir()
                    plt.savefig(dir_path / Path("16_ConfidenceGain_vs_Time_with_KF_Diagnostics.pdf"))
                    if not show:
                        plt.close()

                # ============================================================
                # Fig B: Confidence gain vs rejection fraction (scatter)
                #   "When more measurements are rejected, KF-weighting helps more"
                # ============================================================
                if rej_frac is not None:
                    x = rej_frac[1:]
                    y = conf_gain_rz[1:]
                    good = np.isfinite(x) & np.isfinite(y)
                    if np.count_nonzero(good) > 3:
                        corr = float(np.corrcoef(x[good], y[good])[0, 1])
                    else:
                        corr = float("nan")

                    plt.figure()
                    plt.scatter(x[good], y[good], s=18, alpha=0.8, label=f"frames (corr={corr:.3f})")
                    plt.xlabel("KF rejected fraction (1 - used_rate)")
                    plt.ylabel("Confidence gain (σ_unw_rz / σ_kf_rz)")
                    plt.grid(True)
                    plt.legend(loc="upper left")

                    if save:
                        _ensure_dir()
                        plt.savefig(dir_path / Path("17_ConfidenceGain_vs_KF_RejectionScatter.pdf"))
                        if not show:
                            plt.close()

                # ============================================================
                # Fig C: Confidence gain vs innovation inconsistency (scatter)
                #   "When accepted innovations are inconsistent, KF-weighting helps more"
                # ============================================================
                if has_nis_p95 or has_nis_med:
                    nis_key = "kf_nis_p95_used" if has_nis_p95 else "kf_nis_med_used"
                    x = merged_kf[nis_key].to_numpy(dtype=float)[1:]
                    y = conf_gain_rz[1:]  #rejecting first one as KF's are unintialized
                    good = np.isfinite(x) & np.isfinite(y)
                    if np.count_nonzero(good) > 3:
                        corr = float(np.corrcoef(x[good], y[good])[0, 1])
                    else:
                        corr = float("nan")

                    plt.figure()
                    plt.scatter(x[good], y[good], s=18, alpha=0.8, label=f"frames (corr={corr:.3f})")
                    plt.xlabel(f"KF {nis_key}")
                    plt.ylabel("Confidence gain (σ_unw_rz / σ_kf_rz)")
                    plt.grid(True)
                    plt.legend(loc="upper left")

                    if save:
                        _ensure_dir()
                        plt.savefig(dir_path / Path("18_ConfidenceGain_vs_KF_InnovationScatter.pdf"))
                        if not show:
                            plt.close()

                # ============================================================
                # Fig D: Heatmap of per-feature instability + gain overlay
                #
                # Two options:
                #   1) Use per-feature NIS (endswith '_kf_nis') — best "false positive / mismatch" proxy
                #   2) Use per-feature sigma_px (endswith '_kf_sigma_px') — best "uncertainty field" view
                #
                # We'll prefer NIS heatmap if available; otherwise fall back to sigma_px.
                # ============================================================

                # Collect per-feature series columns
                nis_cols = [c for c in merged_kf.columns if c.endswith("_kf_nis")]
                sig_cols = [c for c in merged_kf.columns if c.endswith("_kf_sigma_px")]

                # Sort feature columns by feature index if they are named like feat_0_..., feat_1_...
                def _feat_sort_key(name: str):
                    # Expected: "feat_{i}_kf_nis" or "feat_{i}_kf_sigma_px"
                    try:
                        # split on '_' and find first integer
                        parts = name.split("_")
                        for p in parts:
                            if p.isdigit():
                                return int(p)
                        # if pattern is feat_{i}, parts[1] likely int
                        if len(parts) > 1 and parts[1].isdigit():
                            return int(parts[1])
                    except Exception:
                        pass
                    return 10 ** 9

                if len(nis_cols) > 0:
                    cols = sorted(nis_cols, key=_feat_sort_key)
                    data_kind = "NIS"
                    # Mask invalid zeros/negatives as NaN
                    M = merged_kf[cols].to_numpy(dtype=float)
                    M[M <= 0.0] = np.nan
                    # Optional: clip for readability (keep extreme spikes from dominating colormap)
                    # (Use percentile clipping so it adapts to runs)
                    vmin = np.nanpercentile(M, 5)
                    vmax = np.nanpercentile(M, 95)

                elif len(sig_cols) > 0:
                    cols = sorted(sig_cols, key=_feat_sort_key)
                    data_kind = "sigma_px"
                    M = merged_kf[cols].to_numpy(dtype=float)
                    M[~np.isfinite(M)] = np.nan
                    vmin = np.nanpercentile(M, 5)
                    vmax = np.nanpercentile(M, 95)

                else:
                    cols = []
                    M = None
                    data_kind = None

                if M is not None and len(cols) > 0:
                    plt.figure(figsize=(10, 6))
                    ax = plt.gca()

                    # Heatmap: time along x, feature index along y
                    # We use imshow with extent so x-axis matches time
                    im = ax.imshow(
                        M.T,
                        aspect="auto",
                        origin="lower",
                        extent=[float(t.iloc[0]), float(t.iloc[-1]), 0, M.shape[1]],
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Feature index")
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label(data_kind)

                    # Overlay confidence gain on a second axis (same x)
                    ax2 = ax.twinx()
                    ax2.plot(t, conf_gain_rz,
                             linewidth=1.0, color='black')
                    ax2.plot(t, conf_gain_rz, label="Confidence gain (σ_unw_rz / σ_kf_rz)",
                             linewidth=0.5, color='white')
                    ax2.axhline(1.0, linestyle="--")
                    ax2.set_ylabel("Confidence gain")

                    # Optional: mark top-K gain spikes with vertical lines
                    # Helps visually align "pollution field" with "brittleness"
                    cg = np.asarray(conf_gain_rz, dtype=float)
                    good = np.isfinite(cg)
                    if np.count_nonzero(good) > 10:
                        # pick top 5 spikes
                        # Optional: mark top-K confidence-gain spikes with vertical lines
                        # Helps visually align "pollution field" with "brittleness"
                        cg = np.asarray(conf_gain_rz, dtype=float)
                        t_arr = np.asarray(t, dtype=float)
                        good = np.isfinite(cg) & np.isfinite(t_arr)

                        if np.count_nonzero(good) > 10:
                            cg_good = cg[good]
                            t_good = t_arr[good]

                            # Pick up to 6 largest gain spikes from this run, regardless of run length
                            k = min(6, cg_good.size)
                            spike_local_idxs = np.argpartition(cg_good, -k)[-k:]
                            spike_local_idxs = spike_local_idxs[np.argsort(cg_good[spike_local_idxs])[::-1]]

                            # Draw in time order so the overlay looks neat
                            spike_times = np.sort(t_good[spike_local_idxs])

                            for xv in spike_times:
                                ax.axvline(float(xv), linestyle=":", linewidth=1, color="red")

                    # Legends
                    h1, l1 = ax2.get_legend_handles_labels()
                    ax2.legend(h1, l1, loc="upper right")

                    plt.title(f"Per-feature {data_kind} heatmap with confidence gain overlay")

                    if save:
                        _ensure_dir()
                        plt.savefig(dir_path / Path(f"19_Heatmap_{data_kind}_with_ConfidenceGainOverlay.pdf"))
                        if not show:
                            plt.close()

            # ============================================================
            # NEW FIGURES: Region-summary stats (NIS + Gain + Rejection)
            # ============================================================

            # ---- configure windows (seconds) ----
            roi_windows = [
                ("Early gain (1.5–3.0s)", 1.5, 3.0),
                ("Moderate Spike (5–7.5s)", 5.0, 7.5),
                ("Nominal (9–11s)", 9.0, 11.0),
                ("Nominal (11–13s)", 11.0, 13.0),
                ("Nominal (17.5–20s)", 17.5, 20.0),
                ("Late gain (21–22s)", 21.0, 22.0),
            ]

            # ---- per-feature KF NIS + used masks ----
            nis_cols = [c for c in merged_kf.columns if c.endswith("_kf_nis")]
            used_cols = [c.replace("_kf_nis", "_kf_used") for c in nis_cols if
                         c.replace("_kf_nis", "_kf_used") in merged_kf.columns]

            # ---- confidence gain (rotation) ----
            eps = 1e-12
            have_gain = ("qnp_sig_rz" in merged_kf.columns) and ("qnp_kf_sig_rz" in merged_kf.columns)
            if have_gain:
                sig_unw = merged_kf["qnp_sig_rz"].to_numpy(dtype=float)
                sig_kf = merged_kf["qnp_kf_sig_rz"].to_numpy(dtype=float)
                conf_gain = (sig_unw + eps) / (sig_kf + eps)
            else:
                conf_gain = None

            # ---- rejection fraction ----
            rej_frac = None
            if "kf_used_rate" in merged_kf.columns:
                rej_frac = 1.0 - merged_kf["kf_used_rate"].to_numpy(dtype=float)

            def _collect_used_nis(t0, t1):
                m = (t >= t0) & (t <= t1)
                if len(nis_cols) == 0 or len(used_cols) == 0:
                    return np.array([], dtype=float), int(np.count_nonzero(m))
                nisM = merged_kf.loc[m, nis_cols].to_numpy(dtype=float)
                useM = merged_kf.loc[m, used_cols].to_numpy(dtype=float)
                x = nisM[(useM > 0.5) & np.isfinite(nisM) & (nisM >= 0.0)]
                return x, int(np.count_nonzero(m))

            def _roi_stat(arr):
                # robust summary (avoid being dominated by rare spikes)
                if arr.size == 0:
                    return dict(med=np.nan, p90=np.nan, p95=np.nan)
                med, p90, p95 = np.percentile(arr, [50, 90, 95])
                return dict(med=float(med), p90=float(p90), p95=float(p95))

            def _roi_gain_stat(t0, t1):
                if conf_gain is None:
                    return dict(med=np.nan, p90=np.nan)
                m = (t >= t0) & (t <= t1)
                g = conf_gain[m]
                g = g[np.isfinite(g)]
                if g.size == 0:
                    return dict(med=np.nan, p90=np.nan)
                med, p90 = np.percentile(g, [50, 90])
                return dict(med=float(med), p90=float(p90))

            def _roi_rej_stat(t0, t1):
                if rej_frac is None:
                    return np.nan
                m = (t >= t0) & (t <= t1)
                r = rej_frac[m]
                r = r[np.isfinite(r)]
                return float(np.mean(r)) if r.size else np.nan

            # ---- collect region datasets ----
            labels = [w[0] for w in roi_windows]
            nis_used_sets = []
            nis_stats = []
            gain_stats = []
            rej_stats = []
            frames_counts = []

            for name, a, b in roi_windows:
                x, nframes = _collect_used_nis(a, b)
                nis_used_sets.append(x)
                nis_stats.append(_roi_stat(x))
                gain_stats.append(_roi_gain_stat(a, b))
                rej_stats.append(_roi_rej_stat(a, b))
                frames_counts.append(nframes)

            # ============================================================
            # FIG 1: Boxplot of used-feature NIS by window
            # ============================================================
            plt.figure(figsize=(8, 5))
            plt.boxplot(nis_used_sets, labels=labels, showfliers=False)
            plt.ylabel("Per-feature NIS (used measurements only)")
            plt.xlabel("Time window")
            plt.grid(True, axis="y")
            plt.title("Per-feature NIS distribution by time window (accepted features)")
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            if save:
                plt.savefig(dir_path / Path("20_ROI_NIS_Boxplot.pdf"))
                if not show:
                    plt.close()

            # ============================================================
            # FIG 2: Robust NIS quantiles (Median / P90 / P95) by window
            # ============================================================
            xpos = np.arange(len(labels))
            width = 0.22
            meds = [s["med"] for s in nis_stats]
            p90s = [s["p90"] for s in nis_stats]
            p95s = [s["p95"] for s in nis_stats]

            plt.figure(figsize=(8, 5))
            plt.bar(xpos - width, meds, width, label="Median")
            plt.bar(xpos, p90s, width, label="P90")
            plt.bar(xpos + width, p95s, width, label="P95")
            plt.xticks(xpos, labels, rotation=15, ha="right")
            plt.ylabel("Per-feature NIS (used measurements only)")
            plt.grid(True, axis="y")
            plt.title("Robust NIS quantiles by time window (accepted features)")
            plt.legend(loc="upper left")
            plt.tight_layout()
            if save:
                plt.savefig(dir_path / Path("21_ROI_NIS_Quantiles.pdf"))
                if not show:
                    plt.close()

            # ============================================================
            # FIG 3: Confidence gain by window + mean rejection fraction overlay
            # ============================================================
            gain_med = [g["med"] for g in gain_stats]
            gain_p90 = [g["p90"] for g in gain_stats]

            plt.figure(figsize=(8, 5))
            ax1 = plt.gca()
            ax1.bar(xpos, gain_med, label="Gain median (σ_unw/σ_kf)")
            ax1.plot(xpos, gain_p90, linestyle="--", marker="o", label="Gain P90")
            ax1.set_xticks(xpos)
            ax1.set_xticklabels(labels, rotation=15, ha="right")
            ax1.set_ylabel("Confidence gain (σ_unweighted / σ_KF-weighted)")
            ax1.grid(True, axis="y")

            ax2 = ax1.twinx()
            ax2.plot(xpos, rej_stats, linestyle=":", marker="s", label="Mean rejected fraction")
            ax2.set_ylabel("Rejected fraction (1 - used_rate)")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left")

            plt.title("Region summary: confidence gain vs measurement rejection")
            plt.tight_layout()
            if save:
                plt.savefig(dir_path / Path("22_ROI_Gain_vs_Rejection.pdf"))
                if not show:
                    plt.close()

            # sigma_pxs_cols = list(filter(lambda x: x.endswith("sigma_px"), merged_kf.columns))
            # sigma_pys_cols = list(filter(lambda x: x.endswith("sigma_py"), merged_kf.columns))
            # sigma_pxs = [merged_kf[px_col] for px_col in sigma_pxs_cols]
            #
            # plt.figure()
            # [plt.plot(t, sigma_px) for sigma_px in sigma_pxs]
            # plt.yscale('log')
            # plt.ylim([0.0, 10.0])

        if show:
            plt.show()

    @staticmethod
    def close_plot():
        plt.close('all')


if __name__ == "__main__":
    pass