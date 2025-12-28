import pandas as pd
import matplotlib.pyplot as plt
from support.io.my_logging import LOG
from pathlib import Path
import numpy as np

class Plotter:
    @staticmethod
    def plot(conf: float, img_dir: Path):

        plt.rcParams['figure.max_open_warning'] = 30

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
            plt.title(f"{comp.upper()} Position vs Time")
            plt.legend()
            plt.grid(True)

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
            plt.ylabel(comp.upper())
            plt.title(f"Quaternion component {comp.upper()} vs Time")
            plt.legend()
            plt.grid(True)

        # -------------------------------
        # Position differences vs PnP
        # -------------------------------
        plt.figure()
        for comp in ["x", "y", "z"]:
            plt.plot(
                t,
                merged[f"qnp_{comp}"] - merged[f"pnp_{comp}"],
                label=f"{comp.upper()} (QnP - PnP)",
            )
        if has_kf_pos:
            for comp in ["x", "y", "z"]:
                plt.plot(
                    t,
                    merged[f"qnp_kf_{comp}"] - merged[f"pnp_{comp}"],
                    linestyle="--",
                    label=f"{comp.upper()} (QnP-KF - PnP)",
                )

        plt.xlabel("Time [s]")
        plt.ylabel("Position difference")
        plt.title("Position Difference vs Time (relative to PnP)")
        plt.legend()
        plt.grid(True)

        # -------------------------------
        # Quaternion component differences vs PnP
        # -------------------------------
        plt.figure()
        for comp in ["qw", "qx", "qy", "qz"]:
            plt.plot(
                t,
                merged[f"qnp_{comp}"] - merged[f"pnp_{comp}"],
                label=f"{comp.upper()} (QnP - PnP)",
            )
        if has_kf_quat:
            for comp in ["qw", "qx", "qy", "qz"]:
                plt.plot(
                    t,
                    merged[f"qnp_kf_{comp}"] - merged[f"pnp_{comp}"],
                    linestyle="--",
                    label=f"{comp.upper()} (QnP-KF - PnP)",
                )

        plt.xlabel("Time [s]")
        plt.ylabel("Quaternion component difference")
        plt.title("Quaternion Component Difference vs Time (relative to PnP)")
        plt.legend()
        plt.grid(True)

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
            plt.title("Position Difference: QnP-KF - QnP")
            plt.legend()
            plt.grid(True)

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
            plt.title("Quaternion Component Difference: QnP-KF - QnP")
            plt.legend()
            plt.grid(True)

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
            plt.title("SolveQnP Residual Scale (s2) vs Time")
            plt.grid(True)
            plt.legend()

        # used_n over time (helps interpret dof jumps / gating)
        if ("qnp_used_n" in merged.columns) or ("qnp_kf_used_n" in merged.columns):
            plt.figure()
            if "qnp_used_n" in merged.columns:
                plt.plot(t, merged["qnp_used_n"], label="QnP used_n")
            if "qnp_kf_used_n" in merged.columns:
                plt.plot(t, merged["qnp_kf_used_n"], label="QnP-KF used_n")
            plt.xlabel("Time [s]")
            plt.ylabel("N points used")
            plt.title("SolveQnP Used Feature Count vs Time")
            plt.grid(True)
            plt.legend()

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
            plt.title("SolveQnP Rotation Uncertainty (Rodrigues tangent) vs Time")
            plt.grid(True)
            plt.legend()

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
            plt.title("SolveQnP Translation Uncertainty vs Time")
            plt.grid(True)
            plt.legend()

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
            plt.title("SolveQnP Combined Uncertainty Magnitudes vs Time")
            plt.grid(True)
            plt.legend()

            # Ratio plots (KF-weighted / unweighted) — nice for “improvement factor” story
            if have_qnp_sig and have_qnp_kf_sig:
                eps = 1e-12
                plt.figure()
                plt.plot(t, (sig_t_mag_kf + eps) / (sig_t_mag + eps), label="||sig_t||_KF / ||sig_t||")
                plt.plot(t, (sig_r_mag_kf + eps) / (sig_r_mag + eps), label="||sig_r||_KF / ||sig_r||")
                plt.axhline(1.0, linestyle="--", label="=1")
                plt.xlabel("Time [s]")
                plt.ylabel("Ratio")
                plt.title("Uncertainty Ratio: KF-weighted vs Unweighted")
                plt.grid(True)
                plt.legend()

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
                plt.title("KF Accepted Measurement Rate vs Time")
                plt.grid(True)
                plt.legend()

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
                plt.title("KF Normalized Innovation Squared (Accepted) vs Time")
                plt.grid(True)
                plt.legend()

            if has_sig_px or has_sig_py:
                plt.figure()
                if has_sig_px:
                    plt.plot(t, merged_kf["kf_sigma_meas_px"], label="sigma_meas_x [px]")
                if has_sig_py:
                    plt.plot(t, merged_kf["kf_sigma_meas_py"], label="sigma_meas_y [px]")
                plt.xlabel("Time [s]")
                plt.ylabel("Sigma [px]")
                plt.title("Estimated Measurement Noise Sigma vs Time")
                plt.grid(True)
                plt.legend()

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
                    plt.title("Histogram of Accepted NIS (All Features)")
                    plt.grid(True)
                    plt.legend()

            # sigma_pxs_cols = list(filter(lambda x: x.endswith("sigma_px"), merged_kf.columns))
            # sigma_pys_cols = list(filter(lambda x: x.endswith("sigma_py"), merged_kf.columns))
            # sigma_pxs = [merged_kf[px_col] for px_col in sigma_pxs_cols]
            #
            # plt.figure()
            # [plt.plot(t, sigma_px) for sigma_px in sigma_pxs]
            # plt.yscale('log')
            # plt.ylim([0.0, 10.0])


        plt.show()

    @staticmethod
    def close_plot():
        plt.close('all')


if __name__ == "__main__":
    pass