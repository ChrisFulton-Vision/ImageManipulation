import pandas as pd
import matplotlib.pyplot as plt
from support.io.Logging import LOG
from pathlib import Path
import numpy as np

class Plotter:
    @staticmethod
    def plot(conf: float, img_dir: Path):

        # Input CSVs
        try:
            pnp = pd.read_csv(img_dir / f"3_pnp_conf{conf:.2f}.csv")
        except FileNotFoundError:
            LOG.warning(fr'Tried to find ' + str(img_dir / f"3_pnp_conf{conf:.2f}.csv" ))
            return
        try:
            qnp = pd.read_csv(img_dir / f"4_qnp_conf{conf:.2f}.csv")
        except FileNotFoundError:
            LOG.warning(fr'Tried to find ' + str(img_dir / f"4_qnp_conf{conf:.2f}.csv" ))
            return

        # Optional KF CSV (for NIS diagnostics)
        kf = None
        try:
            # Adjust name if your pipeline uses a different filename convention
            kf = pd.read_csv(img_dir / f"2_kalman_conf{conf:.2f}.csv")
        except FileNotFoundError:
            # Not required; just skip NIS plots if missing
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
            # Merge KF diagnostics onto the same timeline
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
            # plt.show()

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
            # plt.show()

        # -------------------------------
        # Position differences vs PnP
        #   - QnP - PnP
        #   - QnP-KF - PnP (if available)
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
        # plt.show()

        # -------------------------------
        # Quaternion component differences vs PnP
        #   - QnP - PnP
        #   - QnP-KF - PnP (if available)
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
        # plt.show()

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
                # Reference lines for chi^2_2
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

            # Optional: histogram of per-feature NIS for accepted measurements
            # Requires columns like feat_<id>_kf_used and feat_<id>_kf_nis
            nis_cols = [c for c in merged_kf.columns if c.endswith("_kf_nis")]
            used_cols = [c for c in merged_kf.columns if c.endswith("_kf_used")]

            if len(nis_cols) > 0 and len(used_cols) > 0:
                # Build a flat vector of accepted NIS values
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


        plt.show()

    @staticmethod
    def close_plot():
        plt.close('all')


if __name__ == "__main__":
    Plotter().plot()