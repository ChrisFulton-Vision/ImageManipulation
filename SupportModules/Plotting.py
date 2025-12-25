import pandas as pd
import matplotlib.pyplot as plt
from SupportModules.Logging import LOG
from pathlib import Path

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

        # Merge on image name + time
        merged = pd.merge(
            pnp,
            qnp,
            on=["image_name", "image_time"],
            how="inner",
            suffixes=("_pnp", "_qnp"),
        )

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

        plt.show()

    @staticmethod
    def close_plot():
        plt.close('all')


if __name__ == "__main__":
    Plotter().plot()