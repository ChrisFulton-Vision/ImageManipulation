from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

COLS = [
    "id","trial","is_tanker","t0","t1","t2","t3","distance",
    "rec_cur_pos","rec_est_pos","rec_err_pos","rec_pos_cov",
    "hull_cur_pos","hull_est_pos","hull_err_pos",
    "tip_cur_pos","tip_est_pos","tip_err_pos",
]

POSE_COLS = [
    "rec_cur_pos","rec_est_pos","rec_err_pos",
    "hull_cur_pos","hull_est_pos","hull_err_pos",
    "tip_cur_pos","tip_est_pos","tip_err_pos",
]

def parse_se3_colmajor(cell) -> np.ndarray | None:
    # pandas NA-safe
    if cell is None or pd.isna(cell):
        return None
    s = str(cell).strip()
    if not s:
        return None

    parts = s.split()
    if len(parts) != 16:
        raise ValueError(f"Expected 16 floats for SE3, got {len(parts)}: {s[:80]}...")

    vals = np.array([float(p) for p in parts], dtype=np.float64)
    return vals.reshape((4, 4), order="F")  # column-major

def load_pnp_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    # Force pose-ish columns to be strings so pandas doesn't try to infer weird types
    dtype = {c: "string" for c in POSE_COLS + ["rec_pos_cov"]}

    df = pd.read_csv(
        path,
        engine="python",
        header=0,
        names=COLS,          # enforce expected schema
        index_col=False,     # don't auto-use first column as index
        dtype=dtype,
    )

    # Scalars
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")
    df["trial"] = pd.to_numeric(df["trial"], errors="raise").astype("int64")
    df["is_tanker"] = pd.to_numeric(df["is_tanker"], errors="raise").astype("int64").astype(bool)

    for c in ["t0","t1","t2","t3","distance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # blank -> NaN is fine

    # Parse SE3 matrices into new columns with suffix _T
    for c in POSE_COLS:
        df[c + "_T"] = df[c].map(parse_se3_colmajor)

    return df

# handy helpers
def se3_R(T: np.ndarray) -> np.ndarray:
    return T[:3, :3]

def se3_t(T: np.ndarray) -> np.ndarray:
    return T[:3, 3]

def parse_data(path: str = 'fixed_pnp.csv'):
    df = load_pnp_csv(path)

    df["t_mid"] = 0.5 * (df["t1"] + df["t2"])
    df = df.sort_values(["t_mid", "trial", "is_tanker", "id"], kind="mergesort").reset_index(drop=True)

    time = df["t_mid"]
    is_tanker = df['is_tanker']
    est_T_rec = df["rec_est_pos_T"]  # 4x4 ndarray or None
    tru_T_rec = df["rec_cur_pos_T"]  # 4x4 ndarray or None
    err_T_rec = df["rec_err_pos_T"]  # 4x4 ndarray or None

    est_T_hull = df["hull_est_pos_T"]
    est_T_tip = df["tip_est_pos_T"]

    return list(time / 1000.0), is_tanker, list(est_T_rec), list(tru_T_rec), list(err_T_rec)

parse_data()