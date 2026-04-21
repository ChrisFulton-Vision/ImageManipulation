import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


def parse_sensor_datum_mat4_bundle(file: str | os.PathLike[str]) -> pd.DataFrame:
    """Parses a sensor datum mat4 bundle from the given data.

    Args:
        file (str): The path to the sensor datum mat4 bundle file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the parsed sensor datum mat4 bundle data.
    """
    with Path(file).open() as f:
        raw_file = f.readlines()

    # Prepare the data for parsing
    file_elements: list[list[str]] = []  # List of each line split into elements

    num_prefix_cols = 2  # utc_time, num_msgs
    contains_frame_id = False
    if raw_file[0].strip().startswith("#"):
        # check if frame_id is in the header
        for element in raw_file[0].strip().split():
            if element.lower() == "frame_id":
                num_prefix_cols = 3  # utc_time, frame_id, num_msgs
                contains_frame_id = True
                break

    for raw_line in raw_file:
        stripped_line = raw_line.strip()
        # Remove comments and empty lines
        if stripped_line.startswith("#") or stripped_line == "":
            continue
        # Split the line into elements
        file_elements.append([element.strip() for element in stripped_line.split()])

    # Initialize output data with all expected columns
    out_data: dict[str, list] = {"utc_time": [], "frame_id": [], "object_id": [], **{f"m{j}": [] for j in range(16)}}
    for line_num, line in enumerate(file_elements):
        try:
            utc_time = pd.to_datetime(line[0], format="%Y.%b.%d_%H.%M.%S.%f.UTC", errors="coerce")
            if utc_time is pd.NaT:
                raise ValueError(f"Invalid UTC time format: {line[0]}")

            frame_id = int(line[1]) if contains_frame_id else line_num
            num_msgs = int(line[2]) if contains_frame_id else int(line[1])
            num_expected_elements = num_prefix_cols + num_msgs * 17
            if len(line) != num_expected_elements:
                raise ValueError(f"Expected {num_expected_elements} elements, but got {len(line)}: {line}")
            mat4_start = num_prefix_cols
            for _ in range(num_msgs):
                object_id = line[mat4_start]
                mat4_values = [float(v) for v in line[mat4_start + 1 : mat4_start + 17]]
                out_data.setdefault("utc_time", []).append(utc_time)
                out_data.setdefault("frame_id", []).append(frame_id)
                out_data.setdefault("object_id", []).append(object_id)
                for j in range(16):
                    out_data.setdefault(f"m{j}", []).append(mat4_values[j])
                mat4_start += 17
        except (ValueError, IndexError) as e:
            err_msg = f"Error parsing mat4 bundle sensor datum line. Expected [str, int, int, [int, float*16]*NumMsgs]: {line}"
            raise type(e)(err_msg) from e
    mat4_df = pd.DataFrame(out_data)
    if not contains_frame_id:
        mat4_df["frame_id"] = range(1, len(mat4_df) + 1)
    return mat4_df


def mat4_cols_to_numpy(df: pd.DataFrame) -> npt.NDArray[np.float64]:
    """
    Converts mat4 columns in the DataFrame to numpy arrays.

    Args:
        df (pd.DataFrame): The DataFrame containing mat4 columns.
    
    Returns:
        npt.NDArray[np.float64]: The DataFrame with mat4 columns converted to numpy arrays.
    """
    return df[[f"m{i}" for i in range(16)]].to_numpy(dtype=np.float64).reshape(-1, 4, 4, order="F")


