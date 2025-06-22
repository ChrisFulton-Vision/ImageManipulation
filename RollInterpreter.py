import pandas as pd
import numpy as np
from os.path import join

class RollReader:
    def __init__(self, csv_folder_path: str = None):
        self.roll_dict = None
        self.cmd_dict = None
        self.offset = 0.0
        self.ready = False

        if csv_folder_path is not None:
            self.read_files(csv_folder_path)

    def read_files(self, csv_folder_path: str):
        # Load CSV and extract Time and Roll columns
        try:
            self.roll_dict = pd.read_csv(join(csv_folder_path, 'ATT.csv'))
            self.cmd_dict = pd.read_csv(join(csv_folder_path, 'RCOU.csv'))
            offset_dict = pd.read_csv(join(csv_folder_path, '__TIME_OFFSET.csv'))
        except FileNotFoundError:
            return False

        # Check if required columns are present
        if not {'timestamp', 'Roll', 'DesRoll'}.issubset(self.roll_dict.columns):
            return False

        if not {'timestamp', 'C1', 'C8'}.issubset(self.cmd_dict.columns):
            return False

        self.offset = offset_dict['offset'][0]

        # Sort by Time for proper interpolation
        self.roll_dict = self.roll_dict.sort_values('timestamp').reset_index(drop=True)
        self.cmd_dict = self.cmd_dict.sort_values('timestamp').reset_index(drop=True)
        self.ready = True
        return True

    def get_roll_at(self, query_time):
        if not self.ready:
            return 0.0, 0.0, False

        query_time += self.offset

        # Handle out-of-bounds
        if query_time < self.roll_dict['timestamp'].iloc[0] or query_time > self.roll_dict['timestamp'].iloc[-1]:
            raise ValueError("query_time is outside the range of available times")


        # Use numpy to interpolate
        interpolated_roll = np.interp(
            query_time,
            self.roll_dict['timestamp'],
            self.roll_dict['Roll']
        )
        # Use numpy to interpolate
        interpolated_cmd_roll = np.interp(
            query_time,
            self.roll_dict['timestamp'],
            self.roll_dict['DesRoll']
        )

        interpolated_mode = np.interp(
            query_time,
            self.cmd_dict['timestamp'],
            self.cmd_dict['C8']
        )
        mode = 950 < interpolated_mode < 1400

        return interpolated_roll, interpolated_cmd_roll, mode

# file = filedialog.askopenfilename(initialdir='./')
# print(file)
# file = 'C:/Users/fulto/Desktop/UAS Flight Test/25_Spring/LOGS/00000064/XKF1.csv'
# RR = RollReader(file)
# for i in range(1000):
#     print(RR.get_roll_at(1748534621.7987978 + i))
