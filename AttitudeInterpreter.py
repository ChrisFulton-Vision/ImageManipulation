import pandas as pd
import numpy as np
from os.path import join

class AttitudeReader:
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
        if not {'timestamp', 'Roll', 'DesRoll', 'Pitch', 'DesPitch'}.issubset(self.roll_dict.columns):
            return False

        if not {'timestamp', 'C1', 'C3', 'C8'}.issubset(self.cmd_dict.columns):
            return False

        self.offset = offset_dict['offset'][0]

        # Sort by Time for proper interpolation
        self.roll_dict = self.roll_dict.sort_values('timestamp').reset_index(drop=True)
        self.cmd_dict = self.cmd_dict.sort_values('timestamp').reset_index(drop=True)
        self.ready = True
        return True

    def get_roll_at(self, query_time):

        query_time += self.offset

        if not self.ready or query_time < self.roll_dict['timestamp'].iloc[0] or query_time > \
                self.roll_dict['timestamp'].iloc[-1]:
            return 180.0, 0.0, 180.0, 0.0, 0.0, False

        # if not self.ready or query_time < self.roll_dict['timestamp'].iloc[0]:
        #     return 180.0, 0.0, 180.0, 0.0, False

        # print(f"Img Time: {query_time}")

        # print(f"GPS Time: {query_time}\n")

        # Handle out-of-bounds
        # if query_time < self.roll_dict['timestamp'].iloc[0]:
        #     print('Beginning of file...\n')
        #     return self.roll_dict['Roll'][0], self.roll_dict['DesRoll'][0], self.roll_dict['Pitch'][0], \
        #     self.roll_dict['DesPitch'][0], self.cmd_dict['C8'][0]

        # if query_time > self.roll_dict['timestamp'].iloc[-1]:
        #     print('End of file...\n')
        #     return self.roll_dict['Roll'].iloc[-1], self.roll_dict['DesRoll'].iloc[-1], self.roll_dict['Pitch'].iloc[-1], \
        #     self.roll_dict['DesPitch'].iloc[-1], self.cmd_dict['C8'].iloc[-1]



        # Use numpy to interpolate
        interpolated_roll = np.interp(
            query_time,
            self.roll_dict['timestamp'],
            self.roll_dict['Roll']
        )

        interpolated_cmd_roll = np.interp(
            query_time,
            self.roll_dict['timestamp'],
            self.roll_dict['DesRoll']
        )

        interpolated_pitch = np.interp(
            query_time,
            self.roll_dict['timestamp'],
            self.roll_dict['Pitch']
        )

        interpolated_cmd_pitch = np.interp(
            query_time,
            self.roll_dict['timestamp'],
            self.roll_dict['DesPitch']
        )

        interpolated_cmd_throttle = self.throttle_pwm_to_perc(
            np.interp(
            query_time,
            self.cmd_dict['timestamp'],
            self.cmd_dict['C3']
        ))

        interpolated_mode = np.interp(
            query_time,
            self.cmd_dict['timestamp'],
            self.cmd_dict['C8']
        )
        mode = 950 < interpolated_mode < 1400

        return interpolated_roll, interpolated_cmd_roll, interpolated_pitch, interpolated_cmd_pitch, interpolated_cmd_throttle, mode

    def throttle_pwm_to_perc(self, throttle_pwm: np.array) -> np.array:
        return (throttle_pwm - 1300.0) / (1880.0 - 1330.0) * 100.0
# file = filedialog.askopenfilename(initialdir='./')
# print(file)
# file = 'C:/Users/fulto/Desktop/UAS Flight Test/25_Spring/LOGS/00000064/XKF1.csv'
# RR = AttitudeReader(file)
# for i in range(1000):
#     print(RR.get_roll_at(1748534621.7987978 + i))
