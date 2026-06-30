import csv, os
import datetime


def _parse_log_timestamp_utc_seconds(raw_ts: str, trim_chars: int) -> float:
    dt = datetime.datetime.strptime(raw_ts[:-trim_chars], '%Y.%b.%d_%H.%M.%S.%f')
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()


class ImageTimeReader:
    def __init__(self, filename:str = None):
        self.filename = filename
        self.idsTimes = []
        self.startTimeUTC = None
        self.endTimeUTC = None

        if self.filename is not None:
            self.loadLog([self.filename])

    def loadLog(self, filename_list: list[str] = None):
        if len(filename_list) < 1:
            return False

        filename = filename_list[0]
        if not os.path.exists(filename) or not filename.endswith('.log'):
            return False
        if filename is None and self.filename is None:
            return False

        if filename is not None:
            self.filename = filename

        self.idsTimes = []
        with open(self.filename, newline='') as file:
            reader = csv.reader(file)
            self.startTimeUTC = None
            for row in reader:
                rowList = row[0].split(sep=' ')
                if rowList[0] != '#':
                    image_time = _parse_log_timestamp_utc_seconds(rowList[0], 7)
                    if self.startTimeUTC is None:
                        self.startTimeUTC = image_time
                    id = int(rowList[1])
                    imgName = rowList[2]
                    self.idsTimes.append([imgName, image_time])
                    self.endTimeUTC = image_time
        return True

    @property
    def numImages(self):
        return len(self.idsTimes)

class CarrierTimeReader(ImageTimeReader):
    def loadLog(self, filename:str = None):
        if not os.path.exists(filename) or not filename.endswith('.txt'):
            return
        if filename is None and self.filename is None:
            return

        if filename is not None:
            self.filename = filename

        self.idsTimes = []
        with open(self.filename) as file:
            reader = csv.reader(file)
            startTime = None
            for rowList in reader:
                if rowList[0] != '#':

                    image_time = _parse_log_timestamp_utc_seconds(rowList[0], 5)
                    if self.startTimeUTC is None:
                        self.startTimeUTC = image_time
                    id = int(rowList[1])
                    self.idsTimes.append(
                        [id, image_time, float(rowList[2]), float(rowList[3]),
                         float(rowList[4]), float(rowList[5]), float(rowList[6]), float(rowList[7])])
                    self.endTimeUTC = image_time

    @property
    def numImages(self):
        return len(self.idsTimes)

if __name__ == '__main__':
    reader = ImageTimeReader('TargetImages/___1970.Jan.01_00.14.32.462084352.UTC.log')
    for nameTime in reader.idsTimes:
        print(nameTime)
