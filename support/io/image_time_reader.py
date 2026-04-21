import csv, os
import datetime

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
                    image_time = datetime.datetime.strptime(rowList[0][0:-7], '%Y.%b.%d_%H.%M.%S.%f')
                    if self.startTimeUTC is None:
                        self.startTimeUTC = image_time
                    id = int(rowList[1])
                    imgName = rowList[2]
                    self.idsTimes.append([imgName, (image_time-self.startTimeUTC).total_seconds()])
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

                    image_time = datetime.datetime.strptime(rowList[0][0:-5], '%Y.%b.%d_%H.%M.%S.%f')
                    if self.startTimeUTC is None:
                        self.startTimeUTC = image_time
                    id = int(rowList[1])
                    self.idsTimes.append(
                        [id, (image_time - self.startTimeUTC).total_seconds(), float(rowList[2]), float(rowList[3]),
                         float(rowList[4]), float(rowList[5]), float(rowList[6]), float(rowList[7])])
                    self.endTimeUTC = image_time

    @property
    def numImages(self):
        return len(self.idsTimes)

if __name__ == '__main__':
    reader = ImageTimeReader('TargetImages/___1970.Jan.01_00.14.32.462084352.UTC.log')
    for nameTime in reader.idsTimes:
        print(nameTime)