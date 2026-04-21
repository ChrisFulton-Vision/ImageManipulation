import csv

class MetaYoloReader:
    def __init__(self, filename:str = None):
        self.filename = filename
        self.idsNamesLocs = []
        self.imageSize = 864
        self.numClasses = 95

        if self.filename is not None:
            self.loadMetaYolo()

    def loadMetaYolo(self):
        self.idsNamesLocs = []
        with open(self.filename, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 0 and row[0] == 'yolo_image_size':
                    self.imageSize = int(row[1])
                elif len(row) > 0 and row[0] == 'yolo_num_classes':
                    self.numClasses = int(row[1])
                elif len(row) > 0 and row[0].isdigit():
                    self.idsNamesLocs.append([int(row[0]), row[1], float(row[3]),float(row[4]),float(row[5])])

if __name__ == '__main__':
    reader = MetaYoloReader('../../YOLOModels/GIII_01172025_10_100M_MoreFeatures/meta_yolo.csv')
    for stuff in reader.idsNamesLocs:
        print(stuff)