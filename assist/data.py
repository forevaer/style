from os import listdir
from os.path import exists, isdir, join
from assist.image import imageTensor


class DataGenerator:
    """
    扫描指定文件夹，作为训练数据集合
    """
    def __init__(self, imageDir):
        self.imageDir = imageDir
        self.items = []
        self.load_data()

    def load_data(self):
        if not (exists(self.imageDir) or isdir(self.imageDir)):
            raise Exception(f'illegal imageDir : {self.imageDir}')
        for file in listdir(self.imageDir):
            self.items.append(imageTensor(join(self.imageDir, file)))

    def __iter__(self):
        for item in self.items:
            yield item

