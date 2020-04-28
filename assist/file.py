from os.path import split, splitext, join
from config import config


# singleton训练图片保存
def createSaveFileNameFunc(originFilePath, targetPath=None):
    filePath, completeFileName = split(originFilePath)
    fileName, fileSuffix = splitext(completeFileName)
    if targetPath is None:
        targetPath = filePath

    defaultName = join(targetPath, completeFileName)

    def newSaveFileNameFunc(iteration):
        if config.needIterationTag:
            return join(targetPath, f'{fileName}.{iteration}{fileSuffix}')
        return defaultName

    return newSaveFileNameFunc


# singleton训练半截的图片
def trainedPicture(originFilePath):
    _, completeFileName = split(originFilePath)
    return join(config.create_dir, completeFileName)
