import numpy as np
from os.path import exists
from PIL import Image
from config import config
from matplotlib import pyplot as plt
import uuid

trans = config.singletonTransform()


def readImage(path):
    return Image.open(open(path, 'rb')).convert('RGB')


def image2tensor(image):
    return trans(image).unsqueeze(0)


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * np.array(config.cnn_normalization_std).reshape((1, 3, 1, 1)) + \
            np.array(config.cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]


def saveImage(image, name=None, fromNumpy=True):
    if fromNumpy:
        image = Image.fromarray(image)
    if name is None:
        name = uuid.uuid4()
    image.save(name)
    return name


def saveTensorImage(image, name=None):
    image = tensor2image(image)
    return saveImage(image, name, True)


def imageTensor(path):
    if not exists(path):
        return None
    return image2tensor(readImage(path))


def createShowImageFunc(isTensor=False):
    plt.ion()

    def imageShowFunc(image, iteration=None):
        if isTensor:
            image = tensor2image(image)
        plt.clf()
        plt.imshow(image)
        if iteration is not None:
            plt.title("iteration:{:4.0f}".format(iteration))
        plt.pause(config.imageKeepTime)
        plt.ioff()

    return imageShowFunc
