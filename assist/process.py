import torch
from os.path import exists
from config.config import saveModelPath
from torchvision.models import vgg16
from network.network import StyleTransformNet, ExtractNet


def getExtract(_device):
    vgg = vgg16(pretrained=True)
    return ExtractNet(vgg.features[:23]).to(_device).eval()


def getModel(_device):
    model = StyleTransformNet().to(_device)
    if exists(saveModelPath):
        model.load_state_dict(torch.load(saveModelPath))
    return model

