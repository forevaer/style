from assist.process import getExtract, getModel
import torch
from assist.log import createLogger
from config import config
from assist.data import DataGenerator
from assist.image import imageTensor, createShowImageFunc
from assist.loss import createStyleLossFunc, createContentLossFunc


def prepare():
    device = config.device()
    model = getModel(device)
    extract = getExtract(device)
    data = DataGenerator(config.imageDir)
    targetStyleTensor = imageTensor(config.styleImage).to(device)
    targetStyleFeature = extract(targetStyleTensor)
    styleLossFunc = createStyleLossFunc(targetStyleFeature, config.netStyleWeight)
    optimizer = config.optimizer(model)
    return device, model, optimizer, extract, styleLossFunc, data


def netTrain():
    device, model, optimizer, extract, styleLossFunc, data = prepare()
    model.train()
    log = createLogger(True)
    iteration = [0]

    def f():
        loss = 0
        for item in data:
            optimizer.zero_grad()
            item = item.to(device)
            out = model(item)
            outFeature = extract(out)
            originContentFeature = extract(item)
            contentLoss = createContentLossFunc(originContentFeature, config.netContWeight)(outFeature)
            styleLoss = styleLossFunc(outFeature)
            loss += contentLoss + styleLoss
        if iteration[0] % config.save_interval == 0:
            torch.save(model.state_dict(), config.saveModelPath)
        iteration[0] += 1
        loss.backward(retain_graph=True)
        log(f'iteration : {iteration[0]}, loss : {loss}')
        return loss

    while iteration[0] < config.iterations:
        optimizer.step(f)


def netPredict():
    device, model, optimizer, extract, styleLossFunc, data = prepare()
    model.eval()
    originContentConImage = imageTensor(config.contentImage).to(device)
    generate = model(originContentConImage)
    imshow = createShowImageFunc(True)
    imshow(generate)


if __name__ == '__main__':
    netPredict()
