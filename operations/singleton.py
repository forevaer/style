from assist.file import createSaveFileNameFunc, trainedPicture
from config import config
from torch import optim
from assist.loss import createLossFunc
from assist.process import getExtract
from assist.log import createLogger
from assist.image import createShowImageFunc, saveTensorImage, imageTensor


def train(contentImage, styleImage, reCreate=None):
    device = config.device()
    contentImage = contentImage.to(device)
    styleImage = styleImage.to(device)
    imshow = None
    if config.show_interval > 0:
        imshow = createShowImageFunc(True)
    if reCreate is None:
        reCreate = contentImage.clone()
    extract = getExtract(device)
    originContentFeature = extract(contentImage)
    originStyleFeature = extract(styleImage)
    # 梯度训练的是reCreate
    optimizer = optim.LBFGS([reCreate.requires_grad_()])
    lossFunc = createLossFunc(originContentFeature, originStyleFeature, config.singletonStyleWeight, config.singletonContentWeight)
    iteration = [0]
    newSaveFileFunc = createSaveFileNameFunc(config.contentImage, config.create_dir)
    trainLogger = createLogger(True)

    def singletonTraining():
        optimizer.zero_grad()
        createFeature = extract(reCreate)
        styleLoss, contentLoss = lossFunc(createFeature)
        loss = styleLoss + contentLoss
        if iteration[0] % config.save_interval == 0:
            saveTensorImage(reCreate, newSaveFileFunc(iteration[0]))
        if (imshow is not None) and (iteration[0] % config.show_interval == 0):
            imshow(reCreate, iteration[0])
        if iteration[0] % config.log_interval == 0:
            trainLogger(
                'iter : {:4.0f} \tcontentLoss : {:10.5f} \tstyleLoss : {:10.5f}'.format(iteration[0], contentLoss,
                                                                                        styleLoss))
        # 单次训练需要记住上次的值，否则会释放
        loss.backward(retain_graph=True)
        iteration[0] += 1
        return loss

    # 梯度回传的就是张量本身而非网络结构
    # 需要对单一的相同的张量进行重复计算
    while iteration[0] < config.iterations:
        optimizer.step(singletonTraining)
    return reCreate


def singletonEntrance():
    contentImage = imageTensor(config.contentImage)
    styleImage = imageTensor(config.styleImage)
    trainedImage = imageTensor(trainedPicture(config.contentImage))
    createImage = train(contentImage, styleImage, trainedImage)

    return createImage


if __name__ == '__main__':
    singletonEntrance()
