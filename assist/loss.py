from torch.nn import functional as F


def createContentLossFunc(originContentFeatures, contentWeight):
    def _contentLossFunc(contentFeatures):
        return F.mse_loss(contentFeatures[2], originContentFeatures[2]) * contentWeight

    return _contentLossFunc


def GramMatrix(originMatrix):
    index, channel, high, width = originMatrix.size()
    resizeMatrix = originMatrix.view(index, channel, high * width)
    transposeMatrix = resizeMatrix.transpose(1, 2)
    return resizeMatrix.bmm(transposeMatrix) / (channel * high * width)


def featureGramMatrix(featureMatrix):
    return [GramMatrix(x) for x in featureMatrix]


def createStyleLossFunc(originStyleFeatures, styleWeight):
    originStyleFeatureGram = featureGramMatrix(originStyleFeatures)

    def _styleLossFunc(styleFeatures):
        styleFeatureGram = featureGramMatrix(styleFeatures)
        styleLoss = 0
        for origin, step in zip(originStyleFeatureGram, styleFeatureGram):
            styleLoss += F.mse_loss(step, origin) * styleWeight
        return styleLoss

    return _styleLossFunc


def createLossFunc(originContentFeature, originStyleFeature, styleWeight, contentWeight):
    styleLossFunc = createStyleLossFunc(originStyleFeature, styleWeight)
    contentLossFunc = createContentLossFunc(originContentFeature, contentWeight)

    def lossFunc(currentFeature):
        styleLoss = styleLossFunc(currentFeature)
        contentLoss = contentLossFunc(currentFeature)
        return styleLoss, contentLoss

    return lossFunc
