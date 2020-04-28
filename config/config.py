import torch
from enum import unique, Enum
from torchvision.transforms import transforms


@unique
class Model(Enum):
    SINGLETON = "Singleton"
    NET = 'Net'


@unique
class OPTIMIZER(Enum):
    LBFGS = "LBFGS"
    ADAM = "adam"
    SGD = 'SGC'


@unique
class PHASE(Enum):
    TRAIN = "train"
    PREDICT = "predict"


learn_rate = 0.0001
model = Model.NET
default_optimizer = OPTIMIZER.LBFGS
phase = PHASE.PREDICT

singletonContentWeight = 1
singletonStyleWeight = 1e8
netContWeight = 1
netStyleWeight = 1e6

contentImage = '../data/content/g2.jpg'
styleImage = '../data/style/picasso.jpg'
imageDir = '../data/content'
iterations = 300

log_interval = 1
clear_interval = 10
show_interval = 0
imageKeepTime = 100
save_interval = 10

saveModelPath = '../pts/model.pt'
create_dir = '../create'
needIterationTag = False
################
resize = 224
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)


###############


def device():
    return torch.cuda.is_available() and torch.device('gpu') or torch.device('cpu')


def singletonTransform():
    default_trans = [transforms.ToTensor(), tensor_normalizer]
    if resize is not None:
        default_trans = [transforms.Resize(resize), transforms.CenterCrop(resize)] + default_trans
    return transforms.Compose(default_trans)


def optimizer(network, opt=default_optimizer):
    parameters = network.parameters()
    if opt is OPTIMIZER.ADAM:
        return torch.optim.Adam(parameters, learn_rate)
    if opt is OPTIMIZER.LBFGS:
        return torch.optim.LBFGS(parameters)
    return torch.optim.SGD(parameters, learn_rate)
