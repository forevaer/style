from config.config import model, Model, PHASE, phase
from operations import singletonEntrance, network

if __name__ == '__main__':
    if model is Model.SINGLETON:
        singletonEntrance()
    if model is Model.NET:
        if phase is PHASE.TRAIN:
            network.netTrain()
        else:
            network.netPredict()
