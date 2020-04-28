from tqdm import tqdm
from config import config


def createLogger(processBar=False):
    def simpleLogger(msg):
        print(msg)

    if not processBar:
        return simpleLogger

    bar = tqdm(total=config.iterations, nrows=10)

    def barLogger(msg, step=config.log_interval):
        bar.update(step)
        bar.set_description(msg+"\tProcess")

    return barLogger


