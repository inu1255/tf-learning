import sys
sys.path.append('.')
import cv2
import numpy as np
import tensorflow as tf
from lib.loader import SVHNLoader
from lib.nnet import simple_conv, shufflenet
from lib import utils

def train():
    loader = SVHNLoader('./data')
    net = shufflenet((32, 32, 3))
    net.train(loader.train_samples, loader.train_labels, iteration_steps=2000, chunkSize=64)
    net.test(loader.test_samples, loader.test_labels, chunkSize=200)
    image = cv2.imread('./2.jpg')
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    # image = utils.gray(image)
    image = utils.normalize(image)
    n = net.predict(image)
    print(np.argmax(n))

def main():
    train()

if __name__ == '__main__':
    main()