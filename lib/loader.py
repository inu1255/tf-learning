import os
import numpy as np

class Loader(object):

    def __init__(self, path):
        self.path = path

    def train_data_iterator(self, samples, labels, iteration_steps, chunkSize):
        total = len(samples)
        if total < chunkSize:
            raise Exception('Length of samples must biger than chunkSize')
        if total != len(labels):
            raise Exception('Length of samples and labels must equal')
        stepStart = 0  # initial step
        i = 0
        while i < iteration_steps:
            stepEnd = stepStart + chunkSize
            if stepEnd > total:
                stepEnd -= total
                yield i, np.concatenate((samples[stepStart:total], samples[0:stepEnd])), np.concatenate((labels[stepStart:total], labels[0:stepEnd]))
            else:
                yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            stepStart = stepEnd
            i += 1

    def test_data_iterator(self, samples, labels, chunkSize):
        total = len(samples)
        if total < chunkSize:
            raise Exception('Length of samples must biger than chunkSize')
        if total != len(labels):
            raise Exception('Length of samples and labels must equal')
        stepStart = 0  # initial step
        stepEnd = chunkSize
        i = 0
        while True:
            stepEnd = stepStart + chunkSize
            if stepEnd > total:
                break
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            stepStart = stepEnd
            i += 1


class SVHNLoader(Loader):
    """
    http://ufldl.stanford.edu/housenumbers/
    Street View House Numbers (SVHN): （基于谷歌街景的）一个大规模的房屋门号数据库。
    """

    num_labels = 10
    image_size = 32
    num_channels = 1

    def __init__(self, path):
        self.path = path
        from scipy.io import loadmat as load
        train = load(os.path.join(self.path, 'train_32x32.mat'))
        self.train_samples, self.train_labels = self.reformat(train['X'], train['y'])
        test = load(os.path.join(self.path, 'test_32x32.mat'))
        self.test_samples, self.test_labels = self.reformat(test['X'], test['y'])

    def reformat(self, samples, labels):
        # (图片高，图片宽，通道数，图片数) -> (图片数，图片高，图片宽，通道数)
        samples = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)
        samples = self.normalize(samples)
        labels = labels.flatten()  # (73257, 1) -> (73257, )
        labels[labels == 10] = 0  # [1-10] -> [0-9]
        labels = (np.arange(10) == labels[:, None]).astype(np.uint8)  # to one_hot
        return samples, labels

    def normalize(self, a):
        # a = np.add.reduce(a, keepdims=True, axis=3)  # shape (图片数，图片高，图片宽，通道数)
        # a = a / 3.0
        return a / 128.0 - 1.0
