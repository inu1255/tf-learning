import numpy as np

def gray(image):
    image = np.add.reduce(image, keepdims=True, axis=len(image.shape) - 1)  # shape (图片数，图片高，图片宽，通道数)
    image = image / 3.0
    return image

def normalize(image):
    image = image / 128.0 - 1.0
    return image


def softmax(x, axis=1):
    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)


def train_data_iterator(samples, labels, iteration_steps, chunkSize):
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
            yield i, np.concatenate((samples[stepStart:total], samples[0:stepEnd])), np.concatenate(
                (labels[stepStart:total], labels[0:stepEnd]))
        else:
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
        stepStart = stepEnd
        i += 1


def test_data_iterator(samples, labels, chunkSize):
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