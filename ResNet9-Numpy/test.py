import numpy as np


def test(net, x_test, t_test):
    net.eval()
    images, labels = x_test, t_test
    infers = np.zeros([images.shape[0]], dtype=np.int32)
    for i in range(images.shape[0]):
        infers[i] = net.inference(images[i].reshape(1, 1, 28, 28))
    gt = np.argmax(labels, axis=1)
    return np.sum(infers == gt) / infers.shape[0]
