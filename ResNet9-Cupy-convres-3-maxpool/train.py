from model import *
from test import test
import matplotlib.pyplot as plt
from data.mnist import load_mnist
import time


class Trainer:
    def __init__(self, model, x_train, t_train, num_classes, init_lr, batch_size, train_size):
        self.net = model
        self.x_train, self.t_train = x_train, t_train
        self.cls_num = num_classes
        self.lr = init_lr
        self.batch_size = batch_size
        self.train_size = train_size

    def set_lr(self, lr):
        self.lr = lr

    def iterate(self):
        batch_mask = cp.random.choice(self.train_size, self.batch_size)
        images = self.x_train[batch_mask]
        labels = self.t_train[batch_mask]

        out_tensor = self.net.forward(images)
        one_hot_labels = labels

        loss = cp.sum(-one_hot_labels * cp.log(out_tensor) - (1 - one_hot_labels) * cp.log(
            1 - out_tensor)) / self.batch_size
        out_diff_tensor = (out_tensor - one_hot_labels) / out_tensor / (1 - out_tensor) / self.batch_size

        self.net.backward(out_diff_tensor, self.lr)

        return cp.asnumpy(loss)


if __name__ == '__main__':
    batch_size = 8
    cls_num = 10

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
    train_size = x_train.shape[0]  # 即整个mnist所含数据, x_train.shape --> (60000, 784)

    model = ResNet9(cls_num)
    # model.to_gpu()

    iters = 60000
    init_lr = 0.001
    trainer = Trainer(model, x_train, t_train, cls_num, init_lr, batch_size, train_size)
    history, accurate = [], []
    loss_sum = 0

    model.train()
    # plt.figure(figsize=(10, 5))
    # plt.ion()
    start = time.time()
    test_time = 0
    for i in range(iters):
        loss = cp.asnumpy(trainer.iterate())
        loss_sum += loss
        if i % 100 == 0 and i != 0:
            end = time.time()
            print(f"Total training time: {end - start}s")
            history.append(loss_sum / 100)
            print("iteration = {} || loss = {}".format(str(i), str(loss_sum / 100)))
            loss_sum = 0
            if i % 1000 == 0:
                start_test = time.time()
                model.eval()
                acc = test(model, x_test, t_test)
                accurate.append(acc)
                print(f"Accuracy on test-set: {acc}")
                model.save("model")
                model.train()
                end_test = time.time()
                test_time += (end_test - start_test)

        # plt.cla()
        # plt.subplot(1, 2, 1)
        # plt.plot(loss)
        # plt.subplot(1, 2, 2)
        # plt.plot(accurate)
        # plt.pause(0.1)

        # if i == 15000:
        #     trainer.set_lr(0.001)

    # end = time.time()
    # print(f"Total training time: {end - start - test_time}s")

    plt.figure(figsize=(9, 4))
    # plt.ion()
    # plt.cla()
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.subplot(1, 2, 2)
    plt.plot(accurate)
    # plt.pause(0.1)

    # plt.ioff()
    plt.savefig('loss.png', dpi=600)
    plt.show()
