from model import *
from data.mnist import load_mnist


def demo(model_path, x_test, t_test, cls_num):
    net = ResNet9(cls_num)
    net.load(model_path)
    net.eval()

    images, labels = x_test, cp.argmax(t_test, axis=1)
    top3_pred = cp.zeros([images.shape[0], 3], dtype=cp.int32)

    num_accurate = 0
    unknown_p = 1 / (cls_num + 1)

    for i in range(images.shape[0]):
        out = net.forward(images[i].reshape(1, 1, 28, 28)).reshape(-1)
        order = cp.argsort(out)[::-1]

        if out[order[0]] < unknown_p:
            top3_pred[i, 0] = cls_num - 1
            top3_pred[i, 1:3] = order[0:2]
        elif out[order[1]] < unknown_p:
            top3_pred[i, 0] = order[0]
            top3_pred[i, 1] = cls_num - 1
            top3_pred[i, 2] = order[1]
        elif out[order[2]] < unknown_p:
            top3_pred[i, 0:2] = order[0:2]
            top3_pred[i, 2] = cls_num - 1
        else:
            top3_pred[i] = order[0:3]

        print("Top 3 prediction: {}  {}  {} || truth: {}".format(top3_pred[i, 0], top3_pred[i, 1], top3_pred[i, 2], labels[i]))
        if labels[i] == top3_pred[i, 0] or labels[i] == top3_pred[i, 1] or labels[i] == top3_pred[i, 2]:
            num_accurate += 1

    print("Among {} images, {} images are classified correctly.".format(images.shape[0], num_accurate))
    print("The accurate rate is {}".format(num_accurate / images.shape[0]))


if __name__ == "__main__":
    model_path = "model"
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
    demo(model_path, x_test, t_test, cls_num=10)
