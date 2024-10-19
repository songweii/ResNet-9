from components import *


class ResBlock:
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        self.path1 = [
            conv_layer(in_channels, out_channels, 3, 3, stride=stride, shift=False),
            bn_layer(out_channels),
            relu(),
            conv_layer(out_channels, out_channels, 3, 3, shift=False),
            bn_layer(out_channels)
        ]
        self.path2 = shortcut
        self.relu = relu()

    def train(self):
        self.path1[1].train()
        self.path1[4].train()
        if self.path2 is not None:
            self.path2[1].train()

    def eval(self):
        self.path1[1].eval()
        self.path1[4].eval()
        if self.path2 is not None:
            self.path2[1].eval()

    def forward(self, in_tensor):
        x1 = in_tensor.copy()
        x2 = in_tensor.copy()

        for l in self.path1:
            x1 = l.forward(x1)
        if self.path2 is not None:
            for l in self.path2:
                x2 = l.forward(x2)
        self.out_tensor = self.relu.forward(x1 + x2)

        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert self.out_tensor.shape == out_diff_tensor.shape

        self.relu.backward(out_diff_tensor, lr)
        x1 = self.relu.in_diff_tensor
        x2 = x1.copy()

        for l in range(1, len(self.path1) + 1):
            self.path1[-l].backward(x1, lr)
            x1 = self.path1[-l].in_diff_tensor

        if self.path2 is not None:
            for l in range(1, len(self.path2) + 1):
                self.path2[-l].backward(x2, lr)
                x2 = self.path2[-l].in_diff_tensor

        self.in_diff_tensor = x1 + x2

    def save(self, path, conv_num, bn_num):
        conv_num = self.path1[0].save(path, conv_num)
        bn_num = self.path1[1].save(path, bn_num)
        conv_num = self.path1[3].save(path, conv_num)
        bn_num = self.path1[4].save(path, bn_num)

        if self.path2 is not None:
            conv_num = self.path2[0].save(path, conv_num)
            bn_num = self.path2[1].save(path, bn_num)

        return conv_num, bn_num

    def load(self, path, conv_num, bn_num):
        conv_num = self.path1[0].load(path, conv_num)
        bn_num = self.path1[1].load(path, bn_num)
        conv_num = self.path1[3].load(path, conv_num)
        bn_num = self.path1[4].load(path, bn_num)

        if self.path2 is not None:
            conv_num = self.path2[0].load(path, conv_num)
            bn_num = self.path2[1].load(path, bn_num)

        return conv_num, bn_num


def ResBlockStack(in_channels, out_channels, block_num, stride):
    shortcut = [
        conv_layer(in_channels, out_channels, 1, 1, stride=stride, shift=False),
        bn_layer(out_channels)
    ]
    layers = [ResBlock(in_channels, out_channels, stride=stride, shortcut=shortcut)]

    for _ in range(block_num - 1):
        layers.append(ResBlock(out_channels, out_channels))

    return layers


class ResNet9:
    def __init__(self, num_classes):
        self.pre = [
            conv_layer(1, 64, 3, 3, stride=1, shift=False),
            bn_layer(64),
            relu(),
            conv_layer(64, 128, 3, 3, stride=1, shift=False),
            bn_layer(128),
            relu(),
            max_pooling(2, 2, 2, same=False)
        ]
        self.layer1 = ResBlockStack(128, 128, 1, 1)
        self.pool_1 = max_pooling(2, 2, 2, same=False)
        self.layer2 = ResBlockStack(128, 256, 1, 1)
        self.pool_2 = max_pooling(2, 2, 2, same=False)
        self.layer3 = ResBlockStack(256, 256, 1, 1)

        # self.avg = global_average_pooling()
        self.max_pool = max_pooling(2, 2, 2, same=False)
        self.fc = fc_sigmoid(256, num_classes)

    def train(self):
        self.pre[1].train()
        self.pre[4].train()
        for layer in self.layer1:
            layer.train()
        for layer in self.layer2:
            layer.train()
        for layer in self.layer3:
            layer.train()

    def eval(self):
        self.pre[1].eval()
        self.pre[4].eval()
        for layer in self.layer1:
            layer.eval()
        for layer in self.layer2:
            layer.eval()
        for layer in self.layer3:
            layer.eval()

    def forward(self, in_tensor):
        x = in_tensor
        # print(x.shape)
        for layer in self.pre:
            x = layer.forward(x)
            # print(x.shape)
        for layer in self.layer1:
            x = layer.forward(x)
            # print(x.shape)

        x = self.pool_1.forward(x)
        for layer in self.layer2:
            x = layer.forward(x)
            # print(x.shape)
        x = self.pool_2.forward(x)

        for layer in self.layer3:
            x = layer.forward(x)
            # print(x.shape)
        x = self.max_pool.forward(x)
        # print(x.shape)
        out_tensor = self.fc.forward(x)

        return out_tensor

    def backward(self, out_diff_tensor, lr):
        x = out_diff_tensor
        self.fc.backward(x, lr)
        x = self.fc.in_diff_tensor
        self.max_pool.backward(x, lr)
        x = self.max_pool.in_diff_tensor

        for layer in range(1, len(self.layer3) + 1):
            self.layer3[-layer].backward(x, lr)
            x = self.layer3[-layer].in_diff_tensor

        self.pool_2.backward(x, lr)
        x = self.pool_2.in_diff_tensor
        for layer in range(1, len(self.layer2) + 1):
            self.layer2[-layer].backward(x, lr)
            x = self.layer2[-layer].in_diff_tensor
        self.pool_1.backward(x, lr)
        x = self.pool_1.in_diff_tensor

        for layer in range(1, len(self.layer1) + 1):
            self.layer1[-layer].backward(x, lr)
            x = self.layer1[-layer].in_diff_tensor
        for layer in range(1, len(self.pre) + 1):
            self.pre[-layer].backward(x, lr)
            x = self.pre[-layer].in_diff_tensor
        self.in_diff_tensor = x

    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], -1)
        return cp.argmax(out_tensor, axis=1)

    def save(self, path):
        conv_num = 0
        bn_num = 0

        if not os.path.exists(path):
            os.mkdir(path)

        conv_num = self.pre[0].save(path, conv_num)
        bn_num = self.pre[1].save(path, bn_num)
        conv_num = self.pre[3].save(path, conv_num)
        bn_num = self.pre[4].save(path, bn_num)

        for layer in self.layer1:
            conv_num, bn_num = layer.save(path, conv_num, bn_num)

        for layer in self.layer2:
            conv_num, bn_num = layer.save(path, conv_num, bn_num)

        for layer in self.layer3:
            conv_num, bn_num = layer.save(path, conv_num, bn_num)

        self.fc.save(path)

    def load(self, path):
        conv_num = 0
        bn_num = 0

        conv_num = self.pre[0].load(path, conv_num)
        bn_num = self.pre[1].load(path, bn_num)
        conv_num = self.pre[3].load(path, conv_num)
        bn_num = self.pre[4].load(path, bn_num)

        for layer in self.layer1:
            conv_num, bn_num = layer.load(path, conv_num, bn_num)

        for layer in self.layer2:
            conv_num, bn_num = layer.load(path, conv_num, bn_num)

        for layer in self.layer3:
            conv_num, bn_num = layer.load(path, conv_num, bn_num)

        self.fc.load(path)
