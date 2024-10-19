import cupy as cp
import os


class fc_sigmoid:
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_param()

    def init_param(self):
        self.kernel = cp.random.uniform(
            low=-cp.sqrt(6.0 / (self.out_channels + self.in_channels)),
            high=cp.sqrt(6.0 / (self.in_channels + self.out_channels)),
            size=(self.out_channels, self.in_channels)
        )
        self.bias = cp.zeros([self.out_channels])

    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        self.in_tensor = in_tensor.reshape(in_tensor.shape[0], -1).copy()
        assert self.in_tensor.shape[1] == self.kernel.shape[1]
        self.out_tensor = cp.dot(self.in_tensor, self.kernel.T) + self.bias.T
        self.out_tensor = 1.0 / (1.0 + cp.exp(-self.out_tensor))
        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape
        nonlinear_diff = self.out_tensor * (1 - self.out_tensor) * out_diff_tensor
        kernel_diff = cp.dot(nonlinear_diff.T, self.in_tensor).squeeze()
        bias_diff = cp.sum(nonlinear_diff, axis=0).reshape(self.bias.shape)
        self.in_diff_tensor = cp.dot(nonlinear_diff, self.kernel).reshape(self.shape)
        self.kernel -= lr * kernel_diff
        self.bias -= lr * bias_diff

    def save(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)

        cp.save(os.path.join(path, "fc_weight"), self.kernel)
        cp.save(os.path.join(path, "fc_bias"), self.bias)

    def load(self, path):
        assert os.path.exists(path)

        self.kernel = cp.load(os.path.join(path, "fc_weight.npy"))
        self.bias = cp.load(os.path.join(path, "fc_bias.npy"))


class conv_layer:
    def __init__(self, in_channels, out_channels, kernel_h, kernel_w, same=True, stride=1, shift=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.same = same
        self.stride = stride
        self.shift = shift

        self.init_param()

    def init_param(self):
        self.kernel = cp.random.uniform(
            low=-cp.sqrt(6.0 / (self.out_channels + self.in_channels * self.kernel_h * self.kernel_w)),
            high=cp.sqrt(6.0 / (self.in_channels + self.out_channels * self.kernel_h * self.kernel_w)),
            size=(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)
        )
        self.bias = cp.zeros([self.out_channels]) if self.shift else None

    @staticmethod
    def pad(in_tensor, pad_h, pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        padded = cp.zeros([batch_num, in_channels, in_h + 2 * pad_h, in_w + 2 * pad_w])
        padded[:, :, pad_h:pad_h + in_h, pad_w:pad_w + in_w] = in_tensor
        return padded

    @staticmethod
    def convolution(in_tensor, kernel, stride=1, dilate=1):
        batch_num, in_channels, in_h, in_w = in_tensor.shape
        assert kernel.shape[1] == in_channels
        out_channels, _, kernel_h, kernel_w = kernel.shape

        # 计算输出尺寸
        out_h = int((in_h - (dilate * (kernel_h - 1) + 1)) / stride + 1)
        out_w = int((in_w - (dilate * (kernel_w - 1) + 1)) / stride + 1)

        # 使用 as_strided 创建一个扩展的输入张量视图
        shape = (batch_num, out_h, out_w, in_channels, kernel_h, kernel_w)
        strides = (in_tensor.strides[0], in_tensor.strides[2]*stride, in_tensor.strides[3]*stride,
                in_tensor.strides[1], in_tensor.strides[2]*dilate, in_tensor.strides[3]*dilate)
        extended_in = cp.lib.stride_tricks.as_strided(in_tensor, shape=shape, strides=strides)

        # 将扩展的输入张量与核进行逐元素乘法，然后求和
        kernel = kernel.reshape(out_channels, -1)
        extended_in = extended_in.reshape(batch_num, out_h, out_w, -1)
        
        # 确保 extended_in 和 kernel 的形状匹配
        assert extended_in.shape[-1] == kernel.shape[-1]

        out_tensor = cp.tensordot(extended_in, kernel, axes=((-1,), (-1,)))
        out_tensor = out_tensor.reshape(batch_num, out_h, out_w, out_channels)
        out_tensor = out_tensor.transpose(0, 3, 1, 2)

        return out_tensor

    def forward(self, in_tensor):
        if self.same:
            in_tensor = conv_layer.pad(in_tensor, int((self.kernel_h - 1) / 2), int((self.kernel_w - 1) / 2))

        self.in_tensor = in_tensor.copy()
        self.out_tensor = conv_layer.convolution(in_tensor, self.kernel, self.stride)

        if self.shift:
            self.out_tensor += self.bias.reshape(1, self.out_channels, 1, 1)

        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape

        if self.shift:
            bias_diff = cp.sum(out_diff_tensor, axis=(0, 2, 3)).reshape(self.bias.shape)
            self.bias -= lr * bias_diff

        batch_num = out_diff_tensor.shape[0]
        out_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        extend_out = cp.zeros([batch_num, out_channels, out_h, out_w, self.stride * self.stride])
        extend_out[:, :, :, :, 0] = out_diff_tensor
        extend_out = extend_out.reshape(batch_num, out_channels, out_h, out_w, self.stride, self.stride)
        extend_out = extend_out.transpose(0, 1, 2, 4, 3, 5).reshape(batch_num, out_channels, out_h * self.stride,
                                                                    out_w * self.stride)

        kernel_diff = conv_layer.convolution(self.in_tensor.transpose(1, 0, 2, 3), extend_out.transpose(1, 0, 2, 3))
        kernel_diff = kernel_diff.transpose(1, 0, 2, 3)

        padded = conv_layer.pad(extend_out, self.kernel_h - 1, self.kernel_w - 1)
        kernel_trans = self.kernel.reshape(self.out_channels, self.in_channels, self.kernel_h * self.kernel_w)
        kernel_trans = kernel_trans[:, :, ::-1].reshape(self.kernel.shape)
        self.in_diff_tensor = conv_layer.convolution(padded, kernel_trans.transpose(1, 0, 2, 3))
        assert self.in_diff_tensor.shape == self.in_tensor.shape

        if self.same:
            pad_h = int((self.kernel_h - 1) / 2)
            pad_w = int((self.kernel_w - 1) / 2)
            if pad_h == 0 and pad_w != 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, :, pad_w:-pad_w]
            elif pad_h != 0 and pad_w == 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, :]
            elif pad_h != 0 and pad_w != 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, pad_w:-pad_w]

        self.kernel -= lr * kernel_diff

    def save(self, path, conv_num):
        if os.path.exists(path) == False:
            os.mkdir(path)

        cp.save(os.path.join(path, "conv{}_weight".format(conv_num)), self.kernel)
        if self.shift:
            cp.save(os.path.join(path, "conv{}_bias".format(conv_num)), self.bias)

        return conv_num + 1

    def load(self, path, conv_num):
        assert os.path.exists(path)

        self.kernel = cp.load(os.path.join(path, "conv{}_weight.npy".format(conv_num)))
        if self.shift:
            self.bias = cp.load(os.path.join(path, "conv{}_bias.npy").format(conv_num))

        return conv_num + 1


class max_pooling:
    def __init__(self, kernel_h, kernel_w, stride, same=False):
        assert stride > 1
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.same = same
        self.stride = stride

    @staticmethod
    def pad(in_tensor, pad_h, pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        padded = cp.zeros([batch_num, in_channels, in_h + 2 * pad_h, in_w + 2 * pad_w])
        padded[:, :, pad_h:pad_h + in_h, pad_w:pad_w + in_w] = in_tensor
        return padded

    def forward(self, in_tensor):
        if self.same:
            in_tensor = max_pooling.pad(in_tensor, int((self.kernel_h - 1) / 2), int((self.kernel_w - 1) / 2))
        self.shape = in_tensor.shape

        batch_num, in_channels, in_h, in_w = in_tensor.shape
        out_h = int((in_h - self.kernel_h) / self.stride) + 1
        out_w = int((in_w - self.kernel_w) / self.stride) + 1

        # 使用 as_strided 创建一个扩展的输入张量视图
        shape = (batch_num, in_channels, out_h, out_w, self.kernel_h, self.kernel_w)
        strides = (in_tensor.strides[0], in_tensor.strides[1], in_tensor.strides[2] * self.stride, 
                   in_tensor.strides[3] * self.stride, in_tensor.strides[2], in_tensor.strides[3])
        extended_in = cp.lib.stride_tricks.as_strided(in_tensor, shape=shape, strides=strides)

        # 将扩展的输入张量展平，并找到最大值及其索引
        extended_in = extended_in.reshape(batch_num, in_channels, out_h, out_w, -1)
        out_tensor = cp.max(extended_in, axis=-1)
        self.maxindex = cp.argmax(extended_in, axis=-1)

        self.out_tensor = out_tensor
        return self.out_tensor

    def backward(self, out_diff_tensor, lr=0):
        assert out_diff_tensor.shape == self.out_tensor.shape
        batch_num = out_diff_tensor.shape[0]
        in_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        in_h = self.shape[2]
        in_w = self.shape[3]

        out_diff_tensor = out_diff_tensor.reshape(batch_num * in_channels, out_h, out_w)
        self.maxindex = self.maxindex.reshape(batch_num * in_channels, out_h, out_w)

        self.in_diff_tensor = cp.zeros([batch_num * in_channels, in_h, in_w])

        # 计算 h_index 和 w_index
        h_index = (self.maxindex // self.kernel_h).astype(cp.int32)
        w_index = self.maxindex % self.kernel_h

        # 使用高级索引填充 in_diff_tensor
        indices = (
            cp.arange(batch_num * in_channels).reshape(-1, 1, 1),
            h_index + self.stride * cp.arange(out_h).reshape(1, -1, 1),
            w_index + self.stride * cp.arange(out_w).reshape(1, 1, -1)
        )
        self.in_diff_tensor[indices] += out_diff_tensor

        self.in_diff_tensor = self.in_diff_tensor.reshape(batch_num, in_channels, in_h, in_w)

        if self.same:
            pad_h = int((self.kernel_h - 1) / 2)
            pad_w = int((self.kernel_w - 1) / 2)
            self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, pad_w:-pad_w]

        return self.in_diff_tensor


class global_average_pooling:
    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        out_tensor = in_tensor.reshape(in_tensor.shape[0], in_tensor.shape[1], -1).mean(axis=-1)
        return out_tensor.reshape(in_tensor.shape[0], in_tensor.shape[1], 1, 1)

    def backward(self, out_diff_tensor, lr=0):
        batch_num = self.shape[0]
        in_channels = self.shape[1]
        in_h = self.shape[2]
        in_w = self.shape[3]
        assert out_diff_tensor.shape == (batch_num, in_channels, 1, 1)

        in_diff_tensor = cp.zeros(list(self.shape))
        in_diff_tensor += out_diff_tensor / (in_h * in_w)

        self.in_diff_tensor = in_diff_tensor


class relu:
    def forward(self, in_tensor):
        self.in_tensor = in_tensor.copy()
        self.out_tensor = in_tensor.copy()
        self.out_tensor[self.in_tensor < 0.0] = 0.0
        return self.out_tensor

    def backward(self, out_diff_tensor, lr=0):
        assert self.out_tensor.shape == out_diff_tensor.shape
        self.in_diff_tensor = out_diff_tensor.copy()
        self.in_diff_tensor[self.in_tensor < 0.0] = 0.0


class bn_layer:
    def __init__(self, neural_num, moving_rate=0.1):
        self.gamma = cp.random.uniform(low=0, high=1, size=neural_num)
        self.bias = cp.zeros([neural_num])
        self.moving_avg = cp.zeros([neural_num])
        self.moving_var = cp.ones([neural_num])
        self.neural_num = neural_num
        self.moving_rate = moving_rate
        self.is_train = True
        self.epsilon = 1e-5

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def forward(self, in_tensor):
        assert in_tensor.shape[1] == self.neural_num

        self.in_tensor = in_tensor.copy()

        if self.is_train:
            mean = in_tensor.mean(axis=(0, 2, 3))
            var = in_tensor.var(axis=(0, 2, 3))
            self.moving_avg = mean * self.moving_rate + (1 - self.moving_rate) * self.moving_avg
            self.moving_var = var * self.moving_rate + (1 - self.moving_rate) * self.moving_var
            self.var = var
            self.mean = mean
        else:
            mean = self.moving_avg
            var = self.moving_var

        self.normalized = (in_tensor - mean.reshape(1, -1, 1, 1)) / cp.sqrt(var.reshape(1, -1, 1, 1) + self.epsilon)
        out_tensor = self.gamma.reshape(1, -1, 1, 1) * self.normalized + self.bias.reshape(1, -1, 1, 1)

        return out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.in_tensor.shape
        assert self.is_train

        m = self.in_tensor.shape[0] * self.in_tensor.shape[2] * self.in_tensor.shape[3]

        normalized_diff = self.gamma.reshape(1, -1, 1, 1) * out_diff_tensor
        var_diff = -0.5 * cp.sum(normalized_diff * self.normalized, axis=(0, 2, 3)) / (self.var + self.epsilon)
        mean_diff = -1.0 * cp.sum(normalized_diff, axis=(0, 2, 3)) / cp.sqrt(self.var + self.epsilon)
        in_diff_tensor1 = normalized_diff / cp.sqrt(self.var.reshape(1, -1, 1, 1) + self.epsilon)
        in_diff_tensor2 = var_diff.reshape(1, -1, 1, 1) * (self.in_tensor - self.mean.reshape(1, -1, 1, 1)) * 2 / m
        in_diff_tensor3 = mean_diff.reshape(1, -1, 1, 1) / m
        self.in_diff_tensor = in_diff_tensor1 + in_diff_tensor2 + in_diff_tensor3

        gamma_diff = cp.sum(self.normalized * out_diff_tensor, axis=(0, 2, 3))
        self.gamma -= lr * gamma_diff

        bias_diff = cp.sum(out_diff_tensor, axis=(0, 2, 3))
        self.bias -= lr * bias_diff

    def save(self, path, bn_num):
        if not os.path.exists(path):
            os.mkdir(path)

        cp.save(os.path.join(path, "bn{}_weight".format(bn_num)), self.gamma)
        cp.save(os.path.join(path, "bn{}_bias".format(bn_num)), self.bias)
        cp.save(os.path.join(path, "bn{}_mean".format(bn_num)), self.moving_avg)
        cp.save(os.path.join(path, "bn{}_var".format(bn_num)), self.moving_var)

        return bn_num + 1

    def load(self, path, bn_num):
        assert os.path.exists(path)

        self.gamma = cp.load(os.path.join(path, "bn{}_weight.npy".format(bn_num)))
        self.bias = cp.load(os.path.join(path, "bn{}_bias.npy".format(bn_num)))
        self.moving_avg = cp.load(os.path.join(path, "bn{}_mean.npy".format(bn_num)))
        self.moving_var = cp.load(os.path.join(path, "bn{}_var.npy".format(bn_num)))

        return bn_num + 1
