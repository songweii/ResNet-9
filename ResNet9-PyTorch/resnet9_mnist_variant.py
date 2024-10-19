import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm, trange
import pickle
import random
import time
import os

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1 = self._make_conv_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(self._make_conv_block(128, 128), self._make_conv_block(128, 128))
        self.shortcut1 = self._make_conv_block(128, 128)
        self.res2 = nn.Sequential(self._make_conv_block(128, 256), self._make_conv_block(256, 256))
        self.shortcut2 = self._make_conv_block(128, 256)
        self.res3 = nn.Sequential(self._make_conv_block(256, 256), self._make_conv_block(256, 256))
        self.shortcut3 = self._make_conv_block(256, 256)
        self.classifier = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
                                        nn.Flatten(),
                                        nn.Linear(256, num_classes))

    def _make_conv_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]

        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + self.shortcut1(x)
        x = self.maxpool(x)
        x = self.res2(x) + self.shortcut2(x)
        x = self.maxpool(x) 
        x = self.res3(x) + self.shortcut3(x)
        x = self.classifier(x)
        return x


def calculate_accuracy(pred, y):
    top_pred = pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, dataloader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    batch_loss = 0
    batch_acc = 0
    batch_losses = []
    batch_accuracies = []
    model.train()

    for i, (features, labels) in enumerate(tqdm(dataloader, desc='Training Phase', leave=False)):
        # Sending features and labels to device
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass; making predictions and calculating loss
        pred = model(features)
        loss = criterion(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracies
        acc = calculate_accuracy(pred, labels)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        batch_loss += loss.item()
        batch_acc += acc.item()

        # 每100个batch记录一次平均loss
        if (i + 1) % 100 == 0:
            batch_losses.append(batch_loss/100)
            batch_accuracies.append(batch_acc/100)
            batch_loss = 0
            batch_acc = 0

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), batch_losses, batch_accuracies


def test(model, dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Test Phase', leave=False):
            features = features.to(device)
            labels = labels.to(device)

            pred = model(features)

            loss = criterion(pred, labels)
            acc = calculate_accuracy(pred, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def plot_train_loss(filepath, picpath):
    # 绘制曲线图
    import matplotlib.pyplot as plt

    # 加载数据并绘制图形
    with open(filepath, 'rb') as f:
        loaded_history = pickle.load(f)

    # 绘制batch损失和准确率曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loaded_history['batch_losses'], label='Batch Loss', color='blue')
    plt.xlabel('Per 100 Batches')
    plt.ylabel('Loss')
    plt.title('Batch Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loaded_history['batch_accuracies'], label='Batch Accuracy', color='orange')
    plt.xlabel('Per 100 Batches')
    plt.ylabel('Accuracy')
    plt.title('Batch Accuracy Curve')
    plt.legend()

    # 保存图像到文件
    plt.tight_layout()
    plt.savefig(picpath, bbox_inches='tight')
    plt.close()  # 关闭图像以释放内存


# 固定随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    # 记录时间
    total_time = 0

    # 设置随机种子
    seed_value = 42
    set_random_seed(seed_value)
    
    ## hyperparameters ##
    batch_size = 8
    learning_rate = 0.01
    EPOCHS = 8

    ## load data ##
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False) # if data prepared
    # train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True) # if no prepared data
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet9(1, 10).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # history = {
    #     'train_loss': [],
    #     'train_acc': [],
    #     'test_loss': [],
    #     'test_acc': []
    # }
    history = {'batch_losses': [], 'batch_accuracies': []}

    # record every epoch
    for epoch in trange(EPOCHS, desc="Epoch Number"):
        start_time = time.time()
        train_loss, train_acc, batch_losses, batch_accuracies = train(model, train_loader, optimizer, criterion, device)
        end_time = time.time()
        total_time += (end_time - start_time)
        # history['train_loss'].append(train_loss)
        # history['train_acc'].append(train_acc)
        history['batch_losses'].extend(batch_losses)
        history['batch_accuracies'].extend(batch_accuracies)

        test_loss, test_acc = test(model, test_loader, criterion, device)
        # history['test_loss'].append(test_loss)
        # history['test_acc'].append(test_acc)

        print(f"Epoch: {epoch + 1:02}")
        print(f"\tTrain Loss: {train_loss:>7.3f} | Training Accuracy: {train_acc * 100:>7.2f}%")
        print(f"\tTest Loss: {test_loss:>7.3f} | Test Accuracy: {test_acc * 100:>7.2f}%")

    print(f"Variant (w/ pytorch) Execution time: {total_time:.4f} seconds")
    dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dir, 'variant_training_history.pkl')
    picpath = os.path.join(dir, 'variant_loss_accuracy_curve.png')

    # 保存batch损失和准确率数据到文件
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)

    plot_train_loss(filepath = filepath, picpath = picpath)

if __name__ == '__main__':
    main()

