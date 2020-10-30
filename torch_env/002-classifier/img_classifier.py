#!/home/chhuang/Desktop/venv/bin/python3

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms

torch.backends.cudnn.benchmark = True

def imshow(img):
    img = img/2+0.5
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                nn.Conv2d(3,6,5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(6,16,5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                nn.Linear(16*5*5,120),
                nn.ReLU(inplace=True),
                nn.Linear(120,84),
                nn.ReLU(inplace=True),
                nn.Linear(84,10),
                )

    def forward(self, x):
        return self.net(x)

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10('data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=True)

testset = torchvision.datasets.CIFAR10('data',
                                        train=False,
                                        download=True,
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=128,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(images.shape, labels.shape)
# print('--'.join('%s' % classes[labels[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(images))
print(len(trainloader))

device = torch.device("cuda:0")
net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
    running_loss = 0.0
    for idx, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)


        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # for param in net.parameters():
        #     param.grad = None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (idx+1)%50==0:
            print(f"[{epoch+1}, {idx+1}] loss: {running_loss/50:.3f}")
            running_loss=0

print("Finished Training")
