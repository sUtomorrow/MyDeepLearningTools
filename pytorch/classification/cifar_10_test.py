# -*- coding: utf-8 -*-
# @Time     : 9/13/19 8:25 AM
# @Author   : lty
# @File     : test_pytorch

import os
import torch
import torchvision as tv
from classification.ResNet import *

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

BATCH_SIZE = 128
EPOCH=20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
])

test_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
])

trainset = tv.datasets.CIFAR10(root='/mnt/data4/lty/data/cifar-10', transform=train_transform, train=True, download=True)
testset  = tv.datasets.CIFAR10(root='/mnt/data4/lty/data/cifar-10', transform=test_transform, train=False, download=True)

train_gen = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_gen  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = resnet101(num_classes=10).to(DEVICE)
print(model)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# nllloss = torch.nn.NLLLoss()

for epoch in range(50):
    model.train()
    for batch_idx, (data, target) in enumerate(train_gen):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 80 == 0:
            print('epoch{}: {}/{}, loss:{}'.format(epoch, (batch_idx+1) * BATCH_SIZE, len(train_gen.dataset), loss.item()))

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_gen:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='mean').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_gen)
        print('epoch{}: test loss: {}, acc: {}'.format(epoch, test_loss, correct / len(test_gen.dataset)))

# resnet18 agd:
# epoch49: test loss: -0.4992789445044119, acc: 0.514
# resnet18 adam:
# epoch49: test loss: -0.639438375642028, acc: 0.6399
# resnet101 adam:
# cannot convergence
# resnet101 sgd:

