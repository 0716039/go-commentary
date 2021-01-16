import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from Process_finetune import sgfs2features
from model import CNN
from sklearn.metrics import accuracy_score
    
cnn = CNN()
output = sgfs2features('None', '../sgfs/pretrain_sgf3.sgf')

BW_feature = np.array(output[0])
onehot = np.array(output[1])
train_x = torch.Tensor(BW_feature)
train_y = torch.Tensor(onehot)
trainset = TensorDataset(train_x, train_y)

testdata = sgfs2features('None', '../sgfs/humantag_test.sgf')

test_feature = np.array(testdata[0])
test_onehot = np.array(testdata[1])
test_x = torch.Tensor(test_feature)
test_y = torch.Tensor(test_onehot)
testset = TensorDataset(test_x, test_y)


def accuracy(predictions, labels):
    return torch.mean((predictions == labels).float())
    
BATCH = 256
LR = 0.01
trainloader = DataLoader(trainset, batch_size= BATCH, shuffle=True)
optimizer = torch.optim.Adam(cnn.parameters(), lr= LR)
loss_func = nn.BCELoss()

cnn.train()
train_loss = 0.0
train_correct = 0
for epoch in range(5):
    running_accuracy = 0.0
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data #batch_x, batch_y = batch_data
        outputs = cnn(inputs)
        sigmoid = nn.Sigmoid()
        outputs = sigmoid(outputs)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs = outputs.detach().numpy()#RuntimeError: Can't call numpy() on Tensor that requires grad.
        pred = np.round(outputs)
        pred = torch.tensor(pred)#AttributeError: 'bool' object has no attribute 'float'
        running_accuracy += accuracy(pred, labels)
        if i % 5 == 0:
            print('Epoch: ', epoch, '|train loss: %.4f' % loss.item())
    
    running_accuracy /= len(trainloader.dataset)
    running_accuracy *= BATCH
    print('Epoch: ', epoch, '|train accuracy: %4f' % running_accuracy)

testloader = DataLoader(testset, batch_size = BATCH, shuffle=True)

cnn.eval()
test_loss = 0.0
correct = 0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = cnn(inputs)
        sigmoid = nn.Sigmoid()
        outputs = sigmoid(outputs)
        loss = loss_func(outputs, labels)
        test_loss += loss.item()
        pred = np.round(outputs)
        #correct += pred.eq(labels.view_as(pred)).sum().item()
        correct += accuracy(pred, labels)

test_loss /= len(testloader.dataset)
test_loss *= BATCH
correct /= len(testloader.dataset)
correct *= BATCH
print('\nTestset: average loss: %4f' % test_loss, 'accuracy: %4f' % correct)