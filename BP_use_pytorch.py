# coding=utf-8
'''
@Author: TJUZQC
@Date: 2019-12-09 12:56:14
@LastEditors: TJUZQC
@LastEditTime: 2019-12-09 17:00:05
@Description: None
@FilePath: \ANN\BP_use_paddle.py
'''
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
import torch.nn.functional as F

class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN,self).__init__()
        self.fc_in = torch.nn.Linear(1,30)
        self.activation = torch.nn.Tanh()
        self.fc_hidden = torch.nn.Linear(30,30)
        self.fc_out = torch.nn.Linear(30,1)
    def forward(self, x):
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.fc_hidden(x)
        x = self.activation(x)
        x = self.fc_hidden(x)
        x = self.activation(x)
        x = self.fc_out(x)
        return x

def genrate_actual_label(x):
    noise = np.random.normal(0, 0.05, x.shape)
    return a * np.sin(b * x) + c * np.cos(d * x)

a=2
b=2
c=3
d=3

sample_domain = (-2 * np.pi, 2 * np.pi)
sample_num = 1000
train_x = np.linspace(sample_domain[0], sample_domain[1], sample_num)
train_x = train_x.reshape(-1,1).astype('float32')
train_y = genrate_actual_label(train_x)
test_x = np.linspace(sample_domain[0], sample_domain[1], sample_num)
test_x = test_x.reshape(-1,1).astype('float32')
test_y = genrate_actual_label(test_x)
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

epochs = 900
model = ANN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
for epoch in range(epochs):
    model.train()
    for batch_idx,(x,y) in enumerate(train_loader):
        y = y/2
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

with torch.no_grad():
    model.eval()
    out = []
    for batch_idx, (x,y) in enumerate(test_loader):
        out.append(model(x)*2)
plt.scatter(test_x, test_y, color='red', label='GT')
plt.plot(test_x, out, label='predict')
plt.legend()
plt.show()



    
