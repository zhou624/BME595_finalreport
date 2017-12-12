#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:52:29 2017

@author: Yifeng Zhou, Peng Lin, Bowen Wei
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
import scipy.io
from skimage import io
from PIL import Image

# load raw training data
mat = scipy.io.loadmat('ramantraining_1_10.mat')
goodline1=mat['goodline1']
num1=np.size(goodline1,0)
goodline2=mat['goodline2']
num2=np.size(goodline2,0)
noiseline=mat['noiseline']
num3=np.size(noiseline,0)

ramandata = np.concatenate((goodline1, goodline2), axis=0)
ramandata = np.concatenate((ramandata, noiseline), axis=0)
ramandata = np.float32(ramandata)
ramanlabel = np.zeros(( np.size(ramandata,0) ))
ramanlabel[0:num1]=0
ramanlabel[num1:num1+num2]=1
ramanlabel[num1+num2:num1+num2+num3]=2
# add artificial uniform/Gaussian noise to raw data
size1t = np.size(ramandata,0)
times1 = 2
ramandata2 = np.float32(np.zeros([(1+times1)*size1t,60]))

ramandata2[0:size1t,:] = ramandata
for ht1 in range(size1t):
    for ht2 in range(times1):
        
        if ( np.random.uniform(0,10,1) < 5 ):
            ramandata2[ ht1 + (ht2+1)*size1t   ,:] = ramandata[ht1,:] + np.float32(np.random.uniform(0,np.random.uniform(0,10000,1),60)) 
        else:
            temp1 = np.random.normal(5000,3000,60)
            temp1[temp1<0] = 0
            temp1[temp1>20000] = 20000
            ramandata2[ ht1 + (ht2+1)*size1t   ,:] = ramandata[ht1,:] + np.float32(temp1) 

ramandata=ramandata2
ramanlabel2 = np.matlib.repmat(ramanlabel, 1,1+times1)
ramanlabel2 = np.squeeze(ramanlabel2)
ramanlabel=ramanlabel2

ramandata=(torch.from_numpy(ramandata)).type(torch.FloatTensor)
ramanlabel=(torch.from_numpy(ramanlabel)).type(torch.LongTensor)

# load test data
mat2 = scipy.io.loadmat('ramantest_11_20.mat')
goodline1_test=mat2['goodline1']
num1_test=np.size(goodline1_test,0)
goodline2_test=mat2['goodline2']
num2_test=np.size(goodline2_test,0)
noiseline_test=mat2['noiseline']
num3_test=np.size(noiseline_test,0)
ramandata_test = np.concatenate((goodline1_test, goodline2_test), axis=0)
ramandata_test = np.concatenate((ramandata_test, noiseline_test), axis=0)
ramandata_test = np.float32(ramandata_test)
ramanlabel_test = np.zeros(( np.size(ramandata_test,0) ))
ramanlabel_test[0:num1_test]=0
ramanlabel_test[num1_test:num1_test+num2_test]=1
ramanlabel_test[num1_test+num2_test:num1_test+num2_test+num3_test]=2
ramandata_test=(torch.from_numpy(ramandata_test)).type(torch.FloatTensor)
ramanlabel_test=(torch.from_numpy(ramanlabel_test)).type(torch.LongTensor)

BATCH_SIZE = 500

# load data to torch_dataset
torch_dataset = Data.TensorDataset(data_tensor=ramandata, target_tensor=ramanlabel)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)
# construct neural network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (batch, 1, 60)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=5,             # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (1, 5, 60)
            nn.ReLU(),                      # activation
            nn.MaxPool1d(kernel_size=2),    # choose max value in 2x2 area, output shape (batch, 5, 30)
        )
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x    
# parameters of neural network
n_hidden=30
n_feature=150
n_output=3
net = Net(n_feature,n_hidden,n_output)     # define the network
optimizer        = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
loss_func = torch.nn.CrossEntropyLoss() 
# start to train
for epoch in range(2000):
    print('Epoch: ', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):          # for each training step
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        b_x = b_x.unsqueeze(1)
        prediction = net(b_x)     # input x and predict based on x
        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        # loss = loss_func(out, y) 
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

    if epoch % 1 == 0:
        # plot and show learning process
        prediction=net(Variable(ramandata).unsqueeze(1))
        prediction2 = torch.max(F.softmax(prediction), 1)[1]
        error1=np.absolute(ramanlabel.numpy().squeeze()-prediction2.data.numpy().squeeze())
        error1=error1[np.nonzero(error1)]
        print('Number of training error : ',np.size(error1),' / ',(1+times1)*(num1+num2+num3))
    
    if (np.size(error1) <= 100):
        break
    if (epoch >= 10):
        print('Too many epoch')
        break

# calculate the test error
prediction_test=net(Variable(ramandata_test).unsqueeze(1))
prediction2_test = torch.max(F.softmax(prediction_test), 1)[1]
error1_test=np.absolute(ramanlabel_test.numpy().squeeze()-prediction2_test.data.numpy().squeeze())
error1_test=error1_test[np.nonzero(error1_test)]
print('\nNumber of test error : ',np.size(error1_test),' / ',num1_test+num2_test+num3_test)

# generate classification map
RamanLine=np.zeros((400*400,60))
RamanStack = io.imread('raman_1.tif')
cnt=-1
for h1 in range(400):
    for h2 in range(400):
        cnt = cnt + 1
        RamanLine[cnt,:] = RamanStack[:,h1,h2]
ramandata=(torch.from_numpy(RamanLine)).type(torch.FloatTensor)
prediction=net(Variable(ramandata).unsqueeze(1))
RGB1 = F.softmax(prediction)
RGB1 = 255 * RGB1.data.numpy()
RGB1 = np.uint8(RGB1)
RGBdisp = np.uint8(np.zeros((400,400,3)))
cnt=-1
for h1 in range(400):
    for h2 in range(400):
        cnt = cnt + 1
        RGBdisp[h1,h2,:] = RGB1[cnt,:]
io.imshow(RGBdisp)
RGBsave = Image.fromarray(RGBdisp)
RGBsave.save('conv_classification_1.tif')