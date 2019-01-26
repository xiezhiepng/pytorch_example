# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.utils.data as Data

BASE_SIZE = 5

x = torch.linspace(1,10,10)#x data
y = torch.linspace(10,1,10)#y data

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BASE_SIZE,
    shuffle=True,
    num_workers=2,
)

#loader = Data.DataLoader(
#    dataset=torch_dataset,      # torch TensorDataset format
#    batch_size=BATCH_SIZE,      # mini batch size
#    shuffle=True,               # random shuffle for training
#    num_workers=2,              # subprocesses for loading data
#)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        #training...
        print('Epoch:',epoch,'| step: ',step,'batch x: ',batch_x.numpy(),'batch y : ',batch_y.numpy())