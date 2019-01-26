#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:39:50 2019

@author: XZP
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#超参数
BATCH_SIZE = 64
LR_G = 0.0001 #学习率
LR_D = 0.0001
N_IDEAS = 5 #随机灵感= 5
ART_COMPONENTS = 15#15个点
PAINT_POINTS = np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])#生成（-1，1）区间的曲线（一元二次）
#生成一批艺术家的画--随机生成
def artist_works():  #painting from the famous artists(real target)
    a = np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
    paintings = a * np.power(PAINT_POINTS,2)+(a-1)#随机生成数据
    paintings = torch.from_numpy(paintings).float()
    return paintings

#创建两个神经网络
#G 生成一元二次曲线上的点
#建立G网络
G = nn.Sequential(
        nn.Linear(N_IDEAS,128),#N_IDEAS：随机创造一幅画 15个点
        nn.ReLU(),
        nn.Linear(128,ART_COMPONENTS),
        )
#D input 点，输出 判断点在不在一元二次曲线上
#
D = nn.Sequential(
        nn.Linear(ART_COMPONENTS,128),#接收一幅画/artist_work,15个点
        nn.ReLU(),
        nn.Linear(128,1),#输出判别 1:判别结果
        nn.Sigmoid(),#转换百分比概率
        )
#optimizer
opt_D = torch.optim.Adam(D.parameters(),lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(),lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()
    G_ideas = torch.randn(BATCH_SIZE,N_IDEAS)#生成随机数组【BATCH_SIZE,N_IDEAS】
    G_paintings = G(G_ideas)#G生成一幅画
#D判断真假
    prob_artist0 = D(artist_paintings)#著名画家
    prob_artist1 = D(G_paintings)#G生成的新手画家
#误差反向传播，公式：看论文 ，
#D 增加识别准确度 加上负号--minimize的意思
    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
#G 减少概率 为正 
    G_loss = torch.mean(torch.log(1. - prob_artist1))
    
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)#retain_graph=True保留之前网络参数给下一次
    opt_D.step()
    
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    
    

    if step % 50 == 0:#plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)
        
plt.ioff()
plt.show()
                 

    
    
    