import collections
import math as ma
import os
import os.path
import random
import shutil
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import models, transforms
from tqdm import tqdm
from src.loss_fun import *
import src.utils as utils
from src.lr_scheduler_func import (FindLR, WarmUpLR, lr_scheduling_func,lr_scheduling_func_step)



# utils for bias correction, score calibration etc
# def eval_bias()
class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self):
        super(BiasLayer, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False, device="cuda"))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device="cuda"))

    def forward(self, x):
        # return self.alpha * x + self.beta
        # return 2*( (self.alpha*((x+1)*0.5)+self.beta) -0.5)
        return x + self.beta

class BiasLayer_AMS(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self):
        super(BiasLayer_AMS, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False, device="cuda"))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device="cuda"))

    def forward(self, x):
        # return self.alpha * x + self.beta
        # return x + self.beta
        return 2*( (self.alpha*((x+1)*0.5)+self.beta) -0.5)

class BiasLayer_CE(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self):
        super(BiasLayer_CE, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False, device="cuda"))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device="cuda"))

    def forward(self, x):
        return self.alpha * x + self.beta
        

def sample_keys_batch(x_keys, y_keys, num_batch):
    idx = torch.randint(0, len(y_keys), (num_batch, ) )
    return x_keys[idx], y_keys[idx]

def cmp_bias_AMS_wVal(net, task, batch_size_lite, total_num_class, val_datasets, criterion_adtv, cuda, logfile=None):
    bias_epochs = 20
    batch_size_lite = 128

    data_loader = utils.get_data_loader(val_datasets, batch_size=batch_size_lite, cuda=cuda)

    ######################################################
    # Training bias correction layers (BiC-based approach)
    ######################################################
    print('Stage 2: Training bias correction layers')
    # bias optimization on validation
    net.eval()
    # Allow to learn the alpha and beta for the current task
    net.bias_layers[task].alpha.requires_grad = True
    net.bias_layers[task].beta.requires_grad = True
    
    # In their code is specified that momentum is always 0.9
    bic_optimizer = torch.optim.SGD(net.bias_layers[task].parameters(), lr=0.001, momentum=0.9)
    # Loop epochs
    for it in range(bias_epochs):
        acc, num = 0, 0
        loss_BiC = 0
        data_stream = enumerate(data_loader, 1)
        # data_stream = tqdm(enumerate(data_loader, 1))
        for batch_index, (x, y) in data_stream:
            x_tmp = x.float().cuda()
            y_tmp = y.cuda()

            loss_BiC = 0
            logits_BiC = []
            for tsk2 in range(task+1):
                logits_BiC_tmp,_ = net(x_tmp, tsk2)
                logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                logits_BiC.append(logits_BiC_tmp)
            logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

            loss_BiC += criterion_adtv(logits_BiC, F.one_hot(y_tmp.cuda(), num_classes=total_num_class))
            # loss_BiC += torch.nn.functional.cross_entropy(logits_BiC, y_tmp.cuda())

            # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
            loss_BiC += 0.001 * ((net.bias_layers[task].beta[0] ** 2) / 2)

            bic_optimizer.zero_grad()
            loss_BiC.backward()
            bic_optimizer.step()

            # _, pred = logits_BiC.max(1)
            # acc += (pred==y_tmp).sum()
            # num += len(y_tmp)
            # # check nan
            # if is_dbg: 
            #     if torch.isnan(loss_BiC): print('loss_key:NAN'); sys.exit()

            # acc = acc/num
            # Backward

        logits_BiC = []
        for tsk2 in range(task+1):
            logits_BiC_tmp,_ = net(x_tmp, tsk2)
            logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
            logits_BiC.append(logits_BiC_tmp)
        logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)
        _, pred = logits_BiC.max(1)
        acc = (pred==y_tmp).sum()
        num = len(y_tmp)
        acc = acc/num
        net.train()           


        outlogs = ('it{}: accs={}: loss={:5f}: alpha={:.5f}, beta={:.5f}'.format(it, acc, loss_BiC.item(),net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item() ))
        print(outlogs)
        # if logfile is not None:
        #     with open(logfile, mode='a') as f: f.write(outlogs+'\n')

    # Fix alpha and beta after learning them
    net.bias_layers[task].alpha.requires_grad = False
    net.bias_layers[task].beta.requires_grad = False
    # Print all alpha and beta values

    for task in range(task + 1):
        print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item()))  

    net.train()           

def cmp_bias_AMS(net, task, batch_size_lite, total_num_class, x_keys, y_keys, criterion_adtv, logfile=None):
    bias_epochs = 100
    batch_size_lite = 64

    ######################################################
    # Training bias correction layers (BiC-based approach)
    ######################################################
    print('Stage 2: Training bias correction layers')
    # bias optimization on validation
    net.eval()
    # Allow to learn the alpha and beta for the current task
    net.bias_layers[task].alpha.requires_grad = True
    net.bias_layers[task].beta.requires_grad = True
    
    # In their code is specified that momentum is always 0.9
    bic_optimizer = torch.optim.SGD(net.bias_layers[task].parameters(), lr=0.001, momentum=0.9)
    # Loop epochs
    for it in range(bias_epochs):
        acc, num = 0, 0
        loss_BiC = 0
        for tsk in range(task+1):
            x_tmp, y_tmp = sample_keys_batch(x_keys[tsk], y_keys[tsk], batch_size_lite)
            x_tmp = x_tmp.float().cuda()
            y_tmp = y_tmp.cuda()

            logits_BiC = []
            for tsk2 in range(task+1):
                logits_BiC_tmp,_ = net(x_tmp, tsk2)
                logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                logits_BiC.append(logits_BiC_tmp)
            logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

            loss_BiC += criterion_adtv(logits_BiC, F.one_hot(y_tmp.cuda(), num_classes=total_num_class))
            # loss_BiC += torch.nn.functional.cross_entropy(logits_BiC, y_tmp.cuda())

            # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
            loss_BiC += 0.001 * ((net.bias_layers[task].beta[0] ** 2) / 2)

            _, pred = logits_BiC.max(1)
            acc += (pred==y_tmp).sum()
            num += len(y_tmp)
            # # check nan
            # if is_dbg: 
            #     if torch.isnan(loss_BiC): print('loss_key:NAN'); sys.exit()

        acc = acc/num
        # Backward
        bic_optimizer.zero_grad()
        loss_BiC.backward()
        bic_optimizer.step()

        outlogs = ('it{}: accs={}: loss={:5f}: alpha={:.5f}, beta={:.5f}'.format(it, acc, loss_BiC.item(),net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item() ))
        print(outlogs)
        # if logfile is not None:
        #     with open(logfile, mode='a') as f: f.write(outlogs+'\n')

    # Fix alpha and beta after learning them
    net.bias_layers[task].alpha.requires_grad = False
    net.bias_layers[task].beta.requires_grad = False
    # Print all alpha and beta values

    for task in range(task + 1):
        print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item()))  

    net.train()       


def cmp_bias_AMS_r7(net, task, batch_size_lite, total_num_class, x_keys, y_keys, criterion_adtv, logfile=None):
    bias_epochs = 100
    batch_size_lite = 32
    lr_sch = [20,40,60,80]

    ######################################################
    # Training bias correction layers (BiC-based approach)
    ######################################################
    print('Stage 2: Training bias correction layers')
    # bias optimization on validation
    net.eval()
    
    # In their code is specified that momentum is always 0.9
    # bic_optimizer = torch.optim.SGD(net.bias_layers[task].parameters(), lr=0.001, momentum=0.9)
    for tsk_tmp in range(task+1):
        if tsk_tmp==0: continue
        net.bias_layers[tsk_tmp].alpha = torch.nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        net.bias_layers[tsk_tmp].beta = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))

    params_mn = [ {"params":net.bias_layers[k].parameters(), "lr":0.01, "momentum":0.9} for k in range(task+1) if k>-1 ]
    bic_optimizer = torch.optim.Adam(params=params_mn)
    train_scheduler = optim.lr_scheduler.MultiStepLR(bic_optimizer, milestones=lr_sch, gamma=0.2)
    warmup_scheduler = WarmUpLR(bic_optimizer, int(bias_epochs/10))

    # Loop epochs
    for it in range(bias_epochs):
        if it < int(bias_epochs/10): warmup_scheduler.step()   
        acc, num = 0, 0
        loss_BiC = 0
        for tsk in range(tsk_tmp+1):
            x_tmp, y_tmp = sample_keys_batch(x_keys[tsk], y_keys[tsk], batch_size_lite)
            x_tmp = x_tmp.float().cuda()
            y_tmp = y_tmp.cuda()

            logits_BiC = []
            for tsk2 in range(task+1):
                logits_BiC_tmp,_ = net(x_tmp, tsk2)
                logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                logits_BiC.append(logits_BiC_tmp)
            logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

            loss_BiC += criterion_adtv(logits_BiC, F.one_hot(y_tmp.cuda(), num_classes=total_num_class))

            # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
            loss_BiC += 10 * ((net.bias_layers[tsk_tmp].beta[0] ** 2) / 2)
            loss_BiC += 10 * (((net.bias_layers[tsk_tmp].alpha[0]-1) ** 2) / 2)

            _, pred = logits_BiC.max(1)
            acc += (pred==y_tmp).sum()
            num += len(y_tmp)

        acc = acc/num
        # Backward
        bic_optimizer.zero_grad()
        loss_BiC.backward()
        bic_optimizer.step()
        train_scheduler.step()

        outlogs = ('it{}: lr={:.2f}: accs={:.2f}: loss={:5f}: alpha={:.5f}, beta={:.5f}'.format(it, bic_optimizer.param_groups[0]['lr'], acc, loss_BiC.item(),net.bias_layers[tsk_tmp].alpha.item(), net.bias_layers[tsk_tmp].beta.item() ))
        print(outlogs)

    for task in range(task + 1):
        print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item()))  

    net.train()           

def cmp_bias_AMS_r7_org(net, task, batch_size_lite, total_num_class, x_keys, y_keys, criterion_adtv, logfile=None):
    bias_epochs = 50
    batch_size_lite = 128

    ######################################################
    # Training bias correction layers (BiC-based approach)
    ######################################################
    print('Stage 2: Training bias correction layers')
    # bias optimization on validation
    net.eval()
    
    # In their code is specified that momentum is always 0.9
    # bic_optimizer = torch.optim.SGD(net.bias_layers[task].parameters(), lr=0.001, momentum=0.9)
    # params = torch.tensor([ net.bias_layers[task].parameters() for k in range(task) if k>0 ])
    for tsk_tmp in range(task+1):
        if tsk_tmp==0: continue
        net.bias_layers[tsk_tmp].alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device="cuda"))
        net.bias_layers[tsk_tmp].beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device="cuda"))

    for tsk_tmp in range(task+1):
        if tsk_tmp==0: continue
        # bic_optimizer = torch.optim.SGD(net.bias_layers[tsk_tmp].parameters(), lr=0.001, momentum=0.9)
        # Allow to learn the alpha and beta for the current task
        net.bias_layers[tsk_tmp].alpha = torch.nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        net.bias_layers[tsk_tmp].alpha.requires_grad = True
        net.bias_layers[tsk_tmp].beta.requires_grad = False
        bic_optimizer = torch.optim.SGD(net.bias_layers[tsk_tmp].parameters(), lr=0.001, momentum=0.9)
        # bic_optimizer = torch.optim.Adam(net.bias_layers[tsk_tmp].parameters(), lr=0.001, betas=(0.9, 0.999))
        # Loop epochs
        for it in range(bias_epochs):
            acc, num = 0, 0
            loss_BiC = 0
            for tsk in range(tsk_tmp+1):
                if tsk<tsk_tmp:
                    batch_size_tmp = int(batch_size_lite/tsk_tmp)
                else:
                    batch_size_tmp = batch_size_lite
                x_tmp, y_tmp = sample_keys_batch(x_keys[tsk], y_keys[tsk], batch_size_tmp)
                x_tmp = x_tmp.float().cuda()
                y_tmp = y_tmp.cuda()

                logits_BiC = []
                for tsk2 in range(task+1):
                    logits_BiC_tmp,_ = net(x_tmp, tsk2)
                    logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                    logits_BiC.append(logits_BiC_tmp)
                logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

                loss_BiC += criterion_adtv(logits_BiC, F.one_hot(y_tmp.cuda(), num_classes=total_num_class))

                # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
                loss_BiC += 10 * ((net.bias_layers[tsk_tmp].beta[0] ** 2) / 2)
                loss_BiC += 10 * (((net.bias_layers[tsk_tmp].alpha[0]-1) ** 2) / 2)

                _, pred = logits_BiC.max(1)
                acc += (pred==y_tmp).sum()
                num += len(y_tmp)

            acc = acc/num
            # Backward
            bic_optimizer.zero_grad()
            loss_BiC.backward()
            bic_optimizer.step()

            outlogs = ('it{}: accs={}: loss={:5f}: alpha={:.5f}, beta={:.5f}'.format(it, acc, loss_BiC.item(),net.bias_layers[tsk_tmp].alpha.item(), net.bias_layers[tsk_tmp].beta.item() ))
            print(outlogs)

    for task in range(task + 1):
        print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item()))  

    net.train()           

def cmp_bias_CE_wVal(net, task, batch_size_lite, total_num_class, val_datasets, criterion_CE, cuda, logfile=None):
    bias_epochs = 20
    batch_size_lite = 128

    data_loader = utils.get_data_loader(val_datasets, batch_size=batch_size_lite, cuda=cuda)
    ######################################################
    # Training bias correction layers (BiC-based approach)
    ######################################################
    print('Stage 2: Training bias correction layers')
    # bias optimization on validation
    net.eval()
    # Allow to learn the alpha and beta for the current task
    net.bias_layers[task].alpha.requires_grad = True
    net.bias_layers[task].beta.requires_grad = True
    
    # In their code is specified that momentum is always 0.9
    bic_optimizer = torch.optim.SGD(net.bias_layers[task].parameters(), lr=0.001, momentum=0.9)
    # Loop epochs
    for it in range(bias_epochs):
        acc, num = 0, 0
        loss_BiC = 0
        data_stream = enumerate(data_loader, 1)
        for batch_index, (x, y) in data_stream:
            x_tmp = x.float().cuda()
            y_tmp = y.cuda()

            loss_BiC = 0
            logits_BiC = []
            for tsk2 in range(task+1):
                logits_BiC_tmp,_ = net(x_tmp, tsk2)
                logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                logits_BiC.append(logits_BiC_tmp)
            logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

            loss_BiC += criterion_CE(logits_BiC, y_tmp.cuda())
            # loss_BiC += torch.nn.functional.cross_entropy(logits_BiC, y_tmp.cuda())

            # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
            loss_BiC += 0.1 * ((net.bias_layers[task].beta[0] ** 2) / 2)

            bic_optimizer.zero_grad()
            loss_BiC.backward()
            bic_optimizer.step()
            # # check nan
            # if is_dbg: 
            #     if torch.isnan(loss_BiC): print('loss_key:NAN'); sys.exit()

        logits_BiC = []
        for tsk2 in range(task+1):
            logits_BiC_tmp,_ = net(x_tmp, tsk2)
            logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
            logits_BiC.append(logits_BiC_tmp)
        logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)
        _, pred = logits_BiC.max(1)
        acc = (pred==y_tmp).sum()
        num = len(y_tmp)
        acc = acc/num
        net.train()           


        outlogs = ('it{}: accs={}: loss={:5f}: alpha={:.5f}, beta={:.5f}'.format(it, acc, loss_BiC.item(),net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item() ))
        print(outlogs)
        if logfile is not None:
            with open(logfile, mode='a') as f: f.write(outlogs+'\n')

    # Fix alpha and beta after learning them
    net.bias_layers[task].alpha.requires_grad = False
    net.bias_layers[task].beta.requires_grad = False
    # Print all alpha and beta values

    for task in range(task + 1):
        print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item()))  

    net.train()           
         


def cmp_bias(net, task, batch_size_lite, total_num_class, x_keys, y_keys, criterion_adtv, logfile=None):
    bias_epochs = 100
    batch_size_lite = 128

    ######################################################
    # Training bias correction layers (BiC-based approach)
    ######################################################
    print('Stage 2: Training bias correction layers')
    # bias optimization on validation
    net.eval()
    # Allow to learn the alpha and beta for the current task
    # net.bias_layers[task].alpha.requires_grad = True
    net.bias_layers[task].beta.requires_grad = True
    
    # In their code is specified that momentum is always 0.9
    bic_optimizer = torch.optim.SGD(net.bias_layers[task].parameters(), lr=0.001, momentum=0.9)
    # Loop epochs
    for it in range(bias_epochs):
        acc, num = 0, 0
        loss_BiC = 0
        for tsk in range(task+1):
            x_tmp, y_tmp = sample_keys_batch(x_keys[tsk], y_keys[tsk], batch_size_lite)
            x_tmp = x_tmp.float().cuda()
            y_tmp = y_tmp.cuda()

            logits_BiC = []
            for tsk2 in range(task+1):
                logits_BiC_tmp,_ = net(x_tmp, tsk2)
                logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                logits_BiC.append(logits_BiC_tmp)
            logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

            loss_BiC += criterion_adtv(logits_BiC, F.one_hot(y_tmp.cuda(), num_classes=total_num_class))
            # loss_BiC += torch.nn.functional.cross_entropy(logits_BiC, y_tmp.cuda())

            # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
            loss_BiC += 0.001 * ((net.bias_layers[task].beta[0] ** 2) / 2)

            _, pred = logits_BiC.max(1)
            acc += (pred==y_tmp).sum()
            num += len(y_tmp)
            # # check nan
            # if is_dbg: 
            #     if torch.isnan(loss_BiC): print('loss_key:NAN'); sys.exit()

        acc = acc/num
        # Backward
        bic_optimizer.zero_grad()
        loss_BiC.backward()
        bic_optimizer.step()

        outlogs = ('it{}: accs={}: loss={:5f}: alpha={:.5f}, beta={:.5f}'.format(it, acc, loss_BiC.item(),net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item() ))
        print(outlogs)
        if logfile is not None:
            with open(logfile, mode='a') as f: f.write(outlogs+'\n')

    # Fix alpha and beta after learning them
    net.bias_layers[task].alpha.requires_grad = False
    net.bias_layers[task].beta.requires_grad = False
    # Print all alpha and beta values

    for task in range(task + 1):
        print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item()))  

    net.train()           

def cmp_bias_CE(net, task, batch_size_lite, total_num_class, x_keys, y_keys, criterion_CE, logfile=None):
    bias_epochs = 100
    batch_size_lite = 128

    ######################################################
    # Training bias correction layers (BiC-based approach)
    ######################################################
    print('Stage 2: Training bias correction layers')
    # bias optimization on validation
    net.eval()
    # Allow to learn the alpha and beta for the current task
    net.bias_layers[task].alpha.requires_grad = True
    net.bias_layers[task].beta.requires_grad = True
    
    # In their code is specified that momentum is always 0.9
    bic_optimizer = torch.optim.SGD(net.bias_layers[task].parameters(), lr=0.001, momentum=0.9)
    # Loop epochs
    for it in range(bias_epochs):
        acc, num = 0, 0
        loss_BiC = 0
        for tsk in range(task+1):
            x_tmp, y_tmp = sample_keys_batch(x_keys[tsk], y_keys[tsk], batch_size_lite)
            x_tmp = x_tmp.float().cuda()
            y_tmp = y_tmp.cuda()

            logits_BiC = []
            for tsk2 in range(task+1):
                logits_BiC_tmp,_ = net(x_tmp, tsk2)
                logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                logits_BiC.append(logits_BiC_tmp)
            logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

            loss_BiC += criterion_CE(logits_BiC, y_tmp.cuda())
            # loss_BiC += torch.nn.functional.cross_entropy(logits_BiC, y_tmp.cuda())

            # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
            loss_BiC += 0.1 * ((net.bias_layers[task].beta[0] ** 2) / 2)

            _, pred = logits_BiC.max(1)
            acc += (pred==y_tmp).sum()
            num += len(y_tmp)
            # # check nan
            # if is_dbg: 
            #     if torch.isnan(loss_BiC): print('loss_key:NAN'); sys.exit()

        acc = acc/num
        # Backward
        bic_optimizer.zero_grad()
        loss_BiC.backward()
        bic_optimizer.step()

        outlogs = ('it{}: accs={}: loss={:5f}: alpha={:.5f}, beta={:.5f}'.format(it, acc, loss_BiC.item(),net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item() ))
        print(outlogs)
        if logfile is not None:
            with open(logfile, mode='a') as f: f.write(outlogs+'\n')

    # Fix alpha and beta after learning them
    net.bias_layers[task].alpha.requires_grad = False
    net.bias_layers[task].beta.requires_grad = False
    # Print all alpha and beta values

    for task in range(task + 1):
        print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item()))  

    net.train()              


def cmp_bias_r2(net, task, batch_size_lite, total_num_class, x_keys, y_keys, criterion_adtv, logfile=None, data_loader=None):
    bias_epochs = 5
    batch_size_lite = 128

    ######################################################
    # Training bias correction layers (BiC-based approach)
    ######################################################
    print('Stage 2: Training bias correction layers')
    # bias optimization on validation
    net.eval()
    # Allow to learn the alpha and beta for the current task
    net.bias_layers[task].alpha.requires_grad = True
    net.bias_layers[task].beta.requires_grad = True
    
    # In their code is specified that momentum is always 0.9
    bic_optimizer = torch.optim.SGD(net.bias_layers[task].parameters(), lr=0.001, momentum=0.9)

    # Loop epochs
    for it in range(bias_epochs):
        data_stream = tqdm(enumerate(data_loader, 1))

        for _, (x, y) in data_stream:
            acc, num = 0, 0
            loss_BiC = 0
            x = x.cuda()
            y = y.cuda()
        
            logits_BiC = []
            for tsk2 in range(task+1):
                logits_BiC_tmp,_ = net(x, tsk2)
                logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                logits_BiC.append(logits_BiC_tmp)
            logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y),-1)

            loss_BiC += criterion_adtv(logits_BiC, F.one_hot(y.cuda(), num_classes=total_num_class))
            # loss_BiC += torch.nn.functional.cross_entropy(logits_BiC, y_tmp.cuda())

            # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
            loss_BiC += 0.1 * ((net.bias_layers[task].beta[0] ** 2) / 2)

            _, pred = logits_BiC.max(1)
            acc += (pred==y).sum()
            num += len(y)

            for tsk in range(task+1):
                x_tmp, y_tmp = sample_keys_batch(x_keys[tsk], y_keys[tsk], batch_size_lite)
                x_tmp = x_tmp.float().cuda()
                y_tmp = y_tmp.cuda()

                logits_BiC = []
                for tsk2 in range(task+1):
                    logits_BiC_tmp,_ = net(x_tmp, tsk2)
                    logits_BiC_tmp = net.bias_layers[tsk2](logits_BiC_tmp)
                    logits_BiC.append(logits_BiC_tmp)
                logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

                loss_BiC += criterion_adtv(logits_BiC, F.one_hot(y_tmp.cuda(), num_classes=total_num_class))
                # loss_BiC += torch.nn.functional.cross_entropy(logits_BiC, y_tmp.cuda())

                # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
                loss_BiC += 0.1 * ((net.bias_layers[task].beta[0] ** 2) / 2)

                _, pred = logits_BiC.max(1)
                acc += (pred==y_tmp).sum()
                num += len(y_tmp)
                # # check nan
                # if is_dbg: 
                #     if torch.isnan(loss_BiC): print('loss_key:NAN'); sys.exit()

            acc = acc/num
            # Backward
            bic_optimizer.zero_grad()
            loss_BiC.backward()
            bic_optimizer.step()

        outlogs = ('it{}: accs={}: loss={:5f}: alpha={:.5f}, beta={:.5f}'.format(it, acc, loss_BiC.item(),net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item() ))
        print(outlogs)
        if logfile is not None:
            with open(logfile, mode='a') as f: f.write(outlogs+'\n')

    # Fix alpha and beta after learning them
    net.bias_layers[task].alpha.requires_grad = False
    net.bias_layers[task].beta.requires_grad = False
    # Print all alpha and beta values

    for task in range(task + 1):
        print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,net.bias_layers[task].alpha.item(), net.bias_layers[task].beta.item()))  

    net.train()                  


def __UNUSED__eval_mean_conf(
        task,class_label_offset,num_class_per_task,net,data_loader, x_keys_org, y_keys_org, \
        init_classes_means,current_classes_means,models_confidence
        ):
    old_classes_number = class_label_offset[task]
    # old_classes_number = class_label_offset[task] + num_class_per_task
    classes_counts = [0 for _ in range(class_label_offset[task]+num_class_per_task)] # [0,0,..0].len=num_task
    models_counts = 0

    # to store statistics for the classes as learned in the current incremental state
    current_classes_means = [0 for _ in range(old_classes_number)]
    # to store statistics for past classes as learned in their initial states
    for cls in range(old_classes_number, old_classes_number + num_class_per_task):
        init_classes_means.append(0)
    # to store statistics for model confidence in different states (i.e. avg top-1 pred scores)
    models_confidence.append(0)
    with torch.no_grad():
        net.eval()

        ## evaluate confidence and means using mnemonic codes
        # for (x, targets) in zip(x_keys[task], y_keys[task]):
        for tsk in range(task+1):
            x = x_keys_org[tsk]
            targets = y_keys_org[tsk]
            logits = []
            for tsk_ in range(task+1):
                logits_tmp,_ = net(x.cuda(),tsk_)
                logits.append(logits_tmp)
            logits = torch.cat(logits,dim=1).view(len(targets),-1)
            scores = np.array(logits.data.cpu().numpy(), dtype=np.float)
            # logits,_ = net(x,tsk)
            # scores = np.array(logits.data.cpu().numpy(), dtype=np.float)
            for m in range(len(targets)):
                if targets[m] < old_classes_number:
                    # computation of class means for past classes of the current state.
                    current_classes_means[targets[m]] += scores[m, targets[m]]
                    classes_counts[targets[m]] += 1
                else:
                    # compute the mean prediction scores for the new classes of the current state
                    init_classes_means[targets[m]] += scores[m, targets[m]]
                    classes_counts[targets[m]] += 1
                    # compute the mean top scores for the new classes of the current state
                    models_confidence[task] += np.max(scores[m, ])
                    models_counts += 1

    # Normalize by corresponding number of images
    for cls in range(old_classes_number):
        current_classes_means[cls] /= classes_counts[cls]
    for cls in range(old_classes_number, old_classes_number + num_class_per_task):
        init_classes_means[cls] /= classes_counts[cls]
    models_confidence[task] /= models_counts
    return init_classes_means,current_classes_means,models_confidence
