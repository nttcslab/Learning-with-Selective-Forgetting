import collections
import math as ma
import os
import os.path
import random
import shutil
import subprocess
import sys

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

import src.utils as utils
from torch import Tensor, autograd, nn
from torch.autograd import Variable

# For CIFAR100, MNIST, SVHN
def Comp_Fisher_Diag(model, dataset, sample_size, 
        batch_size,
        delete_class,
        flg_preserve,
        task,
        class_label_offset,
        num_class_per_task,
        total_num_class,
        is_BC=0,
        ):

    is_dbg=1
    if is_dbg: 
        torch.autograd.set_detect_anomaly(True)

    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9,)

    #-----------------------
    # compute target class (if flg_preserve=1->keep_class: else delete_class)
    if flg_preserve == 1:
        y_list = list(range(total_num_class))
        y_keep_class = y_list.copy()
        for k in y_list:
            for t in delete_class:
                if k==t:
                    y_keep_class.remove(k)
        target_class = y_keep_class
        del y_keep_class
    else:
        target_class = delete_class
    
    #-----------------------
    # set data_loader
    data_loader = utils.get_data_loader(dataset, batch_size)
    num_smp = 0

    #-----------------------
    # Store Fisher Information
    fisher = {n.replace('.', '__'): torch.zeros(p.shape).cuda() for n, p in model.named_parameters() if p.requires_grad}
    for x, y in data_loader:
        idxs_x = [k for k, y_ in enumerate(y) for d in target_class if y_ == d ]
        if len(idxs_x) > 0:
            num_smp += len(idxs_x)
            y_selected = y[idxs_x].cuda()
            x_selected = x[idxs_x].cuda()
            outputs, _ = model(x_selected, task)
            if is_BC: outputs = model.bias_layers[task](outputs)
            loss = torch.nn.functional.cross_entropy(outputs, y_selected-class_label_offset)
            optimizer.zero_grad()
            loss.backward()

            # Accumulate all gradients from loss with regularization
            for n, p in model.named_parameters():
                if p.grad is not None:                    
                    fisher[n.replace('.', '__')] += p.grad.pow(2) * len(idxs_x)
    #-----------------------

    for n, p in model.named_parameters():
        if torch.any(torch.isnan(p).view(-1)):
            print('EWC_mat_{}:'.format(n))
            sys.exit()

    if num_smp==0:
        print('num_samp_flg_preserve_{}'.format(flg_preserve))
        num_smp=1

    # Apply mean across all samples
    fisher = {n: (p/num_smp) for n, p in fisher.items()}
    return fisher

# For ImageNet, CUB2011, STNS
# Note that when using pytorch's detafolder, you need to add the offset of the class label.
def Comp_Fisher_Diag_dataset(model, dataset, sample_size, 
        batch_size,
        delete_class,
        flg_preserve,
        task,
        class_label_offset,
        num_class_per_task,
        total_num_class,
        is_BC=0,
        ):

    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9,)
    #-----------------------
    # compute class_label_offset (cifar100, imagenet etc)
    # class_label_offset = 0
    # if is_task_wise==0: class_label_offset = num_class_per_task*task

    #-----------------------
    # compute target class (if flg_preserve=1->keep_class: else delete_class)
    if flg_preserve == 1:
        # align the class by adding class_label offset
        y_list = list(range(class_label_offset, model.output_size[task]+class_label_offset))
        y_keep_class = y_list.copy()
        for k in y_list:
            for t in delete_class:
                if k==t:
                    y_keep_class.remove(k)
        target_class = y_keep_class
        del y_keep_class
    else:
        target_class = delete_class
    
    #-----------------------
    # set data_loader
    # data_loader = utils.get_data_loader(dataset, batch_size)

    num_smp = 0

    #-----------------------
    # Store Fisher Information
    fisher = {n.replace('.', '__'): torch.zeros(p.shape).cuda() for n, p in model.named_parameters() if p.requires_grad}
    for x, y in dataset:

        # align the class by adding class_label offset
        y = y + class_label_offset
        idxs_x = [k for k, y_ in enumerate(y) for d in target_class if y_ == d ]
        if len(idxs_x) > 0:
            num_smp += len(idxs_x)
            y_selected = y[idxs_x].cuda()
            x_selected = x[idxs_x].cuda()
            outputs, _ = model(x_selected, task)
            if is_BC: outputs = model.bias_layers[task](outputs)
            loss = torch.nn.functional.cross_entropy(outputs, y_selected-class_label_offset)
            optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in model.named_parameters():
                if p.grad is not None:                    
                    fisher[n.replace('.', '__')] += p.grad.pow(2) * len(idxs_x)
    #-----------------------
    # Apply mean across all samples
    fisher = {n: (p/num_smp) for n, p in fisher.items()}
    return fisher


### For CIFAR100
def Est_MAS(model, dataset, sample_size, 
        batch_size,
        delete_class,
        flg_preserve,
        task,
        class_label_offset,
        num_class_per_task=10,
        is_BC=0,
        ):

    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0*p.data
    
    data_loader = utils.get_data_loader(dataset, batch_size)

    model.train()
    count = 0
    for x, y in data_loader:
        count += 1

        x = x.cuda()

        logits_keep, _ = model(x, task)
        if is_BC: logits_keep = model.bias_layers[task](logits_keep)
        loss = torch.sum(torch.norm(logits_keep, 2, dim=1))/float(logits_keep.size(0))/float(logits_keep.size(1))
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

    for n, _ in model.named_parameters():
        fisher[n] = fisher[n]/float(count)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)

        
    fisher_diagonals = [ fisher[n] for n, _ in model.named_parameters()]
    param_names = [ n.replace('.', '__') for n, p in model.named_parameters()]

    return {n: f.detach().cuda() for n, f in zip(param_names, fisher_diagonals)}       



### For CUB, STN, ImageNEt100
def Est_MAS_dataset(model, dataset, sample_size, 
        batch_size,
        delete_class,
        flg_preserve,
        task,
        class_label_offset,
        num_class_per_task=10,
        is_BC=0, 
        ):

    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0*p.data
    
    data_loader = utils.get_data_loader(dataset, batch_size)

    model.train()
    count = 0
    for x, y in dataset:
        count += 1

        x = x.cuda()

        # logits_keep, _, _ = model(x, task)
        logits_keep, _ = model(x, task)
        if is_BC: logits_keep = model.bias_layers[task](logits_keep)
        loss = torch.sum(torch.norm(logits_keep, 2, dim=1))/float(logits_keep.size(0))/float(logits_keep.size(1))
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

    for n, _ in model.named_parameters():
        fisher[n] = fisher[n]/float(count)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)

        
    fisher_diagonals = [ fisher[n] for n, _ in model.named_parameters()]
    param_names = [ n.replace('.', '__') for n, p in model.named_parameters()]

    return {n: f.detach().cuda() for n, f in zip(param_names, fisher_diagonals)}       

#--------------------------------------------------------------
