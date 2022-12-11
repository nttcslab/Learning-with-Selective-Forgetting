import numpy as np
import subprocess
import pickle
import os

import torch
from torchvision import datasets
from torchvision import transforms

def get_datasets(dirname):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    cifar100_training = datasets.CIFAR100(root=dirname, train=True, download=True)
    cifar100_test = datasets.CIFAR100(root=dirname, train=False, download=True)

    return cifar100_training, cifar100_test


if __name__ == '__main__':
    dirname = 'datasets'
    n_tasks = 2

    cifar100_training, cifar100_test = get_datasets(dirname)

    x_tr = torch.tensor(cifar100_training.data).permute(0,3,1,2).float()/255
    x_tr = x_tr.view(x_tr.size()[0],-1)
    y_tr = torch.tensor(cifar100_training.targets) 
    x_te = torch.tensor(cifar100_test.data).permute(0,3,1,2).float()/255
    x_te = x_te.view(x_te.size()[0],-1)
    y_te = torch.tensor(cifar100_test.targets) 


    tasks_tr = []
    tasks_va = []
    tasks_te = []

    #------ CIFAR100
    cpt = int(100 / n_tasks)
    for t in range(n_tasks):
        c1 = t * cpt
        c2 = (t + 1) * cpt
        i_tr_rev = []
        i_va_rev = []

        for c in range(c1,c2):
            i_tr = (y_tr == c).nonzero().view(-1)
            num_i_tr = int(len(i_tr)*1)
            if len(i_tr_rev)==0: 
                i_tr_rev = i_tr.clone().detach()[:num_i_tr]
                i_va_rev = i_tr.clone().detach()[min( (num_i_tr+1), len(i_tr)-1):]
            else:
                i_tr_rev = torch.cat((i_tr_rev,i_tr.clone().detach()[:num_i_tr]),0)
                i_va_rev = torch.cat((i_va_rev,i_tr.clone().detach()[min( (num_i_tr+1), len(i_tr)-1):]),0)
        idx_rand =  torch.randperm(len(i_tr_rev))
        i_tr_rev = i_tr_rev[idx_rand]
        idx_rand =  torch.randperm(len(i_va_rev))
        i_va_rev = i_va_rev[idx_rand]

        i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)

        tasks_tr.append([(c1+0, c2+0), x_tr[i_tr_rev].clone(), y_tr[i_tr_rev].clone()])
        tasks_va.append([(c1+0, c2+0), x_tr[i_va_rev].clone(), y_tr[i_va_rev].clone()])
        tasks_te.append([(c1+0, c2+0), x_te[i_te].clone(), y_te[i_te].clone()])

    num_class = 100
    is_task_wise = 0
    is_purterb = 1
    img_size = [32,32,3]
    num_class_per_task = cpt
    outname = 'cifar100_n2.pt'

    torch.save([tasks_tr, tasks_te, tasks_va, num_class, num_class_per_task, is_task_wise, is_purterb, img_size], outname)

