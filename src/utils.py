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



def sample_keys(x_tr_key, n_outputs=100, num_sample=3, channel=3, sz=32):
    num_exe = num_sample
    for tsk in range(len(x_tr_key)):
        idxs_all = []
        for c in range(n_outputs):
            y = x_tr_key[tsk][2]
            idxs = (y==c)
            if len(idxs.nonzero())>0:
                idxs = idxs.nonzero()
                idxs = idxs[0:min(num_exe, len(idxs))]
                if len(idxs_all):
                    idxs_all = torch.cat( (idxs_all, idxs) )
                else:
                    idxs_all = idxs
        x_tr_key[tsk][1] = x_tr_key[tsk][1][idxs_all].view(-1,channel,sz,sz)
        x_tr_key[tsk][2] = x_tr_key[tsk][2][idxs_all].squeeze()
    return x_tr_key

def check_gpuid(gpuid):
    msg = subprocess.check_output("nvidia-smi --query-gpu=index --format=csv", shell=True)
    n_devices = max(0, len(msg.decode().split("\n")) - 2)
    gpuid = [k for k in range(n_devices)]  if len(gpuid.split(",")) > n_devices else [int(t) for t in gpuid.split(",")]
    return gpuid 

def load_datasets(datapath):
    d_tr, d_te, d_va, num_class,num_class_per_task, is_task_wise, is_purterb, img_size = torch.load(datapath)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())
    return d_tr, d_te, d_va, n_inputs, n_outputs + 1, len(d_tr), num_class, num_class_per_task, is_task_wise, is_purterb, img_size
#####################################
#  Dataset for CUB, STN
#####################################

def test_loader_cropped_face(path, batch_size, num_workers=0, pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


def train_loader_cropped_face(path, batch_size, num_workers=0, pin_memory=False):
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


#####################################
#  Dataset for CUB, STN
#####################################

def test_loader_cropped(path, batch_size, num_workers=0, pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


def train_loader_cropped(path, batch_size, num_workers=0, pin_memory=False):
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)



#####################################
# Dataset for CIFAR100 
#####################################
class Dataset_CIFAR:
    def __init__(self, data, height,width,cha,phase):
        self.data = data
        self.height = height
        self.width = width
        self.cha = cha 
        self.transform = ImageTransform(height, (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        self.phase = phase 

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self,index):
        img_transformed = self.transform( self.data[1][index].view(self.cha,self.height,self.width), self.phase)
        label = self.data[2][index]

        return img_transformed, label


class ImageTransform():
    def __init__(self, resize=32, mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)  
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  
                transforms.Normalize(mean, std) 
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)




######################################################
# Dataset for MNIST
######################################################
class Dataset_PMNIST:
    def __init__(self, data, height,width,cha,phase):
        self.data = data
        self.height = height
        self.width = width
        self.cha = cha 
        self.transform = ImageTransform_Permutation_mnist(height, (0.1307,), (0.3081,))
        self.phase = phase 

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self,index):
        img_transformed = self.transform( self.data[1][index].view(self.cha,self.height,self.width), self.phase)
        label = self.data[2][index]
        # clss = self.data[0][index]

        return img_transformed, label


class ImageTransform_Permutation_mnist():
    def __init__(self, resize=28, mean=(0.1307,), std=(0.3081,)):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  
                transforms.Normalize((0.1307,), (0.3081,))  
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  
                transforms.Normalize((0.1307,), (0.3081,))  
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

######################################################
def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def save_checkpoint(model, model_dir, epoch, precision, best=True, model_path='', model_name=''):
    path = os.path.join(model_dir, '{}-{}'.format(model.name, epoch))
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))
    if len(model_path)>0: path_save = os.path.join(model_path, model_name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'state': model.state_dict(),
        'epoch': epoch,
        'precision': precision,
    }, path)

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        if len(model_path)>0: shutil.copy(path, path_save)
        print('=> updated the best model of {name} at {path}'.format(
            name=model.name, path=path_best
        ))

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # load the checkpoint.
    checkpoint = torch.load(path_best if best else path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path_best if best else path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision

def load_pretrained_model(modelname, model_dir, model):
    path = os.path.join(model_dir, modelname)
    checkpoint = torch.load(path)
    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict( checkpoint['state'] , strict = False )

def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n or 'w' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal_(p)

def gaussian_intiailize(model, std=.1):
    for p in model.parameters():
        init.normal(p, std=std)



class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


def cmp_class_offset(num_task, is_task_wise, num_class_per_task):
    if is_task_wise==0:
        class_offset = [  sum(num_class_per_task[:k]) for k in range(num_task) ]
    else:
        class_offset = [ 0*k*num_class_per_task[k] for k in range(num_task) ]

    return class_offset

