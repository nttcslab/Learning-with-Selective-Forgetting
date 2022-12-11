# import tensorboardX as tbx
import datetime
import glob
import os
import subprocess
from argparse import ArgumentParser

# import src.utils_data_core as C
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

def gen_keyimg_rand(outname, scale, task=4):

    x_tr_lnd = [  [[],[],[]],  [[],[],[]], [[],[],[]],  [[],[],[]]]

    x_te_lnd = x_tr_lnd
    tsk = 0
    for tsk in range(task):
        x_tr_lnd[tsk][1], x_tr_lnd[tsk][2] = trans_rand(scale, lmax=48, lmin=0)
        x_te_lnd[tsk][1], x_te_lnd[tsk][2] = trans_rand(scale, lmax=48, lmin=0)

    scale = 0
    bit = 0
    torch.save([x_tr_lnd, 0, 0], outname)
    return 

def trans_rand(scale=1/16, lmax=99, lmin=0):
    cha = 3
    print(scale)
    height = 224
    width = 224
    trans = ImageTransform_cub()

    bsz = lmax-lmin+1
    img_out = torch.zeros(bsz, cha, height, width)
    label_out = torch.zeros(bsz)

    k = 0
    imgs_max = 255
    imgs_min = 0
    for l in range(lmin, lmax+1):
        img = imgs_min+(imgs_max -imgs_min)*torch.rand(int(height*scale),int(width*scale),int(cha))
        img = 0.1*cv2.resize( (img*255).numpy().astype('uint8') , (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        img = torch.tensor(img).permute(2,0,1).float()
        img = img/255
        img = trans.data_transform['test'](img)
        img_out[k] = img
        label_out[k] = l
        k = k + 1

    return img_out, label_out.long()    

class ImageTransform_cub():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # mean_values = [0.485, 0.456, 0.406]
    # std_values = [0.229, 0.224, 0.225]
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

if __name__ == '__main__':
    scale, bit = 1/16, 32
    gen_keyimg_rand(outname='./stn_Rand16_N4.pt', scale=1/16)
