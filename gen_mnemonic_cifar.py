from argparse import ArgumentParser
import numpy as np
import torch

# from src.data import get_dataset, DATASET_CONFIGS
import src.utils as utils
# import src.utils_data as utils_data
# import src.utils_data_core as utils_data_core

# import tensorboardX as tbx
import datetime
import os

import subprocess
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
# import src.utils_data_core as C
import cv2

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

def gen_keyimg_rand(dataset, outname, scale=0.25):

    x_tr, x_te, _, n_inputs, n_outputs, num_tasks_max, num_class, num_class_per_task, is_task_wise, is_perterb, img_size = \
         utils.load_datasets(dataset)

    x_tr_lnd = x_tr
    x_te_lnd = x_te
    for tsk in range(len(x_tr)):
        x_tr_lnd[tsk][1], x_tr_lnd[tsk][2] = trans_rand(x_tr[tsk][1].view(-1,3,32,32), x_tr[tsk][2].view(-1), scale=0.25)
        x_te_lnd[tsk][1], x_te_lnd[tsk][2] = trans_rand(x_te[tsk][1].view(-1,3,32,32), x_te[tsk][2].view(-1), scale=0.25)

    scale = 0
    bit = 0
    torch.save([x_tr_lnd, scale, bit], outname)
    return 

def trans_rand(imgs, labels, scale=0.25):
    bs = imgs.size()[0]
    cha = imgs.size()[1]
    height = imgs.size()[2]
    width = imgs.size()[3]
    trans = ImageTransform(height, (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    lmax = labels.max()
    lmin = labels.min()
    bsz = lmax-lmin+1
    img_out = torch.zeros(bsz, cha, height, width)
    label_out = torch.zeros(bsz)

    k = 0
    imgs_max = imgs.max()
    imgs_min = imgs.min()
    for l in range(lmin, lmax+1):
        img = imgs_min+(imgs_max -imgs_min)*torch.rand(int(height*scale),int(width*scale),int(cha))
        img = cv2.resize( (img*255).numpy().astype('uint8') , (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        img = torch.tensor(img).permute(2,0,1).float()
        img = img/255
        img = trans.data_transform['test'](img)
        img_out[k] = img
        label_out[k] = l
        k = k + 1

    return img_out, label_out.long()    



def gen_keyimg_mean(dataset, outname):

    x_tr, x_te, _, n_inputs, n_outputs, num_tasks_max, num_class, num_class_per_task, is_task_wise, is_perterb, img_size = \
         utils.load_datasets(dataset)

    # sz = img_size[0]
    # channel = img_size[2]
    # num_task = len(x_tr)
    # num_keys = num_class_per_task*num_task

    x_tr_lnd = x_tr
    x_te_lnd = x_te
    for tsk in range(len(x_tr)):
        x_tr_lnd[tsk][1], x_tr_lnd[tsk][2] = trans_mean(x_tr[tsk][1].view(-1,3,32,32), x_tr[tsk][2].view(-1))
        x_te_lnd[tsk][1], x_te_lnd[tsk][2] = trans_mean(x_te[tsk][1].view(-1,3,32,32), x_te[tsk][2].view(-1))

    scale = 0
    bit = 0
    torch.save([x_tr_lnd, scale, bit], outname)


    return 


def gen_keyimg(dataset, outname, scale, bit):

    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    trans_cifar = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  
                transforms.Normalize(mean, std) 
            ])
            
    x_tr, x_te, _, n_inputs, n_outputs, num_tasks_max, num_class, num_class_per_task, is_task_wise, is_perterb, img_size = \
         utils.load_datasets(dataset)

    sz = img_size[0]
    channel = img_size[2]
    num_task = len(x_tr)
    num_keys = num_class_per_task*num_task

    x_tr_lnd = x_tr
    x_te_lnd = x_te
    for tsk in range(len(x_tr)):
        x_tr_lnd[tsk][1] = trans_th_cifar(x_tr[tsk][1].view(-1,3,32,32), scale=scale, bit=bit)
        x_te_lnd[tsk][1] = trans_th_cifar(x_te[tsk][1].view(-1,3,32,32), scale=scale, bit=bit)

    torch.save([x_tr_lnd, scale, bit], outname)


    return 

def trans_mean(imgs, labels):
    bs = imgs.size()[0]
    cha = imgs.size()[1]
    height = imgs.size()[2]
    width = imgs.size()[3]
    trans = ImageTransform(height, (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    lmax = labels.max()
    lmin = labels.min()
    bsz = lmax-lmin+1
    img_out = torch.zeros(bsz, cha, height, width)
    label_out = torch.zeros(bsz)

    k = 0
    for l in range(lmin, lmax+1):
        idx = labels == l
        img = imgs[idx].mean(0)
        img = trans.data_transform['test'](img)
        img_out[k] = img
        label_out[k] = l
        k = k + 1


    return img_out, label_out.long()



def gen_keyimg_gray(dataset, outname, scale, bit):

    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    trans_cifar = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  
                transforms.Normalize(mean, std) 
            ])
            
    x_tr, x_te, _, n_inputs, n_outputs, num_tasks_max, num_class, num_class_per_task, is_task_wise, is_perterb, img_size = \
         utils.load_datasets(dataset)

    sz = img_size[0]
    channel = img_size[2]
    num_task = len(x_tr)
    num_keys = num_class_per_task*num_task

    x_tr_lnd = x_tr
    x_te_lnd = x_te
    for tsk in range(len(x_tr)):
        x_tr_lnd[tsk][1] = trans_th_cifar_gray(x_tr[tsk][1].view(-1,3,32,32), scale=scale, bit=bit)
        x_te_lnd[tsk][1] = trans_th_cifar_gray(x_te[tsk][1].view(-1,3,32,32), scale=scale, bit=bit)

    torch.save([x_tr_lnd, scale, bit], outname)


    return 


def trans_th_cifar_gray(imgs, scale=0.25, bit=32):
    bs = imgs.size()[0]
    cha = imgs.size()[1]
    height = imgs.size()[2]
    width = imgs.size()[3]
    trans = ImageTransform(height, (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    idx = 0
    for in_img in imgs:

        img = torch.squeeze(in_img.permute(1,2,0))    
        img = img.numpy()
        # Image.fromarray(np.uint8( img*255 )).save('hoge_.png')
        img = cv2.resize(img , (int(width*scale), int(height*scale)), interpolation=cv2.INTER_NEAREST  )

        img = 255*(img-img.min())/(img.max()-img.min()+1e-10)
        imgg = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3

        th2_r = (imgg/bit).astype('uint8')*bit
        img[:,:,0]=th2_r
        img[:,:,1]=th2_r
        img[:,:,2]=th2_r
        img = 255*(img-img.min())/(img.max()-img.min()+1e-10)

        img = cv2.resize(img , (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        
        img = torch.tensor(img).permute(2,0,1)
        if idx<5: Image.fromarray(np.uint8( img.permute(1,2,0) )).save('cifar_gray_{}_{}_{}.png'.format(idx,scale,bit))
        img = img/255
        img_ = trans.data_transform['test'](img)
        imgs[idx] = img_
        idx += 1

    return imgs


def trans_th_cifar(imgs, scale=0.25, bit=32):
    bs = imgs.size()[0]
    cha = imgs.size()[1]
    height = imgs.size()[2]
    width = imgs.size()[3]
    trans = ImageTransform(height, (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    idx = 0
    for in_img in imgs:

        img = torch.squeeze(in_img.permute(1,2,0))    
        img = img.numpy()
        # Image.fromarray(np.uint8( img*255 )).save('hoge_.png')
        img = cv2.resize(img , (int(width*scale), int(height*scale)), interpolation=cv2.INTER_NEAREST  )

        img = 255*(img-img.min())/(img.max()-img.min())
        th2_r = (img[:,:,0]/bit).astype('uint8')*bit
        th2_g = (img[:,:,1]/bit).astype('uint8')*bit
        th2_b = (img[:,:,2]/bit).astype('uint8')*bit
        img[:,:,0]=th2_r
        img[:,:,1]=th2_g
        img[:,:,2]=th2_b
        img = cv2.resize(img , (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        
        img = torch.tensor(img).permute(2,0,1)
        if idx<5: Image.fromarray(np.uint8( img.permute(1,2,0) )).save('cifar_{}_{}_{}.png'.format(idx,scale,bit))
        img = img/255
        img_ = trans.data_transform['test'](img)
        imgs[idx] = img_
        idx += 1

    return imgs


def trans_th(imgs):
    bs = imgs.size()[0]
    cha = imgs.size()[1]
    height = imgs.size()[2]
    width = imgs.size()[3]
    trans = ImageTransform(height, (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    idx = 0
    for in_img in imgs:

        img = torch.squeeze(in_img.permute(1,2,0))    
        img = img.numpy()
        # Image.fromarray(np.uint8( img*255 )).save('hoge_.png')
        img = cv2.resize(img , (int(width*.25), int(height*.25)), interpolation=cv2.INTER_NEAREST  )

        # img = (255*(img-img.min())/(img.max()-img.min())).astype('uint8')
        # ret2,th2_r = cv2.threshold(img[:,:,0], 0, 255, cv2.THRESH_OTSU)
        # ret2,th2_g = cv2.threshold(img[:,:,1], 0, 255, cv2.THRESH_OTSU)
        # ret2,th2_b = cv2.threshold(img[:,:,2], 0, 255, cv2.THRESH_OTSU)
        
        img = 255*(img-img.min())/(img.max()-img.min())
        th2_r = (img[:,:,0]/32).astype('uint8')*32
        th2_g = (img[:,:,1]/32).astype('uint8')*32
        th2_b = (img[:,:,2]/32).astype('uint8')*32
        img[:,:,0]=th2_r
        img[:,:,1]=th2_g
        img[:,:,2]=th2_b
        img = cv2.resize(img , (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        
        img = torch.tensor(img).permute(2,0,1)
        img_ = trans.data_transform['train'](img/255)
        # Image.fromarray(np.uint8( img.permute(1,2,0) )).save('hoge.png')
        imgs[idx] = img_
        idx += 1

    return imgs

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


def gen_th2(inp):
    cha = inp.size()[0]
    img = torch.squeeze(inp.permute(1,2,0))
    height = img.size()[0]
    width = img.size()[1]
    
    img = img.numpy()
    img = cv2.resize(img , (int(width*.5), int(height*.5)), cv2.INTER_NEAREST  )

    img = (255*(img-img.min())/(img.max()-img.min())).astype('uint8')
    ret2,th2_r = cv2.threshold(img[:,:,0], 0, 255, cv2.THRESH_OTSU)
    ret2,th2_g = cv2.threshold(img[:,:,1], 0, 255, cv2.THRESH_OTSU)
    ret2,th2_b = cv2.threshold(img[:,:,2], 0, 255, cv2.THRESH_OTSU)
    img[:,:,0]=th2_r
    img[:,:,1]=th2_g
    img[:,:,2]=th2_b
    img = cv2.resize(img , (int(width*1), int(height*1)), cv2.INTER_NEAREST)
    
    img = torch.tensor(img).permute(2,0,1)
    # img = torch.tensor(img).view(1,height, width)
    # if cha > 1: img = img.repeat(cha,1,1)

    Image.fromarray(np.uint8( img.permute(1,2,0) )).save('hoge.png')

    return img

def gen_th(inp):
    cha = inp.size()[0]
    img = torch.squeeze(inp.permute(1,2,0))
    height = img.size()[0]
    width = img.size()[1]
    
    img = img.numpy()
    img = cv2.resize(img , (int(width*0.2), int(height*0.2)), cv2.INTER_NEAREST  )
    img = cv2.resize(img , (int(width*1), int(height*1)), cv2.INTER_NEAREST)

    img = (255*(img-img.min())/(img.max()-img.min())).astype('uint8')

    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img = torch.tensor(th2).view(1,height, width)
    if cha > 1: img = img.repeat(cha,1,1)

    return img


if __name__ == '__main__':
    scale, bit = 0.25, 32
    # gen_keyimg_gray(dataset='./cifar100_n2.pt', outname='./cifar100_n2_key_gray_{}_{}.pt'.format(scale, bit), scale=scale, bit=bit)
    # gen_keyimg_mean(dataset='./cifar100_n2.pt', outname='./cifar100_n2_key_mean.pt')
    gen_keyimg_rand(dataset='./cifar100_n2.pt', outname='./cifar_Rand4_n2.pt', scale=1/4)

    # gen_keyimg_gray(dataset='./cifar100_n5.pt', outname='./cifar100_n5_key_gray_{}_{}.pt'.format(scale, bit), scale=scale, bit=bit)
    # gen_keyimg_mean(dataset='./cifar100_n5.pt', outname='./cifar100_n5_key_mean.pt')
    gen_keyimg_rand(dataset='./cifar100_n5.pt', outname='./cifar_Rand4_n5.pt', scale=1/4)

    # gen_keyimg_gray(dataset='./cifar100_n10.pt', outname='./cifar100_n10_key_gray_{}_{}.pt'.format(scale, bit), scale=scale, bit=bit)
    # gen_keyimg_mean(dataset='./cifar100_n10.pt', outname='./cifar100_n10_key_mean.pt')
    gen_keyimg_rand(dataset='./cifar100_n10.pt', outname='./cifar_Rand4_n10.pt', scale=1/4)
