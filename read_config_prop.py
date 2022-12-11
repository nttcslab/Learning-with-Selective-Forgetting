from argparse import ArgumentParser
import configparser
from distutils.util import strtobool

import numpy as np
import torch

import datetime
import os

import subprocess
from tqdm import tqdm

def input_parameters(ini_file):

    config = configparser.ConfigParser()
    config.read(ini_file)
    print(ini_file)

    params = config["params"]

    task_number         = int(config.get("params","task_number",fallback=2))   # 2
    epochs_per_task     = int(config.get("params","epochs_per_task",fallback=10))   # 100

    gpuid               = str(config.get( "params", "gpuid",fallback='0,1') )# '0,1' 
    delete_class        = str(config.get( "params", "delete_class",fallback='3,4,5') ) #'3,4,5'
    outdir              = str(config.get( "params", "outdir",fallback='test'))  #'test' 
    model_path_dir      = str(config.get( "params", "model_path_dir",fallback='./model'))  #'./models' 
    model_name          = str(config.get( "params", "model_name",fallback='test_HF_model_cifar_CNN'))  #'test_HF_model_cifar_CNN'

    losstype   = str(config.get( "params", "losstype", fallback='adtv'))  # adtv or ce

    batch_size = int(config.get( "params", "batch_size",fallback='128'))   # 128
    test_size= int(config.get( "params", "test_size",fallback='10000'))   # 4048

    random_seed = int(config.get( "params", "random_seed",fallback='0') )  # 0

    lmd_ewc      = float(config.get( "params", "lmd_ewc", fallback='40'))   # 100
    lmd_mas      = float(config.get( "params", "lmd_mas", fallback='1'))   # 100
    lmd_lwf      = float(config.get( "params", "lmd_lwf", fallback='1'))   # 100
    lmd_lwm      = float(config.get( "params", "lmd_lwm", fallback='1'))   # 100
    weight_rev      = float(config.get( "params", "weight_rev", fallback='1'))  #1
    alpha_key    = float(config.get( "params", "alpha_key",fallback='1') )  # 0.5
    weight_decay = float(config.get( "params", "weight_decay",fallback='0') )  # 0

    optm_method = str(config.get( "params", "optm_method",fallback='Adam') ) #'Adam'
    lr_sch      = str(config.get( "params", "lr_sch"     ,fallback='40,60,120,160,200') )  # 1e-4
    mode        = str(config.get( "params", "mode",fallback='MH_MT') )  # 1e-4
    lr = float(config.get( "params", "lr",fallback='1e-4') )  # 1e-4

    am_margin = float(config.get( "params", "am_margin",fallback='0.35') )  # 0.5
    am_scale = float(config.get( "params", "am_scale",fallback='30') )  # 30
    lmd_keep = float(config.get( "params", "lmd_keep",fallback='1') )  # 1

    modeltype   = str(config.get( "params", "modeltype",fallback='CNN') )  # 0
    train_dataset     = str(config.get( "params", "train_dataset",fallback='./data/cifar100_n2.pt') )  # 0
    keys_dataset     = str(config.get( "params", "keys_dataset",fallback='./data/cifar100_n2_key.pt') )  # 0

    is_load_model = int(config.get( "params", "is_load_model",fallback='0') )  # 0
    is_lwf_only = int(config.get( "params", "is_lwf_only", fallback='0') )

    num_keys_per_cls = int(config.get( "params", "num_keys_per_cls",fallback='1') )  # 1
    num_batch_for_keep = int(config.get( "params", "num_batch_for_keep",fallback='1') )  # 1

    is_reg = str(config.get( "params", "is_reg",fallback='EWC') )
    is_dataloader =  int(config.get( "params", "is_dataloader",fallback='1') ) 
    is_lwf_rand =  int(config.get( "params", "is_lwf_rand",fallback='1') ) 
    is_ewcp =  int(config.get( "params", "is_ewcp",fallback='1') ) 
    is_NCM =  int(config.get( "params", "is_NCM",fallback='1') ) 
    is_BC =  int(config.get( "params", "is_BC",fallback='1') ) 



    params_set = [
        task_number, epochs_per_task, \
        gpuid, delete_class, outdir, model_path_dir, model_name, \
        batch_size, test_size, \
        random_seed, \
        lmd_ewc, lmd_mas, weight_rev, alpha_key, weight_decay, \
        optm_method, lr, lr_sch, \
        am_margin, am_scale, lmd_keep, lmd_lwf, lmd_lwm, is_lwf_only, \
        is_load_model, modeltype, train_dataset, keys_dataset, \
        losstype, mode, \
        num_keys_per_cls, num_batch_for_keep, \
        is_reg, is_dataloader, is_lwf_rand, is_ewcp, is_NCM, is_BC, \
        ]    

    return  params_set

def out_parameters(
        net,
        res_dir,
        task_number, epochs_per_task, \
        gpuid, delete_class, outdir, model_path_dir, model_name, \
        batch_size, test_size, \
        random_seed, \
        lmd_ewc, lmd_mas, weight_rev, alpha_key, weight_decay, \
        optm_method, lr, lr_sch, \
        am_margin, am_scale, lmd_keep, lmd_lwf, lmd_lwm, is_lwf_only, \
        is_load_model, modeltype, train_dataset, keys_dataset, \
        losstype, mode, \
        num_keys_per_cls, num_batch_for_keep, \
        is_reg, is_dataloader, is_lwf_rand, is_ewcp, is_NCM, is_BC, \
        ):
    logfile = res_dir + '/param.txt'
    
    netfile = res_dir + '/net.txt'
    with open(netfile, 'a') as f:
        f.write('{}'.format(net))

    with open(logfile, 'a') as f:
        f.write('{}\n'.format(task_number))
        f.write('task_number:{}\n'.format( task_number))
        f.write('epochs_per_task:{}\n'.format(  epochs_per_task ))
        f.write('gpuid:{}\n'.format(  gpuid))
        f.write('delete_class:{}\n'.format(  delete_class))
        f.write('outdir:{}\n'.format(  outdir))
        f.write('model_path_dir:{}\n'.format(  model_path_dir))
        f.write('model_name:{}\n'.format(  model_name))
        f.write('batch_size:{}\n'.format(  batch_size ))
        f.write('test_size:{}\n'.format(  test_size))
        f.write('random_seed:{}\n'.format(  random_seed))
        f.write('lmd_ewc:{}\n'.format(  lmd_ewc))
        f.write('lmd_mas:{}\n'.format(  lmd_mas))
        f.write('lmd_lwm:{}\n'.format(  lmd_lwm))
        f.write('weight_rev:{}\n'.format(  weight_rev))
        f.write('alpha_key:{}\n'.format(  alpha_key))
        f.write('weight_decay:{}\n'.format(  weight_decay))
        f.write('optm_method:{}\n'.format(  optm_method))
        f.write('lr:{}\n'.format(  lr))
        f.write('lr_sch:{}\n'.format(  lr_sch))
        f.write('mode:{}\n'.format(  mode))
        f.write('am_margin:{}\n'.format(  am_margin))
        f.write('am_scale:{}\n'.format(  am_scale))
        f.write('lmd_keep:{}\n'.format(  lmd_keep))
        f.write('lmd_lwf:{}\n'.format(  lmd_lwf))
        f.write('is_lwf_only:{}\n'.format(  is_lwf_only))
        f.write('is_load_model:{}\n'.format(  is_load_model))
        f.write('modeltype:{}\n'.format(  modeltype))
        f.write('train_dataset:{}\n'.format(  train_dataset))
        f.write('keys_dataset:{}\n'.format(  keys_dataset))
        f.write('losstype:{}\n'.format(  losstype))
        f.write('num_keys_per_cls:{}\n'.format(  num_keys_per_cls))
        f.write('num_batch_for_keep:{}\n'.format(  num_batch_for_keep))
        f.write('is_reg :{}\n'.format(is_reg))
        f.write('is_dataloader:{}\n'.format(is_dataloader))
        f.write('is_lwf_rand:{}\n'.format(is_lwf_rand))
        f.write('is_ewcp:{}\n'.format(is_ewcp))
        f.write('is_NCM:{}\n'.format(is_NCM))
        f.write('is_BC:{}\n'.format(is_BC))
    return