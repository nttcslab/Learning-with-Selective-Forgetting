from argparse import ArgumentParser
import numpy as np
import torch

from src.train import train
import src.utils as utils

from src.networks.ResNet import ResNet_MH

import datetime
import os

import subprocess
from tqdm import tqdm

import read_config_prop as read_config

from PIL import Image


parser = ArgumentParser('Learning with Selective Forgetting')
parser.add_argument('--ini-file', type=str, default='./ini/cifar_t5-r3-E10.ini')

parser.add_argument('--outdir', type=str, default= '' )
parser.add_argument('--lr-sch', type=str, default= '')
parser.add_argument('--mode', type=str, default= '')
parser.add_argument('--delete-class', type=str, default= '')
parser.add_argument('--outputsize', type=str, default= '')
parser.add_argument('--num-class-per-task', type=str, default= '')

parser.add_argument('--model-name', type=str, default= '')
parser.add_argument('--lmd-ewc', type=float, default= -1)
parser.add_argument('--lmd-lwf', type=float, default= -1)
parser.add_argument('--lmd-cnt', type=float, default= -1)
parser.add_argument('--lmd-lwm', type=float, default= -1)
parser.add_argument('--alpha-key', type=float, default=-1)
parser.add_argument('--beta-key', type=float, default=-1)
parser.add_argument('--weight-rev', type=float, default= -1 )
parser.add_argument('--lmd-keep', type=float, default= -1)

parser.add_argument('--num-keys-per-cls', type=int, default=-1)
parser.add_argument('--keys-dataset', type=str, default='')
parser.add_argument('--keys-name', type=str, default='')
parser.add_argument('--task-number', type=int, default=-1)
parser.add_argument('--epochs-per-task', type=int, default=-1)
parser.add_argument('--is-load-model', type=int, default=-1)
parser.add_argument('--is-lwf-only', type=int, default=-1)
parser.add_argument('--losstype', type=int, default=-1)

parser.add_argument('--num-batch-for-keep', type=int, default=-1)
parser.add_argument('--random-seed', type=int, default=-1)

parser.add_argument('--is-reg', type=str, default='EWC')
parser.add_argument('--is-dataloader', type=int, default=-1)
parser.add_argument('--is-lwf-rand', type=int, default=-1)
parser.add_argument('--is-ewcp', type=int, default=-1)
parser.add_argument('--is-NCM', type=int, default=-1)
parser.add_argument('--is-BC', type=int, default=-1)

def sample_keys(x_tr_key, n_outputs=100, num_sample=10, channel=3, sz=32):
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


if __name__ == '__main__':
    args = parser.parse_args()

    [
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
        ] = read_config.input_parameters(args.ini_file)
    

    if len(args.outdir)>0: outdir = args.outdir
    if len(args.delete_class)>0: delete_class = args.delete_class
    if len(args.lr_sch)>0: lr_sch = args.lr_sch
    if len(args.mode)>0: mode = args.mode
    if len(args.model_name)>0: model_name = args.model_name
    if len(args.keys_dataset)>0: keys_dataset = args.keys_dataset
    if args.is_reg!='EWC': is_reg = args.is_reg

    if args.num_keys_per_cls>-1: num_keys_per_cls = args.num_keys_per_cls
    if args.lmd_ewc>-1: lmd_ewc = args.lmd_ewc
    if args.lmd_lwm>-1: lmd_lwm = args.lmd_lwm
    if args.alpha_key>-1: alpha_key = args.alpha_key
    if args.beta_key>-1: beta_key = args.beta_key
    if args.weight_rev>-1: weight_rev = args.weight_rev
    if args.lmd_keep>-1: lmd_keep = args.lmd_keep 
    if args.lmd_lwf>-1: lmd_lwf = args.lmd_lwf 
    if args.task_number>-1: task_number = args.task_number 
    if args.epochs_per_task>-1: epochs_per_task = args.epochs_per_task 
    if args.is_load_model>-1: is_load_model = args.is_load_model 
    if args.is_lwf_only>-1: is_lwf_only = args.is_lwf_only 
    if args.losstype>-1: losstype = args.losstype
    if args.random_seed>-1: random_seed = args.random_seed
    if args.num_batch_for_keep>-1: num_batch_for_keep = args.num_batch_for_keep
    if args.is_dataloader>-1: is_dataloader = args.is_dataloader
    if args.is_BC>-1: is_BC = args.is_BC

    if args.is_lwf_rand>-1: is_lwf_rand = args.is_lwf_rand
    if args.is_ewcp>-1: is_ewcp = args.is_ewcp
    if args.is_NCM>-1: is_NCM = args.is_NCM

    if len(args.num_class_per_task)>0: 
        num_class_per_task = args.num_class_per_task
        num_class_per_task = [int(t) for t in num_class_per_task.split(",")]
    else:
        num_class_per_task = [20,20,20,20,20]


    str_prm = 'lmd_ewc:{}'.format(lmd_ewc) +'_weight_rev:{}'.format(weight_rev) \
        + '\n LwF:{}'.format(lmd_lwf) + '\n lmd_keep:{}'.format(lmd_keep) \
        + '\n alpha_key:{}'.format(alpha_key) +  '\n is_lwf_only:{}'.format(is_lwf_only) \
        + '\n losstype:{}'.format(losstype) \
        + '\n am_margin:{}'.format(am_margin) + '\n am_scale:{}'.format(am_scale) \
        + '\n num_batch_for_keep:{}'.format(num_batch_for_keep) \
        + '\n keys_dataset:{}'.format(keys_dataset)    

    gpuid = [int(t) for t in gpuid.split(",")]


    #-------------------------------------------
    # define target classes 
    #-------------------------------------------
    delete_class_org = delete_class.split(',')
    delete_class = [int(t) for t in delete_class_org]

    lr_sch = lr_sch.split(',')
    lr_sch = [int(t) for t in lr_sch]

    #-------------------------------------------
    # define output directry
    #-------------------------------------------
    now = datetime.datetime.now()
    res_dir = './result/' + outdir + '_' + now.strftime('%m%d_%H%M%S') 
    print(res_dir)
    if not os.path.exists('./result/'): os.mkdir('./result/')
    if not os.path.exists(res_dir): os.mkdir(res_dir) 


    #-------------------------------------------
    # decide whether to use cuda or not.
    #-------------------------------------------
    cuda = torch.cuda.is_available()

    #-------------------------------------------
    # generate permutations for the tasks.
    #-------------------------------------------
    np.random.seed(random_seed)

    #-------------------------------------------
    # load datasets
    #-------------------------------------------
    x_tr, x_te, x_va, n_inputs, n_outputs, num_tasks_max, num_class_max, _, is_task_wise, is_perterb, img_size = \
         utils.load_datasets(train_dataset)
    x_tr_key, scale, bit = torch.load(keys_dataset)
    sz = img_size[0]
    channel = img_size[2]
    task_number = min(task_number, num_tasks_max) 
    x_tr_key = sample_keys(x_tr_key, n_outputs=n_outputs, num_sample=num_keys_per_cls, channel=channel, sz=sz)

    

    if is_perterb == 0:
        train_datasets = [utils.Dataset_SCL_woAgm(x_tr[p], sz, sz, channel,phase="train") for p in range(task_number)]
        test_datasets  = [utils.Dataset_SCL_woAgm(x_te[p], sz, sz, channel,phase="test") for p in range(task_number)]
    else: 
        train_datasets = [utils.Dataset_CIFAR(x_tr[p], sz, sz, channel,phase="train") for p in range(task_number)]
        test_datasets  = [utils.Dataset_CIFAR(x_te[p], sz, sz, channel,phase="test") for p in range(task_number)]


    #-------------------------------------------
    # Define Network
    #-------------------------------------------
    if is_reg=='EWC':
        lmd_weight_loss = lmd_ewc 
    elif is_reg=='MAS':
        lmd_weight_loss = lmd_mas     
    else: 
        lmd_weight_loss = 0
    
    if mode=='SH_INC':                    
        net = ResNet_MH(sz**2, num_class_per_task, lmd_weight_loss = lmd_weight_loss,  num_task=task_number )
    elif mode=='SH_MT':
        head_number = 1
        net = ResNet_MH(sz**2, num_class_per_task, lmd_weight_loss = lmd_weight_loss,  num_task=head_number )
    else:
        net = ResNet_MH(sz**2, num_class_per_task, lmd_weight_loss = lmd_weight_loss,  num_task=task_number )
    
    #-------------------------------------------
    # output the loaded parameters (e.g. outputdir, model path etc)
    #-------------------------------------------
    read_config.out_parameters(
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
        )


    # initialize the parameters.
    utils.xavier_initialize(net)

    if cuda: net.cuda()

    train(
        net, 
        train_datasets, 
        test_datasets,
        x_tr_key,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        test_size=test_size,
        lr=lr,
        weight_decay=weight_decay,
        cuda=cuda,
        weight_rev = weight_rev,
        res_dir = res_dir,
        delete_class = delete_class,
        gpuid=gpuid,
        optm_method = optm_method,
        lmd_weight_loss = lmd_weight_loss ,
        lmd_lwf = lmd_lwf,
        lmd_lwm = lmd_lwm,
        str_prm = str_prm,
        am_margin = am_margin,
        am_scale = am_scale,
        lmd_keep = lmd_keep,
        alpha_key = alpha_key,
        input_size = img_size,
        is_task_wise = is_task_wise,
        losstype = losstype,
        is_lwf_only = is_lwf_only,
        num_keys_per_cls = num_keys_per_cls,
        num_batch_for_keep = num_batch_for_keep,
        lr_sch=lr_sch,
        mode=mode,
        is_reg = is_reg,
        is_dataloader = is_dataloader,
        is_lwf_rand=is_lwf_rand,
        is_ewcp=is_ewcp,
        is_NCM=is_NCM,
        is_BC=is_BC,
        model_path_dir=model_path_dir,
        is_load = is_load_model,
    )



