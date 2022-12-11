import copy
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CyclicLR, StepLR
from tqdm import tqdm

import src.utils as utils
import src.utils_eval as utils_eval
import src.utils_BC as utils_BC
from src.utils_BC import BiasLayer_AMS as BiasLayer
import src.utils_Regs as utils_Regs
#------------
from src.loss_fun import *
from src.lr_scheduler_func import (FindLR, WarmUpLR, lr_scheduling_func,lr_scheduling_func_step)
from math import log, e
import datetime


def cmp_entropy(logits):
    f = F.softmax(logits, dim=1)
    f = ((f+1E-10).log())
    return -f.sum()


def sample_keys_batch(x_keys, y_keys, num_batch):
    idx = torch.randint(0, len(y_keys), (num_batch, ) )
    return x_keys[idx], y_keys[idx]

def sample_keys_batch_cls(x_keys, y_keys, labels):
    x_tmp = []
    for id, y_tmp in enumerate(labels):
        idxs = y_keys == y_tmp
        idxs = torch.nonzero(idxs.int(), as_tuple=False)

        if len(idxs)>0:
            idx = idxs[torch.randperm(len(idxs))[0]]

            if len(x_tmp) == 0:
                x_tmp = x_keys[idx]
            else:
                x_tmp_tmp = x_keys[idx]
                x_tmp = torch.cat((x_tmp, x_tmp_tmp), dim = 0)

    return x_tmp, labels

def sample_keys(x_keys_org, y_keys_org, keep_class, num_task, channel, sz):
    x_keys = [ x_keys_org[t].clone() for t in range(len(x_keys_org)) ]
    y_keys = [ y_keys_org[t].clone() for t in range(len(x_keys_org)) ]
    for t in range(num_task):
        idxs_all = []
        for c in keep_class:
            y = y_keys[t]
            idxs = y==c
            if len(torch.nonzero(idxs.int(), as_tuple=False))>0:
                if len(idxs_all):
                    idxs_all = torch.cat( (idxs_all, torch.nonzero(idxs.int(), as_tuple=False)) )
                else:
                    idxs_all = torch.nonzero(idxs.int(), as_tuple=False)
        x_keys[t] = x_keys[t][idxs_all].clone().view(-1,channel,sz,sz)
        y_keys[t] = y_keys[t][idxs_all].clone().squeeze()
    return x_keys, y_keys

#----------------------------------------------------
def gen_yrand_for_dist(y_rand, num_class_per_task, delete_class, task, class_label_offset):
    for tsk in range(task):
        keep_num = 0

        for k in range(num_class_per_task[tsk]):
            c_tmp = k + class_label_offset[tsk]
            if c_tmp in delete_class:
                keep_num += 1
        if keep_num==0: keep_num=1

        for k in range(num_class_per_task[tsk]):
            c_tmp = k + class_label_offset[tsk]
            if c_tmp in delete_class:
                y_rand[tsk][:,k] = 0
            else:
                y_rand[tsk][:,k] = 1/keep_num
    return y_rand

#----------------------------------------------------
def gen_mask_for_dist(batch_size, num_class_per_task, delete_class, task, num_task, class_label_offset):
    msk_dist = [torch.ones(batch_size, num_class_per_task[i]) for i in range(len(num_class_per_task))]
    for tsk in range(task):
        for k in range(num_class_per_task[tsk]):
            # Exception Handling for Multihead-Class Incremental 
            c_tmp = k + class_label_offset[tsk]
            if c_tmp in delete_class:
                msk_dist[tsk][:,k] = 0
    return msk_dist

#----------------------------------------------------
def mix_Keyimg(x, Keyimgs, beta_key, alpha_key):
    mixed_imgs = torch.rand(Keyimgs.size())
    for id, (x_tmp, Keyimg_tmp) in enumerate(zip(x, Keyimgs)):
        if alpha_key != 0:
            if beta_key==0:
                if np.random.rand()>0.5:
                    lam = 1
                else:
                    lam = 0
            else:
                lam = np.random.beta(beta_key, beta_key)


            if id == 0:
                mixed_imgs = (1-alpha_key)*x_tmp + alpha_key*(lam*x_tmp + (1-lam)*Keyimg_tmp)
            else:
                mixed_imgs_tmp = (1-alpha_key)*x_tmp + alpha_key*(lam*x_tmp + (1-lam)*Keyimg_tmp)
                mixed_imgs = torch.cat((mixed_imgs, mixed_imgs_tmp), dim = 0)
        else:
            if id == 0:
                mixed_imgs = x_tmp
            else:
                mixed_imgs = torch.cat((mixed_imgs, x_tmp), dim = 0)

    mixed_imgs = mixed_imgs.view_as(x)
    mixed_imgs = mixed_imgs.float()
    return mixed_imgs

def gen_idxs_c(num_class_per_task):
    idxs = torch.ones(torch.tensor(num_class_per_task).sum())
    tmp = torch.tensor(num_class_per_task).clone()
    tmp2 = torch.tensor(num_class_per_task).clone()
    for k in range(len(num_class_per_task)): tmp[k]=tmp2[:(k+1)].sum()
    for i in range(len(idxs)): idxs[i] = (tmp<i).float().sum().item()
    return idxs


def train(net,
          train_datasets,
          test_datasets,
          x_tr_key,
          epochs_per_task=10,
          batch_size=64,
          test_size=1024,
          lr=1e-3,
          optm_method = 'Adam',
          weight_decay=1e-5,
          cuda=False,
          res_dir = "out",
          delete_class = [0,1,2,3,4,5],
          gpuid=[0],
          lmd_weight_loss = 40,
          weight_rev = 0.0,
          lmd_lwf = 1,
          lmd_keep = 1,
          lmd_lwm=1,
          alpha_key = 1,
          str_prm = 'none',
          am_margin = 0.35,
          am_scale = 30,
          input_size = [28,28, 1],
          is_task_wise = 1,
          is_lwf_only = 0,
          losstype = 'adtv',
          num_keys_per_cls = 1,
          num_batch_for_keep = 32,
          lr_sch=[40,60,120,160,200],
          mode='MH_MT',
          is_reg ='EWC',
          is_dataloader=1,
          is_lwf_rand=1,
          is_ewcp=1,
          is_NCM=0,
          is_BC=1,
          model_path_dir='',
          is_load = 1,
        ):
    fisher_est_sample=1024
    batch_size_lite = batch_size
    num_batch_for_keep = batch_size

    print('is_reg:{}_lwf_rand:{}_ewcp:{}_NCM:{}_lwm:{}_BC:{}'.format(is_reg, is_lwf_rand, is_ewcp, is_NCM, lmd_lwm, is_BC))

    is_dbg=0
    is_save=1

    if is_dbg: torch.autograd.set_detect_anomaly(True)

    is_CM=1
    clipgrad=10000

    suffix = 'EW{}'.format(int(lmd_weight_loss)) + '_LF{}'.format(int(lmd_lwf)) + '_K{}'.format(int(lmd_keep)) \
        + '_A{}'.format(int(alpha_key))+ '_NB{}'.format(int(num_batch_for_keep)) + '_{}'.format(mode)

    ##################################
    # Store the original mnemonic as ***_org
    ##################################
    x_keys = []
    y_keys = []
    for keys_tmp in x_tr_key:
        x_keys.append(keys_tmp[1])
        y_keys.append(keys_tmp[2])
    x_keys_org = x_keys.copy()
    y_keys_org = y_keys.copy()

    # Copy the network for distillation 
    net_copy = copy.deepcopy(net)
    net_copy.cuda().eval()
    
    #####################################
    ###------- set initial val
    #####################################
    # ***The class of each task is the same as the number of classes of each head*** 
    # ***Here we assume the same number for all (net.output_size))***
    num_class_per_task = net.output_size

    sz = input_size[1]
    channel = input_size[2]
    num_task = len(train_datasets)

    # ***Calculate the offset of the class label when the task is switched.***
    class_label_offset = utils.cmp_class_offset(num_task, is_task_wise, num_class_per_task)

    idxs = gen_idxs_c(num_class_per_task)

    #####################################
    ###   init
    #####################################
    weight_loss, weight_loss_val = 0, 0
    loss_key = 0
    loss_val = 0
    loss_keep_val = 0
    loss_keep_new_val = 0
    loss_rev_val = 0
    loss_lwf_val = 0
    loss_key_val  = 0
    weight_loss_val = 0
    loss_lwf_val = 0
    loss_lwm = 0
    loss_lwm_val = 0
    loss_new = 0
    precision = 0
    prec_trgt_old = 0
    prec_non_trgt_old = 0
    H, H2 = 0, 0

    A_keep, A_ave, F_keep, F_del = 0, 0, 0, 0

    #####################################
    #### generate output dir
    #####################################
    res_dir_ch = res_dir + '/sup'
    if not os.path.exists(res_dir): os.mkdir(res_dir)
    if not os.path.exists(res_dir_ch): os.mkdir(res_dir_ch)

    logfile = res_dir + '/log.txt'

    #####################################
    ### prepare the loss criterion and the optimizer.
    #####################################
    net = torch.nn.DataParallel(net, gpuid)

    #####################################
    ### define loss functions
    #####################################
    # ***aditive margin loss***
    criterion_adtv   = AdditiveMarginLoss(scale=am_scale, margin=am_margin)

    # ***distilition loss with randamized value for dilition sets***
    criterion_rand   = Distillation_Loss(T=2.0, scale=am_scale)
    criterion_dist_NC   = Distillation_Loss_Forget(T=2.0, scale=am_scale)
    criterion_dist_org   = Distillation_Loss_org(T=2.0, scale=am_scale)

    yrand    = [(torch.ones(batch_size, k)/k).cuda() for k in num_class_per_task]
    yrand_NC = [(torch.ones(batch_size_lite, k)/k).cuda() for k in num_class_per_task]
    yrand_rev = [(torch.ones(num_task,batch_size_lite, k)/k).cuda() for k in num_class_per_task]

    # ***distiliation loss with mask***
    criterion_dist   = Distillation_Loss_MSK(T=2.0, scale=am_scale)

    # ***attention distilition loss for LwM (not used) ***
    # if lmd_lwm>0: criterion_adist  = Attention_Distillation_Loss()

    #####################################
    ### define classes
    #####################################
    # *** Generate a save class as a complement to the delete class. ***
    # *** Complementary set of classes for all tasks ***
    keep_class = [int(t) for t in range(sum(num_class_per_task)) if t not in delete_class]

    #####################################
    ##### set the model's mode to training mode.
    #####################################
    net.train()

    loss_p, acc_all = np.zeros((1,1)), np.zeros((1,len(train_datasets)))
    iter_p = []

    loss_p_part = np.zeros((1,6))
    loss_p_name = ['Loss','Keep(LwF)','Loss_New','EWC', 'Keep_land', 'Revgrad']

    acc_selective = [ np.zeros( (len(train_datasets), 2) ) ]
    iteration = 0
    acc_key = 0

    #################################
    ## Load Terminated Results
    #################################
    if is_load==1 and os.path.exists(model_path_dir):
        cptfile = model_path_dir
        cpt = torch.load(cptfile)
        stdict_m = cpt['model_state_dict']
        stdict_o = cpt['opt_state_dict']
        # optimizer.load_state_dict(stdict_o)
        task_terminated = cpt['task']
        epoch_terminated = cpt['epoch']
        net.module.consolidate_selective_plus_ini(task_terminated)
        net.module.load_state_dict(stdict_m)
        is_optim_loaded = 0

        acc_all = cpt['acc_all']
        epoch_terminated = cpt['epoch']
        acc_selective = cpt['acc_selective']
        num_selective = cpt['num_selective']
        loss_p = cpt['loss_p']
        iter_p = cpt['iter_p']
        loss_p_part = cpt['loss_p_part']
        A_keep_all = cpt['A_keep_all']
        A_ave_all = cpt['A_ave_all'] 
        F_keep_all = cpt['F_keep_all'] 
        F_del_all = cpt['F_del_all'] 
        current_task_iteration = cpt['current_task_iteration']
    else:
        task_terminated = 0
        epoch_terminated = 0



    #----------------------------
    for task, train_dataset in enumerate(train_datasets, 0):
        # if the task was already finished: the task is skipped:
        if task > 0: total_class_prev = torch.tensor(num_class_per_task).clone()[:(task)].sum().item()
        if task < task_terminated: 
            print('skip the task: task:{}'.format(task))
            #####################################
            #### add a bias layer for the new classes (BiC)
            #####################################
            if is_BC:
                net.module.bias_layers.append(BiasLayer().cuda())
                for k in range(task+1):
                    net.module.bias_layers[k].alpha.data = torch.ones(1, requires_grad=False, device="cuda")
                    net.module.bias_layers[k].beta.data  = torch.zeros(1, requires_grad=False, device="cuda") 

                for k in range(task + 1):
                    print('Task {:d}: alpha={:.5f}, beta={:.5f}'.format(k,net.module.bias_layers[k].alpha.item(), net.module.bias_layers[k].beta.item()))  

            continue            

        #####################################
        ##### output samples
        #####################################
        if is_dbg: 
            # *** Output images for each class ***
            utils_eval.out_images(train_dataset, res_dir_ch, cuda, task,is_dataloader)
            # *** Output mnemonic code ***
            for n in range(5):
                img = x_tr_key[task][1][n].permute(1,2,0)
                img = 255*(img-img.min())/(img.max()-img.min()+0.001)
                Image.fromarray(np.uint8( img.squeeze() )).save('{}/key_{}_{}.png'.format(res_dir_ch,task,n))

        #####################################
        ##### set leraning protcols
        #####################################
        ##### define optim parames
        optim_param = net.parameters() # if optm_layer == 'all' else net.final_layer_param()
        ##### set optimizer
        if optm_method == 'SGD':
            optimizer = optim.SGD(optim_param, lr=lr, weight_decay=5e-4, momentum=0.9,)
        elif optm_method == 'Adam':
            optimizer = optim.Adam(optim_param, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(optim_param, lr=lr, weight_decay=5e-4, momentum=0.9)

        ##### set scheduler
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_sch, gamma=0.2)
        if is_dataloader==1:
            warmup_scheduler = WarmUpLR(optimizer, int(len(train_datasets[0])/batch_size))
        else:
            warmup_scheduler = WarmUpLR(optimizer, int(len(train_datasets[task])))

        # this zero gradient update is needed to avoid a warning message, issue #8.
        optimizer.zero_grad()
        optimizer.step()

        if (is_load ==1 and  is_optim_loaded ==0):
            optimizer.load_state_dict(stdict_o)
            is_optim_loaded = 1


        #####################################
        ##### lifelong leraning with Selective Forgetting
        #####################################

        # ---------------------------------------------------
        #####################################
        ##### difine mask for distilltion loss and copy the network for previous parameters
        #####################################
        if task > 0:
            # generate mask for distilltion loss
            if is_lwf_only == 0:
                msk_dist =  gen_mask_for_dist(batch_size, num_class_per_task, delete_class, task, num_task, class_label_offset)
                msk_dist_NC =  gen_mask_for_dist(batch_size_lite, num_class_per_task, delete_class, task, num_task, class_label_offset)
            else:
                msk_dist = torch.ones(batch_size, num_class_per_task)
                msk_dist_NC =  gen_mask_for_dist(batch_size_lite, num_class_per_task, delete_class, task, num_task, class_label_offset)

            # yrand_rev = gen_yrand_for_dist(yrand_rev, num_class_per_task, delete_class, task, class_label_offset)
            msk_dist = [a.cuda() for a in msk_dist]
            msk_dist_NC = [a.cuda() for a in msk_dist_NC]
            # yrand_rev = yrand_rev.cuda()

            # sampling mnemonic codes
            x_keys, y_keys         = sample_keys(x_keys_org, y_keys_org, keep_class, task, channel, sz)
            x_keys_del, y_keys_del = sample_keys(x_keys_org, y_keys_org, delete_class, task, channel, sz)

            # copy the previous model's parameters
            net_copy.load_state_dict(net.module.state_dict(), strict = False)
            net_copy.eval()
            # ---------------------------------------------------

        ######### initillze precs 
        precs_all = torch.zeros(len(train_datasets))
        precs_selective = torch.zeros(len(train_datasets),2)

        ######### compute total # of class
        total_num_class = num_class_per_task if is_task_wise==1 else sum(num_class_per_task[:(task+1)])

        previous_task_iteration = sum([ epochs_per_task * len(d) // batch_size for d in train_datasets[:task]])

        # prepare the data loaders.
        if is_dataloader==1:
            # for cifar100, mnist, svhn
            data_loader = utils.get_data_loader(train_dataset, batch_size=batch_size, cuda=cuda)
        else:
            # for imagenet, stanford cars, cub2011
            data_loader = train_dataset
        # evaluate data sets size
        dataset_batches = len(data_loader)

        #####################################
        #### add a bias layer for the new classes (BiC)
        #####################################
        if is_BC:
            net.module.bias_layers.append(BiasLayer().cuda())
            net.module.bias_layers[task].alpha.data = torch.ones(1, requires_grad=False, device="cuda")
            net.module.bias_layers[task].beta.data  = torch.zeros(1, requires_grad=False, device="cuda") 
            for k in range(task + 1):
                print('Task {:d}: alpha={:.5f}, beta={:.5f}'.format(k,net.module.bias_layers[k].alpha.item(), net.module.bias_layers[k].beta.item()))  

        #####################################
        ##### loop epochs for training
        #####################################
        for epoch in range(1, epochs_per_task+1):
            if epoch <= epoch_terminated:
                print('skip the training step :epoch:{}'.format(epoch))            
                continue
            else:
                epoch_terminated = 0

            data_stream = tqdm(enumerate(data_loader, 1))

            # #####################################
            # #### add a bias layer for the new classes (BiC)
            # #####################################
            # if is_BC:
            #     net.module.bias_layers.append(BiasLayer().cuda())

            for batch_index, (x, y) in data_stream:
                # if batch_index==10: break
                if epoch <= 1: warmup_scheduler.step()                    
                if y.size()[0] != batch_size: break

                # If not a detaloder, add the offset of the class (Cub, STN, ImageNet)
                if is_dataloader is not 1: y = y + class_label_offset[task]

                #### compute progress info.
                current_task_iteration = (epoch-1)*dataset_batches + batch_index
                iteration = previous_task_iteration + current_task_iteration

                # #### sample keys from set of mnemonic codes
                if alpha_key != 0:
                    if is_dataloader:
                        # If a detaloder, Not add the offset of the class (Cub, STN, ImageNet)
                        x_tmp, y_tmp = sample_keys_batch_cls(x_keys[task], y_keys[task], y)
                    else:
                        # If not a detaloder, add the offset of the class (Cub, STN, ImageNet)
                        x_tmp, y_tmp = sample_keys_batch_cls(x_keys[task], y_keys[task], y-class_label_offset[task])
                        y_tmp = y_tmp + class_label_offset[task]
                    x_tmp = mix_Keyimg(x, x_tmp, beta_key=1, alpha_key=1)
                    x_tmp = x_tmp.float().cuda()
                    y_tmp = y_tmp.cuda()

                ####----------------------------
                # transfar to gpu
                x = x.cuda()
                y = y.cuda()

                #---------------------------------------
                net.train()
                optimizer.zero_grad()


                #####################################
                ##### compute class loss
                #####################################
                # ***compute logits for new task (t=task)***
                if mode=='MH_MT':
                    logits_new, norm_logits = net(x, task)
                    # ***bias correction***
                    if is_BC:
                        logits_new = net.module.bias_layers[task](logits_new)
                else:
                    # *** In case of multi-class-IL, heads are integrated to calculate logits ***
                    logits_new = []
                    for tsk in range(task+1):
                        logits_tmp,_ = net(x,tsk)
                        # ***bias correction***
                        if is_BC:
                            logits_tmp = net.module.bias_layers[tsk](logits_tmp)
                        logits_new.append(logits_tmp)
                    logits_new = torch.cat(logits_new,dim=1).view(len(y),-1)

                # evaluate clasification loss for new task
                if mode=='MH_MT':
                    # Align class offsets (subtract zero in the new version: first task)
                    y_new = y - class_label_offset[task]
                    y_new = y_new.cuda()
                    # evaluate clasification loss for new task
                    loss_new = criterion_adtv(logits_new, F.one_hot(y_new, num_classes=num_class_per_task[task]))
                else:
                    # evaluate clasification loss for new task
                    loss_new = criterion_adtv(logits_new, F.one_hot(y.cuda(), num_classes=total_num_class))

                        
                # check nan
                if is_dbg: 
                    if torch.isnan(loss_new): print('loss_new:NAN'); sys.exit()

                #--------------------------------------------
                #####################################
                ##### compute mnemonic loss
                #####################################
                if alpha_key != 0:
                    # compute logits for nimonic loss
                    if mode=='MH_MT':
                        logits_key, _ = net(x_tmp, task)
                        if is_BC: 
                            logits_key = net.module.bias_layers[task](logits_key)
                    else:
                        logits_key = []
                        for tsk in range(task+1):
                            logits_key_tmp,_ = net(x_tmp,tsk)
                            if is_BC:
                                logits_key_tmp = net.module.bias_layers[tsk](logits_key_tmp)
                            logits_key.append(logits_key_tmp)
                        logits_key = torch.cat(logits_key,dim=1).view(len(y),-1)


                    # compute mnemonic loss
                    if mode=='MH_MT':
                        # Align class offsets (subtract zero in the new version: first task)
                        y_new_tmp = y_tmp - class_label_offset[task]
                        y_new_tmp = y_new_tmp.cuda()
                        loss_key = criterion_adtv(logits_key, F.one_hot(y_new_tmp, num_classes=num_class_per_task[task]))
                    else:
                        loss_key = criterion_adtv(logits_key, F.one_hot(y_tmp.cuda(), num_classes=total_num_class))

                    # check nan
                    if is_dbg: 
                        if torch.isnan(loss_key): print('loss_key:NAN'); sys.exit()


                #--------------------------------------------
                #####################################
                ##### compute EWC loss or MAS loss based on weight coff.
                #####################################
                if task > 0:
                    # weight_loss_val = 0

                    if lmd_weight_loss > 0:
                        if is_ewcp:
                            # compute ewc+ (memory save mode)
                            weight_loss = net.module.weight_loss_light(cuda=cuda, weight_rev=weight_rev, task=task)
                        else:
                            # compute naive ewc (compute ewc for each task)
                            weight_loss = net.module.weight_loss(cuda=cuda, weight_rev=weight_rev, task=task)
                        weight_loss_val = weight_loss.item()

                        # check nan
                        if is_dbg: 
                            if torch.isnan(weight_loss): print('weight_loss:NAN'); sys.exit()

                # loss_all = loss_new + loss_key + weight_loss
                loss_all = loss_new + loss_key + weight_loss + loss_lwm

                #--------------------
                # For momory savemode
                loss_all.backward(retain_graph=False)
                loss_all = 0
                #--------------------

                ####################################################################
                if task > 0:
                    ##################################################
                    ## LwF (distillation loss) for mnemonic codes with original samples
                    ## ToDo: memory and computational time reduction
                    ##################################################
                    if lmd_lwf != 0:
                        loss_lwf_val = 0

                        # tsk_prev = int(idxs[torch.randint(low=0, high=total_class_prev, size=(1,)).item()].item())
                        for tsk_prev in range(task):
                            logits_lwf, norm_logits  = net(x, tsk_prev)
                            if is_BC:
                                logits_lwf = net.module.bias_layers[tsk_prev](logits_lwf)

                            target_logits, _ = net_copy(x, tsk_prev)
                            if is_BC:
                                target_logits = net.module.bias_layers[tsk_prev](target_logits)

                            # compute distillation loss
                            if is_lwf_rand:
                                # using random value for delition classes
                                loss_lwf_tmp = criterion_rand(logits_lwf, target_logits, msk_dist[tsk_prev], lmd_lwf, yrand[tsk_prev])
                            else:
                                # ignoring delition classes by mask
                                loss_lwf_tmp = criterion_dist(logits_lwf, target_logits, msk_dist[tsk_prev], lmd_lwf)

                            loss_lwf_val += loss_lwf_tmp.item()
                            loss_all += loss_lwf_tmp
                            # loss_all += loss_lwf_tmp*float(task)

                            if is_dbg: 
                                if torch.isnan(loss_lwf_tmp): print('loss_lwf_tmp:NAN tskpred:{}'.format(tsk_prev)); sys.exit()

                        ##################################################
                        # For momory savemode
                        loss_all.backward(retain_graph=False)
                        loss_all = 0
                        ##################################################

                    ##################################################
                    ## LwF (distillation loss) for original samples
                    ##################################################
                    if lmd_lwf != 0:
                        if alpha_key != 0:
                            # tsk_prev = int(idxs[torch.randint(low=0, high=total_class_prev, size=(1,)).item()].item())
                            for tsk_prev in range(task):
                                logits_lwf, _  = net(x_tmp, tsk_prev)
                                if is_BC:
                                    logits_lwf = net.module.bias_layers[tsk_prev](logits_lwf)

                                target_logits, _    = net_copy(x_tmp, tsk_prev)
                                if is_BC:
                                    target_logits = net.module.bias_layers[tsk_prev](target_logits)

                                if is_lwf_rand:
                                    loss_lwf_tmp = criterion_rand(logits_lwf, target_logits, msk_dist[tsk_prev], lmd_lwf, yrand[tsk_prev])
                                else:
                                    loss_lwf_tmp = criterion_dist(logits_lwf, target_logits, msk_dist[tsk_prev], lmd_lwf)

                                loss_lwf_val += loss_lwf_tmp.item()
                                loss_all += loss_lwf_tmp
                                # loss_all += loss_lwf_tmp*float(task)

                                if is_dbg: 
                                    if torch.isnan(loss_lwf_tmp): print('alpha_ loss_lwf_tmp:NAN tskpred:{}'.format(tsk_prev)); sys.exit()

                            #--------------------
                            # For momory savemode
                            loss_all.backward(retain_graph=False)
                            loss_all = 0
                            #--------------------

                    #--------------------------------------------


                    #--------------------------------------------
                    ##################################################
                    # SF Loss
                    ##################################################
                    if (alpha_key != 0) and (lmd_keep != 0):

                        loss_keep_val = 0 # loss_rev_val = 0
                        batch_size_lite = num_batch_for_keep

                        ####################################
                        # tsk_prev = int(idxs[torch.randint(low=0, high=total_class_prev, size=(1,)).item()].item())
                        for tsk_prev in range(task):
                            x_tmp, y_tmp = sample_keys_batch(x_keys[tsk_prev], y_keys[tsk_prev], batch_size_lite)
                            # If not a detaloder, add the offset of the class (Cub, STN, ImageNet)
                            if is_dataloader is not 1: y_tmp = y_tmp + class_label_offset[tsk_prev]

                            x_tmp = x_tmp.float().cuda()

                            y_tmp = y_tmp.cuda()
                            logits_keep, _ = net(x_tmp, tsk_prev)
                            if is_BC:
                                logits_keep = net.module.bias_layers[tsk_prev](logits_keep)
                            
                            loss_keep_tmp = criterion_adtv(logits_keep, F.one_hot(y_tmp-class_label_offset[tsk_prev], num_classes=num_class_per_task[tsk_prev] ), lmd_keep)

                            loss_keep_val += loss_keep_tmp.item()
                            loss_all += loss_keep_tmp
                            # loss_all += loss_keep_tmp*float(task)

                            ### check nan for loss-keep
                            if is_dbg: 
                                if torch.isnan(loss_keep_tmp): print('loss_keep_tmp:NAN tskpred:{}'.format(tsk_prev)); sys.exit()

                        loss_all.backward(retain_graph=False)
                        loss_all = 0


                        if lmd_lwm > 0:
                            loss_all = 0
                            # tsk_prev = int(idxs[torch.randint(low=0, high=total_class_prev, size=(1,)).item()].item())
                            for tsk_prev in range(task):
                                x_tmp, y_tmp = sample_keys_batch(x_keys_del[tsk_prev], y_keys_del[tsk_prev], batch_size_lite)

                                # If not a detaloder, add the offset of the class (Cub, STN, ImageNet)
                                if is_dataloader is not 1: y_tmp = y_tmp + class_label_offset[tsk_prev]

                                x_tmp = x_tmp.float().cuda()

                                y_tmp = y_tmp.cuda()

                                logits_lwf, _  = net(x_tmp, tsk_prev)
                                loss_all += (1/num_class_per_task[tsk_prev])*cmp_entropy(logits_lwf)
                                # loss_all += (1/num_class_per_task[tsk_prev])*cmp_entropy(logits_lwf)*float(task)

                            loss_all.backward(retain_graph=False)

                        loss_all = 0
                #--------------------------------------------
                
                ##### Clip gradient norm: clipgrad=10000 ##### 
                torch.nn.utils.clip_grad_norm_(net.module.parameters(), clipgrad) # clipgrad=10000
                
                # update parameters
                optimizer.step()

                #########################################################################

                if (iteration % 200 == 0) or (batch_index==1):
                    precision = net.module.evl_acc(logits_new, y)

                    now_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
                    ologs = (
                        '{now}: T:{task}/{tasks}| Ep:{epoch}/{epochs}| lr:{lr:.2f} | '
                        'ac:{prec:.0f}%|'
                        'cls:{ce_loss:.2f}, key:{key_loss:.2f}, ewc:{weight_loss_val}, lwf:{loss_lwf_val:.2f}, lwm:{loss_lwm_val}, kp:{loss_keep_val:.2f}|'
                        ).format(
                        now=now_time,
                        task=(task+1), tasks=len(train_datasets),
                        epoch=epoch, epochs=epochs_per_task,
                        prec=float(precision*100),
                        ce_loss=float( loss_new.item()),
                        weight_loss_val=float( weight_loss_val),
                        loss_lwf_val=float( loss_lwf_val),
                        loss_lwm_val=float( loss_lwm_val),
                        loss_keep_val=float( loss_keep_val),
                        key_loss=float( loss_key_val ),
                        lr = optimizer.param_groups[0]['lr'],
                        )
                    data_stream.set_description(ologs)

            ######################################################
            # Training bias correction layers (BiC-based approach)
            ######################################################
            # if (task>0 and is_BC and epoch > 40  and epoch%10==0): utils_BC.cmp_bias_AMS_r7(net.module, task, batch_size_lite, total_num_class, x_keys, y_keys, criterion_adtv)         

            ######################################################
            if is_BC:
                precs_all, precs_selective, cm_trgt_all, num_slctv = utils_eval.val_selective_bc(net, test_datasets, test_size, batch_size, delete_class, list(range(total_num_class)), cuda, task, class_label_offset, is_dataloader, mode, num_class_per_task)
                # *** Restore the calculated bias (alpha and beta) ***
                net.module.bias_layers[task].alpha.data = torch.ones(1, requires_grad=False, device="cuda")
                net.module.bias_layers[task].beta.data  = torch.zeros(1, requires_grad=False, device="cuda") 
            else:
                if is_NCM:
                    precs_all, precs_selective, cm_trgt_all, num_slctv = utils_eval.val_selective(net, test_datasets, test_size, batch_size, delete_class, list(range(total_num_class)), cuda, task, class_label_offset, is_dataloader, mode, x_keys_org, y_keys_org)
                else:
                    precs_all, precs_selective, cm_trgt_all, num_slctv = utils_eval.val_selective(net, test_datasets, test_size, batch_size, delete_class, list(range(total_num_class)), cuda, task, class_label_offset, is_dataloader, mode)

            acc_key = utils_eval.eval_keys(net.module, task, batch_size_lite, total_num_class, x_keys, y_keys)

            if task==0:
                prec_trgt_old = 0
                prec_non_trgt_old = 0
            else:
                prec_trgt_old = float(precs_selective[0][0])
                prec_non_trgt_old = float(precs_selective[0][1])

            now_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
            outlogs = (
                '{now}: T:{task}/{tasks}| Ep:{epoch}/{epochs}| lr:{lr:.2f} |'
                'ac:{prec:.0f}%| ac_k:{prec_k:.0f}%|'
                'Aave {Aave:.0f}%, Fd {Fd:.0f}%|'
                'Now: {prec_trgt:.0f}%, {prec_non_trgt:.0f}%| Prv: {prec_trgt_old:.0f}%, {prec_non_trgt_old:.0f}%|'
                'cls:{ce_loss:.1f}, key:{key_loss:.1f}, ewc:{weight_loss_val:.1f}, lwf:{loss_lwf_val:.1f}, lwm:{loss_lwm_val:.1f}, kp:{loss_keep_val:.1f}|'
                'H:{H:.3f}, H2:{H2:.3f}|'
                ).format(
                now=now_time,
                task=(task+1), tasks=len(train_datasets),
                epoch=epoch, epochs=epochs_per_task,
                prec=float(precision*100),
                prec_k=float(acc_key*100),
                prec_trgt = float(precs_selective[task][0]*100),
                prec_non_trgt = float(precs_selective[task][1]*100),
                prec_trgt_old = float(prec_trgt_old*100),
                prec_non_trgt_old = float(prec_non_trgt_old*100),
                ce_loss=float( loss_new),
                weight_loss_val=float( weight_loss_val),
                loss_lwf_val=float( loss_lwf_val),
                loss_lwm_val=float( loss_lwm_val),
                loss_keep_val=float( loss_keep_val),
                key_loss=float( loss_key_val ),
                H=float( H ),
                H2=float( H2 ),
                lr = optimizer.param_groups[0]['lr'],
                Aave = A_ave*100, Fd = F_del*100,
                )
                # 'Ak:{Ak:.0f}%, Aave {Aave:.0f}%/ Fk: {Fk:.0f}%, Fd {Fd:.0f}%|'

            print(outlogs)
            # with open(logfile, mode='a') as f: f.write(outlogs+'\n')

            #------------------------------
            acc_all = np.array([precs_all]) if len(iter_p)<1 else np.append(acc_all, np.array([precs_all]), axis=0)
            acc_selective = [np.array(precs_selective)]  if len(iter_p)<1 else np.append(acc_selective, [np.array(precs_selective)], axis=0)
            num_selective = [np.array(num_slctv)]  if len(iter_p)<1 else np.append(num_selective, [np.array(num_slctv)], axis=0)
            loss_p = np.array([[loss_val]])  if len(iter_p)<1 else np.append(loss_p, np.array([[loss_val]]), axis=0)
            loss_p_part = np.array([[loss_val, loss_keep_val, loss_new, weight_loss_val, loss_keep_new_val, loss_rev_val ]])  if len(iter_p)<1 else np.append(loss_p_part,np.array([[loss_val, loss_keep_val, loss_new.item(),  weight_loss_val , loss_keep_new_val, loss_rev_val ]] ),axis=0)

            A_keep, A_ave, F_keep, F_del = utils_eval.eval_metric(acc_selective, num_selective,task)

            A_keep_all = np.array([[A_keep]])  if len(iter_p)<1 else np.append(A_keep_all, np.array([[A_keep]]), axis=0)
            A_ave_all  = np.array([[A_ave]]) if len(iter_p)<1 else np.append(A_ave_all, np.array([[A_ave]]), axis=0)
            F_keep_all = np.array([[F_keep]]) if len(iter_p)<1 else np.append(F_keep_all, np.array([[F_keep]]), axis=0)
            F_del_all  = np.array([[F_del]]) if len(iter_p)<1 else np.append(F_del_all, np.array([[F_del]]), axis=0)
            iter_p = np.array([[iteration/dataset_batches]])  if len(iter_p)<1 else np.append(iter_p, np.array([[  current_task_iteration/dataset_batches + task*epochs_per_task  ]]), axis=0)

            np.save(res_dir_ch+'/acc_all', acc_all)
            np.save(res_dir_ch+'/acc_selective', acc_selective)
            np.save(res_dir_ch+'/num_selective', num_selective)
            np.save(res_dir_ch+'/loss_p', loss_p)
            np.save(res_dir_ch+'/iter_p', iter_p)
            np.save(res_dir_ch+'/loss_p_part', loss_p_part)
            np.save(res_dir_ch+'/A_keep_all', A_keep_all)
            np.save(res_dir_ch+'/A_ave_all', A_ave_all)
            np.save(res_dir_ch+'/F_keep_all', F_keep_all)
            np.save(res_dir_ch+'/F_del_all', F_del_all)

            if len(iter_p)>1:
                utils_eval.show_accuracy_graphs(acc_all, acc_selective, loss_p, iter_p, res_dir, test_datasets, delete_class, total_num_class, str_prm, suffix, loss_p_part, loss_p_name)

            with open(logfile, mode='a') as f: f.write(outlogs+'\n')

            if is_CM>0: 
                H, H2 = utils_eval.compute_entropy_rev(cm_trgt_all, res_dir_ch, task, epoch, delete_class, keep_class, num_class_per_task)
                utils_eval.show_confusion_matrix_total(cm_trgt_all, res_dir_ch, task, epoch, delete_class, total_num_class)

            if is_save==1:
                # save the models
                outfile = res_dir+'/out-{}.cpt'.format(task)
                torch.save({'task': task,
                            'epoch': epoch,
                            'model_state_dict': net.module.state_dict(),
                            'opt_state_dict': optimizer.state_dict(),
                            'acc_all': acc_all,
                            'acc_selective': acc_selective,
                            'num_selective': num_selective,
                            'loss_p': loss_p,
                            'iter_p': iter_p,
                            'loss_p_part': loss_p_part,
                            'A_keep_all': A_keep_all,
                            'A_ave_all': A_ave_all,
                            'F_keep_all': F_keep_all,
                            'F_del_all': F_del_all,
                            'current_task_iteration': current_task_iteration,
                            }, outfile)                

            train_scheduler.step()
        ######################################################
        ######################################################


        #-----------------------------------------------------------------
        test_size_full = 1000000
        ######################################################
        # Training bias correction layers (BiC-based approach)
        ######################################################
        if (task>0 and is_BC):                
            precs_all, precs_selective, cm_trgt_all, num_slctv = utils_eval.val_selective_bc(net, test_datasets, test_size_full, batch_size, delete_class, list(range(total_num_class)), cuda, task, class_label_offset, is_dataloader, mode, num_class_per_task)
            print(precs_selective)
            for k in range(task+1):
                net.module.bias_layers[k].alpha.data = torch.ones(1, requires_grad=False, device="cuda")
                net.module.bias_layers[k].beta.data  = torch.zeros(1, requires_grad=False, device="cuda") 
            utils_BC.cmp_bias_AMS_r7(net.module, task, batch_size_lite, total_num_class, x_keys, y_keys, criterion_adtv, logfile)                

        ######################################################
        # Evaluate acc etc
        ######################################################

        if is_BC:
            # evaluate based on li2m
            precs_all, precs_selective, cm_trgt_all, num_slctv = utils_eval.val_selective_bc(net, test_datasets, test_size_full, batch_size, delete_class, list(range(total_num_class)), cuda, task, class_label_offset, is_dataloader, mode, num_class_per_task)
            print(precs_selective)
            if (task+1) < len(train_datasets):
                for k in range(task+1):
                    net.module.bias_layers[k].alpha.data = torch.ones(1, requires_grad=False, device="cuda")
                    net.module.bias_layers[k].beta.data  = torch.zeros(1, requires_grad=False, device="cuda") 

            for k in range(task + 1):
                print('Task {:d}: alpha={:.5f}, beta={:.5f}'.format(k,net.module.bias_layers[k].alpha.item(), net.module.bias_layers[k].beta.item()))  

        else:
            if is_NCM:
                precs_all, precs_selective, cm_trgt_all, num_slctv = utils_eval.val_selective(net, test_datasets, test_size_full, batch_size, delete_class, list(range(total_num_class)), cuda, task, class_label_offset, is_dataloader, mode, x_keys_org, y_keys_org)
            else:
                precs_all, precs_selective, cm_trgt_all, num_slctv = utils_eval.val_selective(net, test_datasets, test_size_full, batch_size, delete_class, list(range(total_num_class)), cuda, task, class_label_offset, is_dataloader, mode)

        acc_all = np.append(acc_all, np.array([precs_all]), axis=0)
        acc_selective = np.append(acc_selective, [np.array(precs_selective)], axis=0)
        num_selective = np.append(num_selective, [np.array(num_slctv)], axis=0)
        iter_p = np.append(iter_p, np.array([[  current_task_iteration/dataset_batches + task*epochs_per_task  ]]), axis=0)
        loss_p = np.append(loss_p, np.array([[loss_val]]), axis=0)
        loss_p_part = np.append(loss_p_part, np.array([[loss_val, loss_keep_val, loss_new,  weight_loss_val , loss_keep_new_val, loss_rev_val ]] ),axis=0 )

        A_keep, A_ave, F_keep, F_del = utils_eval.eval_metric(acc_selective, num_selective,task)

        A_keep_all = np.append(A_keep_all, np.array([[A_keep]]), axis=0)
        A_ave_all  = np.append(A_ave_all, np.array([[A_ave]]), axis=0)
        F_keep_all = np.append(F_keep_all, np.array([[F_keep]]), axis=0)
        F_del_all  = np.append(F_del_all, np.array([[F_del]]), axis=0)

        np.save(res_dir_ch+'/acc_all', acc_all)
        np.save(res_dir_ch+'/acc_selective', acc_selective)
        np.save(res_dir_ch+'/num_selective', num_selective)
        np.save(res_dir_ch+'/loss_p', loss_p)
        np.save(res_dir_ch+'/iter_p', iter_p)
        np.save(res_dir_ch+'/loss_p_part', loss_p_part)
        np.save(res_dir_ch+'/A_keep_all', A_keep_all)
        np.save(res_dir_ch+'/A_ave_all', A_ave_all)
        np.save(res_dir_ch+'/F_keep_all', F_keep_all)
        np.save(res_dir_ch+'/F_del_all', F_del_all)

        if len(iter_p)>1: utils_eval.show_accuracy_graphs(acc_all, acc_selective, loss_p, iter_p, res_dir, test_datasets, delete_class, total_num_class, str_prm, suffix, loss_p_part, loss_p_name)
        if is_CM>0: 
            H, H2 = utils_eval.compute_entropy_rev(cm_trgt_all, res_dir_ch, task, epoch, delete_class, keep_class, num_class_per_task)
            utils_eval.show_confusion_matrix_total(cm_trgt_all, res_dir_ch, task, epoch, delete_class, total_num_class)


        now_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        outlogs = (
            '{now}: T:{task}/{tasks}| Ep:{epoch}/{epochs}| lr:{lr:.2f} |'
            'ac:{prec:.0f}%|'
            'Aave {Aave:.0f}%, Fd {Fd:.0f}%|'
            'Now: {prec_trgt:.0f}%, {prec_non_trgt:.0f}%| Prv: {prec_trgt_old:.0f}%, {prec_non_trgt_old:.0f}%|'
            'cls:{ce_loss:.2f}, key:{key_loss:.2f}, ewc:{weight_loss_val:.2f}, lwf:{loss_lwf_val:.2f}, lwm:{loss_lwm_val:.2f}, kp:{loss_keep_val:.2f}|'
            'H:{H:.3f}, H2:{H2:.3f}|'
            ).format(
            now=now_time,
            task=(task+1), tasks=len(train_datasets),
            epoch=epoch, epochs=epochs_per_task,
            prec=float(precision*100),
            prec_trgt = float(precs_selective[task][0]*100),
            prec_non_trgt = float(precs_selective[task][1]*100),
            prec_trgt_old = float(prec_trgt_old*100),
            prec_non_trgt_old = float(prec_non_trgt_old*100),
            ce_loss=float( loss_new),
            weight_loss_val=float( weight_loss_val),
            loss_lwf_val=float( loss_lwf_val),
            loss_lwm_val=float( loss_lwm_val),
            loss_keep_val=float( loss_keep_val),
            key_loss=float( loss_key_val ),
            lr = optimizer.param_groups[0]['lr'],
            Aave = A_ave*100, Fd = F_del*100,
            H=float( H ),
            H2=float( H2 ),
            )
        print(outlogs)
        with open(logfile, mode='a') as f: f.write(outlogs+'\n')


        #####################################
        ##### compute Regs
        #####################################
        if is_dataloader==1:
            ##### EWC
            if is_reg == 'EWC':
                if (lmd_weight_loss>0) and  ((task+1) < len(train_datasets)):
                    batch_fisher = 128
                    fisher_pos = utils_Regs.Comp_Fisher_Diag(net.module, train_dataset, fisher_est_sample, batch_fisher, delete_class, float(1),  task, class_label_offset[task], num_class_per_task, total_num_class, is_BC=is_BC)
                    fisher_neg = utils_Regs.Comp_Fisher_Diag(net.module, train_dataset, fisher_est_sample, batch_fisher, delete_class, float(-1), task, class_label_offset[task], num_class_per_task, total_num_class, is_BC=is_BC)
                    if is_ewcp:
                        net.module.consolidate_selective_plus( fisher_pos, fisher_neg, task)
                    else:
                        net.module.consolidate_selective( fisher_pos, fisher_neg, task)

            ##### MAS
            if is_reg == 'MAS':
                if (lmd_weight_loss>0) and  ((task+1) < len(train_datasets)):
                    batch_fisher = 128
                    fisher_pos = utils_Regs.Est_MAS(net.module, train_dataset, fisher_est_sample, batch_fisher, delete_class, float(1), task, class_label_offset[task], num_class_per_task, is_BC=is_BC)
                    if is_ewcp:
                        net.module.consolidate_selective_plus( fisher_pos, fisher_pos, task)
                    else:
                        net.module.consolidate_selective( fisher_pos, fisher_pos, task)
        else:
            if is_reg == 'EWC':
                if (lmd_weight_loss>0) and  ((task+1) < len(train_datasets)):
                    batch_fisher = 128
                    fisher_pos = utils_Regs.Comp_Fisher_Diag_dataset(net.module, train_dataset, fisher_est_sample, batch_fisher, delete_class, float(1),  task, class_label_offset[task], num_class_per_task, total_num_class, is_BC=is_BC)
                    fisher_neg = utils_Regs.Comp_Fisher_Diag_dataset(net.module, train_dataset, fisher_est_sample, batch_fisher, delete_class, float(-1), task, class_label_offset[task], num_class_per_task, total_num_class, is_BC=is_BC)
                    if is_ewcp:
                        net.module.consolidate_selective_plus( fisher_pos, fisher_neg, task)
                    else:
                        net.module.consolidate_selective( fisher_pos, fisher_neg, task)

            ##### MAS
            if is_reg == 'MAS':
                if (lmd_weight_loss>0) and  ((task+1) < len(train_datasets)):
                    batch_fisher = 128
                    fisher_pos = utils_Regs.Est_MAS_dataset(net.module, train_dataset, fisher_est_sample, batch_fisher, delete_class, float(1), task, class_label_offset[task], num_class_per_task, is_BC=is_BC)
                    if is_ewcp:
                        net.module.consolidate_selective_plus( fisher_pos, fisher_pos, task)
                    else:
                        net.module.consolidate_selective( fisher_pos, fisher_pos, task)

        epoch_terminated = 0


