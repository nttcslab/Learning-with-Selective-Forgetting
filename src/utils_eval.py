import os
import os.path
import shutil
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import init
# from src.data import DATASET_CONFIGS
from tqdm import tqdm
# import math as ma
# import tensorboardX as tbx
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import plot_confusion_matrix,  confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
import src.utils as utils

from PIL import Image
import random
import cv2

from math import log, e

def sample_keys_batch(x_keys, y_keys, num_batch):
    idx = torch.randint(0, len(y_keys), (num_batch, ) )
    return x_keys[idx], y_keys[idx]


def eval_keys(net, task, batch_size_lite, total_num_class, x_keys, y_keys):
    net.eval()

    # Loop epochs
    acc_key, num = 0, 0
    loss_BiC = 0
    for tsk in range(task+1):
        x_tmp, y_tmp = sample_keys_batch(x_keys[tsk], y_keys[tsk], batch_size_lite)
        x_tmp = x_tmp.float().cuda()
        y_tmp = y_tmp.cuda()

        logits_BiC = []
        for tsk2 in range(task+1):
            logits_BiC_tmp,_ = net(x_tmp, tsk2)
            logits_BiC.append(logits_BiC_tmp)
        logits_BiC = torch.cat(logits_BiC,dim=1).view(len(y_tmp),-1)

        _, pred = logits_BiC.max(1)
        acc_key += (pred==y_tmp).sum()
        num += len(y_tmp)

    net.train()           
    return acc_key/num

#---------------------------------
def out_images(dataset, res_dir_ch,cuda,task,is_dataloader):
    batch_size=128
    if is_dataloader==1:
        data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
        for x, y in data_loader:
            for k in range(10):
                img = x[k].cpu().permute(1,2,0)
                img = 255*(img-img.min())/(img.max()-img.min()+0.001)
                Image.fromarray(np.uint8( img.squeeze() )).save(  (res_dir_ch+'/sample_{}_{}.png'.format(k,task))  )
            break
    else:
        for x, y in dataset:
            for k in range(10):
                img = x[k].cpu().permute(1,2,0)
                img = 255*(img-img.min())/(img.max()-img.min()+0.001)
                Image.fromarray(np.uint8( img.squeeze() )).save(  (res_dir_ch+'/sample_{}_{}.png'.format(k,task))  )
            break


#---------------------------------
# For Multi-head multi-task, and multi-heand-head incremental-task 
def val_selective(model, datasets, test_size, batch_size,  delete_class, y_list, cuda, task_now, class_label_offset, is_dataloader, mode, x_keys_org=None, y_keys_org=None):

    precs = []
    precs_selective = []
    num_selective = []
    cm_label_all = []

    for task in range(len(datasets)):
        if task <= task_now:
            if mode=='MH_MT':
                precs_tmp, precs_slctv_tmp, cm_label_tmp, num_slctv_tmp = \
                    val_selective_core(
                        model, datasets[task], test_size, batch_size, cuda, delete_class, y_list, class_label_offset[task], task, is_dataloader, x_keys_org, y_keys_org
                            )
            else:
                precs_tmp, precs_slctv_tmp, cm_label_tmp, num_slctv_tmp = \
                    val_selective_inc_core(
                        model, datasets[task], test_size, batch_size, cuda, delete_class, y_list, class_label_offset, task_now, is_dataloader, x_keys_org, y_keys_org
                            )
        else:
            precs_tmp = 0
            precs_slctv_tmp = [0, 0]
            num_slctv_tmp = [0, 0]
            cm_label_tmp = []

        precs.append(precs_tmp)
        precs_selective.append(precs_slctv_tmp)
        num_selective.append(num_slctv_tmp)
        cm_label_all.append(cm_label_tmp)

    return precs, precs_selective, cm_label_all, num_selective

#---------------------------------
# Algorithm 1: iCaRL NCM Classify
def classify_NCM(model, task, features, x_keys_org, y_keys_org, is_inc=0):
    if is_inc==0:
        x_keys = x_keys_org[task].squeeze()
        y_keys = y_keys_org[task].squeeze()
    else:
        x_keys = torch.stack(x_keys_org[:task+1])
        x_keys = x_keys.view(-1, x_keys.size(2), x_keys.size(3), x_keys.size(4))
        y_keys = torch.stack(y_keys_org[:task+1]).view(-1)

    # expand means to all batch images
    _, feats_keys = model(x_keys, task) # class, dim_fea

    means = torch.stack([feats_keys] * features.shape[0]) # batch_size, class, dimfea
    means = means.transpose(1, 2) # batch_size, dimfea, class

    features = features.unsqueeze(2) # batch_size, dimfea, 1
    features = features.expand_as(means) # batch_size, dimfea, class

    dists = (features - means).pow(2).sum(1).squeeze() # batch_size, class

    _, pred_idx = dists.min(1)
    predicted = y_keys[pred_idx]
    return predicted

#---------------------------------
def val_selective_core(
    model, dataset, test_size, batch_size, cuda, delete_class, y_list, class_label_offset, task, is_dataloader, x_keys_org=None, y_keys_org=None):

    # gen non-target class
    non_delete_class = y_list.copy()
    for k in y_list:
        for t in delete_class:
            if k==t:
                non_delete_class.remove(k)

    total_tested_delete = 0
    total_correct_delete = 0
    total_tested_non_delete = 0
    total_correct_non_delete = 0
    y_true_delete = []
    y_pred_delete = []
    y_true_non_delete = []
    y_pred_non_delete = []

    if is_dataloader:
        data_eval = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    else:
        data_eval = dataset

    model.eval()
    for x, y in data_eval:
        if (total_tested_delete +  total_tested_non_delete) >= test_size: break

        # ***If not a detaloder, add the offset of the class (Cub, STN, ImageNet)***
        if is_dataloader is not 1: y = y + class_label_offset

        if x_keys_org==None:
            logits, _ = model(x, task)
            _, predicted = logits.max(1)
        else:
            _, features = model(x, task)
            predicted =  classify_NCM(model, task, features, x_keys_org, y_keys_org, is_inc=0) 
            predicted = predicted - class_label_offset

        predicted = predicted + class_label_offset

        y_cpu = y.data.to('cpu').detach().clone().numpy()

        idx_delete = [k for k, y_ in enumerate(y_cpu) for d in delete_class if y_ == d ]
        y_delete = [y_cpu[x_]  for x_ in idx_delete]

        idx_non_delete = [k for k, y_ in enumerate(y_cpu) for d in non_delete_class if y_ == d ]
        y_non_delete =  [y_cpu[x_]  for x_ in idx_non_delete]

        y_true_delete = np.concatenate([y_true_delete, y_delete], 0)
        y_true_non_delete = np.concatenate([y_true_non_delete, y_non_delete], 0)

        predicted_delete = predicted[idx_delete].to('cpu').detach().clone().numpy()
        predicted_non_delete = predicted[idx_non_delete].to('cpu').detach().clone().numpy()

        y_pred_delete = np.concatenate([y_pred_delete, predicted_delete], 0)
        y_pred_non_delete = np.concatenate([y_pred_non_delete, predicted_non_delete], 0)

        # update statistics.
        total_correct_delete += int((predicted_delete == y_delete).sum())
        total_correct_non_delete += int((predicted_non_delete == y_non_delete ).sum())

        total_tested_delete += len(predicted_delete)
        total_tested_non_delete += len(predicted_non_delete)

    model.train()

    prec_delete = total_correct_delete / (total_tested_delete + 0.0001)
    prec_non_delete = total_correct_non_delete / (total_tested_non_delete + 0.0001)
    prec = (total_correct_delete+total_correct_non_delete)/ (total_tested_delete + total_tested_non_delete + 0.0001)

    y_true = np.concatenate([y_true_delete, y_true_non_delete],0)
    y_pred = np.concatenate([y_pred_delete, y_pred_non_delete],0)

    return prec, [prec_delete, prec_non_delete], [y_true, y_pred], [total_tested_delete, total_tested_non_delete]


#---------------------------------
def val_selective_inc_core(
    model, dataset, test_size, batch_size, cuda, delete_class, y_list, class_label_offset, task_now, is_dataloader, x_keys_org=None, y_keys_org=None):
    # class_label_offset = [0,10,20,...]

    # gen non-target class
    non_delete_class = y_list.copy()
    for k in y_list:
        for t in delete_class:
            if k==t: non_delete_class.remove(k)

    total_tested_delete = 0
    total_correct_delete = 0
    total_tested_non_delete = 0
    total_correct_non_delete = 0
    y_true_delete = []
    y_pred_delete = []
    y_true_non_delete = []
    y_pred_non_delete = []

    if is_dataloader:
        data_eval = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    else:
        data_eval = dataset

    model.eval()
    for x, y in data_eval:
        if (total_tested_delete +  total_tested_non_delete) >= test_size: break

        if x_keys_org==None:
            logits = []
            for tsk in range(task_now+1):
                logits_tmp,_ = model(x,tsk)
                logits.append(logits_tmp)
            logits = torch.cat(logits,dim=1).view(len(y),-1)
            _, predicted = logits.max(1)
        else:
            _, features  = model(x, task_now)
            predicted =  classify_NCM(model, task_now, features, x_keys_org, y_keys_org, is_inc=1) 



        y_cpu = y.data.to('cpu').detach().clone().numpy()

        idx_delete = [k for k, y_ in enumerate(y_cpu) for d in delete_class if y_ == d ]
        y_delete = [y_cpu[x_]  for x_ in idx_delete]

        idx_non_delete = [k for k, y_ in enumerate(y_cpu) for d in non_delete_class if y_ == d ]
        y_non_delete =  [y_cpu[x_]  for x_ in idx_non_delete]

        y_true_delete = np.concatenate([y_true_delete, y_delete], 0)
        y_true_non_delete = np.concatenate([y_true_non_delete, y_non_delete], 0)

        predicted_delete = predicted[idx_delete].to('cpu').detach().clone().numpy()
        predicted_non_delete = predicted[idx_non_delete].to('cpu').detach().clone().numpy()

        y_pred_delete = np.concatenate([y_pred_delete, predicted_delete], 0)
        y_pred_non_delete = np.concatenate([y_pred_non_delete, predicted_non_delete], 0)

        # update statistics.
        total_correct_delete += int((predicted_delete == y_delete).sum())
        total_correct_non_delete += int((predicted_non_delete == y_non_delete ).sum())

        total_tested_delete += len(predicted_delete)
        total_tested_non_delete += len(predicted_non_delete)

    model.train()

    prec_delete = total_correct_delete / (total_tested_delete + 0.0001)
    prec_non_delete = total_correct_non_delete / (total_tested_non_delete + 0.0001)
    prec = (total_correct_delete+total_correct_non_delete)/ (total_tested_delete + total_tested_non_delete + 0.0001)

    y_true = np.concatenate([y_true_delete, y_true_non_delete],0)
    y_pred = np.concatenate([y_pred_delete, y_pred_non_delete],0)

    return prec, [prec_delete, prec_non_delete], [y_true, y_pred], [total_tested_delete, total_tested_non_delete]


#---------------------------------
# For Multi-head multi-task, and multi-heand-head incremental-task 
# For Il2m
# def val_selective_bc(model, datasets, test_size, batch_size,  delete_class, y_list, cuda, task_now, class_label_offset, is_dataloader, mode, num_class_per_task, init_classes_means,current_classes_means,models_confidence):
def val_selective_bc(model, datasets, test_size, batch_size,  delete_class, y_list, cuda, task_now, class_label_offset, is_dataloader, mode, num_class_per_task):

    precs = []
    precs_selective = []
    num_selective = []
    cm_label_all = []

    for task in range(len(datasets)):
        if task <= task_now:
            precs_tmp, precs_slctv_tmp, cm_label_tmp, num_slctv_tmp = \
                val_selective_inc_core_bc(
                    model, datasets[task], test_size, batch_size, cuda, delete_class, y_list, class_label_offset[task], num_class_per_task, task_now, is_dataloader
                        )
        else:
            precs_tmp = 0
            precs_slctv_tmp = [0, 0]
            num_slctv_tmp = [0, 0]
            cm_label_tmp = []

        precs.append(precs_tmp)
        precs_selective.append(precs_slctv_tmp)
        num_selective.append(num_slctv_tmp)
        cm_label_all.append(cm_label_tmp)

    return precs, precs_selective, cm_label_all, num_selective

#----------------------------------
def val_selective_inc_core_bc(
    model, dataset, test_size, batch_size, cuda, delete_class, y_list, class_label_offset, num_class_per_task, task_now, is_dataloader):
    # class_label_offset = [0,10,20,...]

    # gen non-target class
    non_delete_class = y_list.copy()
    for k in y_list:
        for t in delete_class:
            if k==t: non_delete_class.remove(k)

    total_tested_delete = 0
    total_correct_delete = 0
    total_tested_non_delete = 0
    total_correct_non_delete = 0
    y_true_delete = []
    y_pred_delete = []
    y_true_non_delete = []
    y_pred_non_delete = []

    if is_dataloader:
        data_eval = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    else:
        data_eval = dataset

    model.eval()

    #-------------------------------------------------------
    for x, y in data_eval:
        if (total_tested_delete +  total_tested_non_delete) >= test_size: break
        # If not a detaloder, add the offset of the class (Cub, STN, ImageNet)
        if is_dataloader is not 1: y = y + class_label_offset

        logits = []
        for tsk in range(task_now+1):
            logits_tmp,_ = model(x,tsk)
            logits_tmp = model.module.bias_layers[tsk](logits_tmp)

            logits.append(logits_tmp)
        logits = torch.cat(logits,dim=1).view(len(y),-1)

        _, predicted = logits.max(1)

        #------------------------------------------------
        y_cpu = y.data.to('cpu').detach().clone().numpy()

        idx_delete = [k for k, y_ in enumerate(y_cpu) for d in delete_class if y_ == d ]
        y_delete = [y_cpu[x_]  for x_ in idx_delete]

        idx_non_delete = [k for k, y_ in enumerate(y_cpu) for d in non_delete_class if y_ == d ]
        y_non_delete =  [y_cpu[x_]  for x_ in idx_non_delete]

        y_true_delete = np.concatenate([y_true_delete, y_delete], 0)
        y_true_non_delete = np.concatenate([y_true_non_delete, y_non_delete], 0)

        predicted_delete = predicted[idx_delete].to('cpu').detach().clone().numpy()
        predicted_non_delete = predicted[idx_non_delete].to('cpu').detach().clone().numpy()

        y_pred_delete = np.concatenate([y_pred_delete, predicted_delete], 0)
        y_pred_non_delete = np.concatenate([y_pred_non_delete, predicted_non_delete], 0)

        # update statistics.
        total_correct_delete += int((predicted_delete == y_delete).sum())
        total_correct_non_delete += int((predicted_non_delete == y_non_delete ).sum())

        total_tested_delete += len(predicted_delete)
        total_tested_non_delete += len(predicted_non_delete)

    model.train()

    prec_delete = total_correct_delete / (total_tested_delete + 0.0001)
    prec_non_delete = total_correct_non_delete / (total_tested_non_delete + 0.0001)
    prec = (total_correct_delete+total_correct_non_delete)/ (total_tested_delete + total_tested_non_delete + 0.0001)

    y_true = np.concatenate([y_true_delete, y_true_non_delete],0)
    y_pred = np.concatenate([y_pred_delete, y_pred_non_delete],0)

    return prec, [prec_delete, prec_non_delete], [y_true, y_pred], [total_tested_delete, total_tested_non_delete]



#----------------------------------
def __UNUSED__val_selective_inc_core_bc_li2m(
    model, dataset, test_size, batch_size, cuda, delete_class, y_list, class_label_offset, num_class_per_task, task_now, is_dataloader, \
    init_classes_means,current_classes_means,models_confidence):
    # class_label_offset = [0,10,20,...]

    # gen non-target class
    non_delete_class = y_list.copy()
    for k in y_list:
        for t in delete_class:
            if k==t: non_delete_class.remove(k)

    total_tested_delete = 0
    total_correct_delete = 0
    total_tested_non_delete = 0
    total_correct_non_delete = 0
    y_true_delete = []
    y_pred_delete = []
    y_true_non_delete = []
    y_pred_non_delete = []

    if is_dataloader:
        data_eval = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    else:
        data_eval = dataset

    model.eval()

    #-------------------------------------------------------
    rectified_msk_1d = None
    for x, y in data_eval:
        if (total_tested_delete +  total_tested_non_delete) >= test_size: break

        logits = []
        for tsk in range(task_now+1):
            logits_tmp,_ = model(x,tsk)
            logits.append(logits_tmp)
        logits = torch.cat(logits,dim=1).view(len(y),-1)
        rectified_outputs = torch.nn.functional.log_softmax(logits, dim=1)

        old_classes_number = class_label_offset[task_now]

        if old_classes_number and rectified_msk_1d==None:
            rectified_msk_1d = torch.ones(1, logits.size(1))
            for o in range(old_classes_number):
                o_task = int(( (torch.tensor(class_label_offset)+num_class_per_task) <= o).sum())
                rectified_msk_1d[0, o] = (init_classes_means[o] / current_classes_means[o]) * (models_confidence[-1] / models_confidence[o_task])

        if old_classes_number: rectified_outputs *= rectified_msk_1d.expand(len(y),-1) .cuda()
    
        _, pred = rectified_outputs.max(1)
        #------------------------------------------------
        y_cpu = y.data.to('cpu').detach().clone().numpy()

        idx_delete = [k for k, y_ in enumerate(y_cpu) for d in delete_class if y_ == d ]
        y_delete = [y_cpu[x_]  for x_ in idx_delete]

        idx_non_delete = [k for k, y_ in enumerate(y_cpu) for d in non_delete_class if y_ == d ]
        y_non_delete =  [y_cpu[x_]  for x_ in idx_non_delete]

        y_true_delete = np.concatenate([y_true_delete, y_delete], 0)
        y_true_non_delete = np.concatenate([y_true_non_delete, y_non_delete], 0)

        pred_delete = pred[idx_delete].to('cpu').detach().clone().numpy()
        pred_non_delete = pred[idx_non_delete].to('cpu').detach().clone().numpy()

        y_pred_delete = np.concatenate([y_pred_delete, pred_delete], 0)
        y_pred_non_delete = np.concatenate([y_pred_non_delete, pred_non_delete], 0)

        # update statistics.
        total_correct_delete += int((pred_delete == y_delete).sum())
        total_correct_non_delete += int((pred_non_delete == y_non_delete ).sum())

        total_tested_delete += len(pred_delete)
        total_tested_non_delete += len(pred_non_delete)

    model.train()

    prec_delete = total_correct_delete / (total_tested_delete + 0.0001)
    prec_non_delete = total_correct_non_delete / (total_tested_non_delete + 0.0001)
    prec = (total_correct_delete+total_correct_non_delete)/ (total_tested_delete + total_tested_non_delete + 0.0001)

    y_true = np.concatenate([y_true_delete, y_true_non_delete],0)
    y_pred = np.concatenate([y_pred_delete, y_pred_non_delete],0)

    return prec, [prec_delete, prec_non_delete], [y_true, y_pred], [total_tested_delete, total_tested_non_delete]

def show_confusion_matrix_total(cm_label_all, res_dir_ch, task, epoch, delete_class, total_num_class):
    # y_list = list(range(total_num_class))
    # y_list = [str(n) for n in y_list]

    for ts in range(len(cm_label_all)):
        if len(cm_label_all[ts])>0:
            #--------------------
            title = ( 'Confusion matrix \n' )
            label_true, label_pred = cm_label_all[ts]       

            label_true = label_true.tolist()
            label_pred = label_pred.tolist()
            label_true = label_true + [k for k in range(total_num_class)] 
            label_pred = label_pred + [k for k in range(total_num_class)]

            cm_y = confusion_matrix( label_true, label_pred)

            # substract 1 for diag elems.
            for i in range(cm_y.shape[0]):
                cm_y[i][i] -= 1
            
            # rescale as log(1+x)
            cm_y = np.log(1+cm_y)

            sns.heatmap(cm_y, annot=False, fmt="d")
            plt.title(title)
            plt.savefig( res_dir_ch+'/CM_Task{ts}_TaskALL{task}.png'.format(ts=ts, task = task) )
            plt.close() 

def compute_entropy(cm_label_all, res_dir_ch, task, epoch, delete_class, keep_class, num_class_per_task):
    base = e
    for ts in range(1):
        keep_class_tmp = [k+ts*num_class_per_task for k in range(num_class_per_task) if k+ts*num_class_per_task in keep_class]
        if len(cm_label_all[ts])>0:
            #--------------------
            title = ( 'Confusion matrix \n' )
            label_true, label_pred = cm_label_all[ts]       

            label_true = label_true.tolist()
            label_pred = label_pred.tolist()
            label_true = label_true + [k for k in range(num_class_per_task)] 
            label_pred = label_pred + [k for k in range(num_class_per_task)]

            cm_y = confusion_matrix( label_true, label_pred)
            for i in range(cm_y.shape[0]):
                cm_y[i][i] -= 1

            num_keep = len(keep_class_tmp)
            vech = torch.zeros(len(delete_class))
            vec2h = torch.zeros(len(delete_class))
            for k in delete_class:
                vec = torch.tensor(cm_y[k]).clone().detach()
                num_keep = torch.tensor(vec).clone().detach().sum()+1E-10
                vec2 = torch.tensor([vec[l] for l in keep_class_tmp])
                vec2p = vec2/num_keep                
                vecp = vec/num_keep                
                for p in vecp:  vech[k]  -= p * log(p+1E-10, base)
                for p in vec2p: vec2h[k] -= p * log(p+1E-10, base)

    return vech.mean(), vec2h.mean()

def compute_entropy_rev(cm_label_all, res_dir_ch, task, epoch, delete_class, keep_class, num_class_per_task):
    base = e
    vech_mean = []
    vec2h_mean = []

    if task==0:
        vech_mean = 0
        vec2h_mean = 0
    else:
        for ts in range(task):
            keep_class_tmp = [k for k in range(sum(num_class_per_task[:ts+1])) if k in keep_class]
            delete_class_tmp = [k for k in range(sum(num_class_per_task[:ts+1])) if k in delete_class]
            if len(cm_label_all[ts])>0:
                #--------------------
                title = ( 'Confusion matrix \n' )
                label_true, label_pred = cm_label_all[ts]       

                label_true = label_true.tolist()
                label_pred = label_pred.tolist()
                label_true = label_true + [k for k in range(sum(num_class_per_task[:task+1])) ] 
                label_pred = label_pred + [k for k in range(sum(num_class_per_task[:task+1])) ]

                cm_y = confusion_matrix( label_true, label_pred)
                for i in range(cm_y.shape[0]):
                    cm_y[i][i] -= 1

                num_keep = len(keep_class_tmp)+1E-10
                vech = torch.zeros(len(delete_class_tmp))
                vec2h = torch.zeros(len(delete_class_tmp))

                for n,k in enumerate(delete_class_tmp):
                    vec = torch.tensor(cm_y[k]).clone().detach()
                    num_keep = vec.sum()+1E-10
                    vec2 = torch.tensor([vec[l] for l in keep_class_tmp])
                    vec2p = vec2/num_keep                
                    vecp = vec/num_keep                
                    for p in vecp:  vech[n]  -= p * log(p+1E-10, base)
                    for p in vec2p: vec2h[n] -= p * log(p+1E-10, base)
            vech_mean.append(vech.mean())
            vec2h_mean.append(vec2h.mean())

        vech_mean = torch.tensor(vech_mean).mean()
        vec2h_mean = torch.tensor(vec2h_mean).mean()

    return vech_mean, vec2h_mean


def show_confusion_matrix_total_face(cm_label_all, res_dir_ch, task, epoch, delete_class, total_num_class):
    for ts in range(len(cm_label_all)):
        if len(cm_label_all[ts])>0:
            #--------------------
            title = ( 'Confusion matrix \n')
            label_true, label_pred = cm_label_all[ts]       

            label_true = label_true.tolist()
            label_pred = label_pred.tolist()
            label_true = label_true + [k for k in range(total_num_class)] 
            label_pred = label_pred + [k for k in range(total_num_class)]

            cm_y = confusion_matrix( label_true, label_pred)

            for i in range(cm_y.shape[0]):
                cm_y[i][i] -= 1
            sns.heatmap(cm_y, annot=False, fmt="d")
            plt.title(title)
            plt.savefig( res_dir_ch+'/CM_Task{ts}_TaskALL{task}.png'.format(ts=ts, task = task) )
            plt.close() 


#----------------------------------
def eval_metric(acc_selective, num_selective, tsk):

    mat_acc = torch.tensor(acc_selective)
    num_acc = torch.tensor(num_selective)

    mat_acc_fin = mat_acc[mat_acc.size(0)-1]
    mat_num_fin = num_acc[num_acc.size(0)-1]

    mat_acc_max, _ = mat_acc.max(0)
    if tsk>0:
        mat_acc_keep = mat_acc_fin[:tsk,1]
        mat_num_keep = mat_num_fin[:tsk,1]
        mat_num_del  = mat_num_fin[:tsk,0]
        mat_acc_keep_fin = mat_acc_fin[tsk,0]
        mat_num_keep_fin = mat_num_fin[tsk,0]

        A_keep = ((mat_acc_keep*mat_num_keep).sum())/((mat_num_keep).sum())
        A_ave = ((mat_acc_keep*mat_num_keep).sum() + (mat_acc_keep_fin*mat_num_keep_fin).sum()   )/( (mat_num_keep).sum()+(mat_num_keep_fin).sum()   )
        
        mat_F_keep = (mat_acc_max[:tsk,1] - mat_acc_fin[:tsk,1])
        mat_F_del  = (mat_acc_max[:tsk,0] - mat_acc_fin[:tsk,0])
        F_keep = ((mat_F_keep*mat_num_keep).sum())/((mat_num_keep).sum())
        F_del  = ((mat_F_del*mat_num_del).sum())/((mat_num_del).sum())
    else:
        mat_acc_keep = 0
        mat_num_keep = 0
        mat_num_del  = 0
        mat_acc_keep_fin = mat_acc_fin[tsk,0]
        mat_num_keep_fin = mat_num_fin[tsk,0]

        A_keep = 0
        A_ave = ((mat_acc_keep_fin*mat_num_keep_fin).sum()   )/( (mat_num_keep_fin).sum()   )
        
        F_keep = 0
        F_del  = 0
    return A_keep, A_ave, F_keep, F_del


#---------------------------------
def show_accuracy_graphs(acc_all, acc_selective, loss, iters, res_dir, test_datasets, delete_class, total_num_class, suffix, suffix2, loss_part, loss_name):
    acc_all_t = acc_all.T
    acc_selective_t = acc_selective.transpose(2,1,0)
    iter_t = iters.T
    loss_t = loss.T
    loss_part_t = loss_part.T

    y_list = list(range(total_num_class))
    preserve_class = y_list.copy()
    for k in y_list:
        for t in delete_class:
            if k==t:
                preserve_class.remove(k)

    #---------------------------------------------- 
    # plot accuracy 
    #---------------------------------------------- 
    fig = plt.figure(facecolor='skyblue', figsize=(15, 5))
    ax1 = fig.add_axes((0.1, 0.1, 0.4, 0.8), ylim=(0, 1))
    ax2 = fig.add_axes((0.5, 0.1, 0.4, 0.8), sharey=ax1)
    ax2.tick_params(left=False, right=True, labelleft=False, labelright=True)
    fig.text(0, 1, suffix, va='top')
    title = ('precision:\n' )
    # title = ('precision:\n' + 'delete_class='+str(delete_class))
    for t in range(len(test_datasets)):
        Y = acc_selective_t[0][t]
        X = iter_t[0]
        ax1.plot(X,Y)
    ax1.set_title('delete:'+title)
    ax1.set_ylim(0,1)

    title = ('precision:\n' )
    # title = ('precision:\n' + 'delete_class='+str(delete_class))
    for t in range(len(test_datasets)):
        Y = acc_selective_t[1][t]
        X = iter_t[0]
        ax2.plot(X,Y)
    ax2.set_title('preserve:'+title)
    ax2.set_ylim(0,1)
    plt.savefig(res_dir + '/res_acc.png')  
    plt.savefig(res_dir + '/res_acc_{}.png'.format(suffix2))  
    plt.close()                         

    # #---------------------------------------------- 
    # ####------------- plot loss for each term
    # #---------------------------------------------- 
    # title = ('Loss:\n' + str(suffix) + '\n :delete_class='+str(preserve_class))
    # for t in range(len(loss_name)):
    #     Y = loss_part_t[t]
    #     X = iter_t[0]
    #     plt.plot(X,Y, label=loss_name[t])
    # plt.title(title)
    # plt.legend()
    # plt.savefig(res_dir + '/res_loss_part.png')  
    # plt.close()                         


def show_tsne_with_landimg(model, feat, dataset, c_num, fname, suffix, epoch, landimg_old):
    batch_size, sample_size = 64, 2048

    data_loader = utils.get_data_loader(dataset, batch_size)
    feats = []
    labels = []

    model.eval()
    _, _, _  = model(landimg_old.cuda())

    y_land = torch.tensor( [ c_num+k for k in range(landimg_old.size()[0]) ] )
    feat_ = feat.to('cpu').detach().view(landimg_old.size()[0],-1).numpy().copy() 
    labels = y_land
    feats = feat_


    for x, y in data_loader:
        _, _, _  = model(x.cuda())
        y_cpu = y.data.to('cpu').detach().numpy().copy()
        if len(y_cpu) == batch_size:
            feat_ = feat.to('cpu').detach().view(batch_size,-1).numpy().copy() 
            labels = np.concatenate((labels, y_cpu))
            feats = np.concatenate((feats, feat_))
            if feats.shape[0] > sample_size: break        

    perplexity_list = [5, 30, 50]
    title = ( 't-SNE \n' + str(suffix) + '\n  epoch:{epoch}'.format(epoch=epoch)
        )

    _, ax = plt.subplots(1, len(perplexity_list), figsize=(15, 5))
    for p, k in enumerate(perplexity_list):
        feat2d = TSNE(n_components=2,perplexity=k).fit_transform(feats)
        for i in range(c_num+1):
            # _, ax = plt.subplots(1, p+1, len(perplexity_list), figsize=(10, 10))
            target = feat2d[labels == i]
            ax[p].scatter(x=target[:, 0], y=target[:, 1], label=str(i), alpha=0.2)

        for i in range(landimg_old.size()[0]):
            target = feat2d[labels == c_num+i]
            ax[p].scatter(x=target[:, 0], y=target[:, 1], marker='${}$'.format(str(i)),label=str(i), alpha=1, s=70, color='k')
        
        ax[p].set_title(title)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')           
    plt.savefig(fname)
    plt.close() 

    model.train()

    return feat2d

