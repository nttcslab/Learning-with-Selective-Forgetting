from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
#--------
import src.utils as utils
import torch
from sklearn.manifold import TSNE
from torch import autograd, nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.functional import avg_pool2d, relu
from src.utils_BC import BiasLayer

from src.loss_fun import ReverseLayerF

##################################
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet_MH(nn.Module):
    def __init__(self, input_size, output_size, lmd_weight_loss=40, scale=30, num_task=4, \
        block = BasicBlock, num_block =  [2, 2, 2, 2]):

        super(ResNet_MH, self).__init__()

        self.in_channels = 64
        self.r = 0.5

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # bias layer for BiC (see CVPR19)
        self.bias_layers = []

        layer     = [nn.Linear(512 * block.expansion, output_size[i], bias=True) for i in range(num_task)]

        self.w2_mul = nn.ModuleList(layer)
        
        self.input_size = input_size
        self.output_size = output_size
        self.lmd_weight_loss = lmd_weight_loss
        self.scale = scale
        self.dim_fea = 512 * block.expansion

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)


    def forward(self, x, task, is_rev=0):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)        
        logits = output.view(output.size(0), -1)

        logits = F.normalize(logits, p=2, dim=1)
        self.feats = logits.detach().clone()

        w2_weight = self.w2_mul[task]
        w2_norm = w2_weight.weight / torch.norm(w2_weight.weight, dim=1, keepdim=True)
        if is_rev==1: logits = ReverseLayerF.apply(logits)
        logits = torch.matmul(logits, w2_norm.T)

        return logits, self.feats

    def evl_acc(self, logits, labels): 
        self.eval()
        _, y_pred = logits.max(1)
        precision = (y_pred == labels).sum().float() / len(logits)
        self.train()
        return precision

    def consolidate_selective(self, fisher_pos, fisher_neg, task):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_{}_mean'.format(n,task), p.data.clone())
            self.register_buffer('{}_{}_fisher_pos'.format(n,task), fisher_pos[n].data.clone())
            self.register_buffer('{}_{}_fisher_neg'.format(n,task), fisher_neg[n].data.clone())

    def consolidate_selective_plus_ini(self, task):
        # Store current parameters for the next task
        # self.w2_mul[task]
        if task>0:
            for n, p in self.named_parameters():
                if 'w2_mul' in n:
                    n = n.replace('.', '__')
                    for t in range(task):
                        self.register_buffer('{}_{}_mean'.format(n,t), p.data.clone())
                        self.register_buffer('{}_{}_fisher_pos'.format(n,t), p.data.clone()*0)
                        self.register_buffer('{}_{}_fisher_neg'.format(n,t), p.data.clone()*0)
                else:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_mean'.format(n), p.data.clone())
                    self.register_buffer('{}_fisher_pos'.format(n), p.data.clone()*0)
                    self.register_buffer('{}_fisher_neg'.format(n), p.data.clone()*0)


    def consolidate_selective_plus(self, fisher_pos, fisher_neg, task):
        # Store current parameters for the next task
        # self.w2_mul[task]
        for n, p in self.named_parameters():
            if 'w2_mul' in n:
                n = n.replace('.', '__')
                self.register_buffer('{}_{}_mean'.format(n,task), p.data.clone())
                self.register_buffer('{}_{}_fisher_pos'.format(n,task), fisher_pos[n].data.clone())
                self.register_buffer('{}_{}_fisher_neg'.format(n,task), fisher_neg[n].data.clone())
            else:
                n = n.replace('.', '__')
                if task > 0:
                    f_pos_old = getattr(self, '{}_fisher_pos'.format(n))
                    f_neg_old = getattr(self, '{}_fisher_neg'.format(n))
                else:
                    f_pos_old = fisher_pos[n].data.clone()
                    f_neg_old = fisher_neg[n].data.clone()

                self.register_buffer('{}_mean'.format(n), p.data.clone())
                self.register_buffer('{}_fisher_pos'.format(n), self.r * f_pos_old + (1 - self.r ) *  fisher_pos[n].data.clone())
                self.register_buffer('{}_fisher_neg'.format(n), self.r * f_neg_old + (1 - self.r ) * fisher_neg[n].data.clone())

    def ewcp_loss_selective(self, cuda, weight_rev, task):
        try:
            losses = []
            for n, p in self.named_parameters():
                if 'w2_mul' in n:
                    n = n.replace('.', '__')
                    for task_prev in range(task):
                        # retrieve the consolidated mean and fisher information.
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_{}_mean'.format(n,task_prev))
                        fisher_pos = getattr(self, '{}_{}_fisher_pos'.format(n,task_prev))
                        fisher_neg = getattr(self, '{}_{}_fisher_neg'.format(n,task_prev))

                        # wrap mean and fisher in variables.
                        mean = mean.cuda()
                        fisher_diag_weight = torch.clamp( fisher_pos+weight_rev*fisher_neg, min=0)
                        fisher_diag_weight = fisher_diag_weight.cuda()
                        losses.append(( fisher_diag_weight * (p-mean)**2).sum())

                else:
                    # retrieve the consolidated mean and fisher information.
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_mean'.format(n))
                    fisher_pos = getattr(self, '{}_fisher_pos'.format(n))
                    fisher_neg = getattr(self, '{}_fisher_neg'.format(n))
                    # wrap mean and fisher in variables.
                    mean = mean.cuda()
                    fisher_diag_weight = torch.clamp( fisher_pos+weight_rev*fisher_neg, min=0)
                    fisher_diag_weight = fisher_diag_weight.cuda()
                    losses.append(( fisher_diag_weight * (p-mean)**2).sum())
            return (self.lmd_weight_loss/2)*sum(losses)
        except AttributeError:
            print('error')
            return Variable(torch.zeros(1)).cuda()

    def weight_loss_light(self, cuda, weight_rev, task):
        try:
            losses = []
            for n, p in self.named_parameters():
                if 'w2_mul' in n:
                    n = n.replace('.', '__')
                    for task_prev in range(task):
                        # retrieve the consolidated mean and fisher information.
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_{}_mean'.format(n,task_prev))
                        fisher_pos = getattr(self, '{}_{}_fisher_pos'.format(n,task_prev))
                        fisher_neg = getattr(self, '{}_{}_fisher_neg'.format(n,task_prev))

                        # wrap mean and fisher in variables.
                        mean = mean.cuda()
                        fisher_diag_weight = torch.clamp( fisher_pos+weight_rev*fisher_neg, min=0)
                        fisher_diag_weight = fisher_diag_weight.cuda()
                        losses.append(( fisher_diag_weight * (p-mean)**2).sum())

                else:
                    # retrieve the consolidated mean and fisher information.
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_mean'.format(n))
                    fisher_pos = getattr(self, '{}_fisher_pos'.format(n))
                    fisher_neg = getattr(self, '{}_fisher_neg'.format(n))
                    # wrap mean and fisher in variables.
                    mean = mean.cuda()
                    fisher_diag_weight = torch.clamp( fisher_pos+weight_rev*fisher_neg, min=0)
                    fisher_diag_weight = fisher_diag_weight.cuda()
                    losses.append(( fisher_diag_weight * (p-mean)**2).sum())
            return (self.lmd_weight_loss/2)*sum(losses)
        except AttributeError:
            # print('error')
            return Variable(torch.zeros(1)).cuda()

    def ewc_loss_selective(self, cuda, weight_rev, task):
        try:
            losses = []
            for n, p in self.named_parameters():
                for task_prev in range(task):
                    # retrieve the consolidated mean and fisher information.
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_{}_mean'.format(n,task_prev))
                    fisher_pos = getattr(self, '{}_{}_fisher_pos'.format(n,task_prev))
                    fisher_neg = getattr(self, '{}_{}_fisher_neg'.format(n,task_prev))

                    # wrap mean and fisher in variables.
                    mean = mean.cuda()
                    fisher_diag_weight = torch.clamp( fisher_pos+weight_rev*fisher_neg, min=0)
                    fisher_diag_weight = fisher_diag_weight.cuda()
                    losses.append(( fisher_diag_weight * (p-mean)**2).sum())

                    # if torch.any(torch.isnan(torch.tensor(losses)).view(-1)):
                    #     print('EWC_mat_ERROR_{}:'.format(n))
                    #     sys.exit()
                    
            return (self.lmd_weight_loss/2)*sum(losses)
        except AttributeError:
            # print('error')
            return Variable(torch.zeros(1)).cuda()


    def weight_loss(self, cuda, weight_rev, task):
        try:
            losses = []
            for n, p in self.named_parameters():
                for task_prev in range(task):
                    # retrieve the consolidated mean and fisher information.
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_{}_mean'.format(n,task_prev))
                    fisher_pos = getattr(self, '{}_{}_fisher_pos'.format(n,task_prev))
                    fisher_neg = getattr(self, '{}_{}_fisher_neg'.format(n,task_prev))

                    # wrap mean and fisher in variables.
                    mean = mean.cuda()
                    fisher_diag_weight = torch.clamp( fisher_pos+weight_rev*fisher_neg, min=0)
                    fisher_diag_weight = fisher_diag_weight.cuda()
                    losses.append(( fisher_diag_weight * (p-mean)**2).sum())

            return (self.lmd_weight_loss/2)*sum(losses)
        except AttributeError:
            print('error')
            return Variable(torch.zeros(1)).cuda()


    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def show_tsne_with_keys_v2(self, datasets, fname, suffix, epoch, x_keys, task, showlist=[0,4,10], num_class_per_task=10):
        batch_size, sample_size = 64, 4096
        c_num = num_class_per_task*(task+1)

        feats = []
        labels = []
        for tsk in range(task+1):
            n_sample = 0
            num_keys = x_keys[tsk].size()[0]
            data_loader = utils.get_data_loader(datasets[tsk], batch_size)

            _, _ = self(x_keys[tsk].cuda(), tsk)
            y_keys = torch.tensor( [ c_num+tsk*num_class_per_task+k for k in range(num_keys) ] )

            feat_ = self.feats.to('cpu').detach().view(num_keys,-1).numpy().copy() 
            if len(labels) == 0:
                labels = y_keys
                feats = feat_
            else:
                labels = np.concatenate((labels, y_keys))
                feats = np.concatenate((feats, feat_))


            for x, y in data_loader:
                _, _  = self(x.cuda(), tsk)
                y_cpu = y.data.to('cpu').detach().numpy().copy() + tsk*num_class_per_task

                if len(y_cpu) == batch_size:
                    n_sample += batch_size
                    feat_ = self.feats.to('cpu').detach().view(batch_size,-1).numpy().copy() 
                    labels = np.concatenate((labels, y_cpu))
                    feats = np.concatenate((feats, feat_))
                    if n_sample > sample_size: break        
                    # if feats.shape[0] > sample_size: break        

        perplexity_list = [5, 30, 50]
        title = ( 't-SNE \n' + str(suffix) + '\n  epoch:{epoch}'.format(epoch=epoch)
            )

        _, ax = plt.subplots(1, len(perplexity_list), figsize=(15, 5))
        for p, k in enumerate(perplexity_list):
            feat2d = TSNE(n_components=2,perplexity=k).fit_transform(feats)
            # for i in range(c_num):
            for i in range(c_num):
                if i in showlist:
                    # _, ax = plt.subplots(1, p+1, len(perplexity_list), figsize=(10, 10))
                    target = feat2d[labels == i]
                    ax[p].scatter(x=target[:, 0], y=target[:, 1], label=str(i), alpha=0.1)
                else:
                    if torch.rand(1)>0.9:
                        target = feat2d[labels == i]
                        ax[p].scatter(x=target[:, 0], y=target[:, 1], alpha=0.001, color='k')

            for i in range(num_keys):
                target = feat2d[labels == c_num+i]
                ax[p].scatter(x=target[:, 0], y=target[:, 1], marker='${}$'.format(str(i)),label=str(i), alpha=1, s=70, color='k')
            
            ax[p].set_title(title)

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')           
        plt.savefig(fname)
        plt.close() 

        return feat2d
