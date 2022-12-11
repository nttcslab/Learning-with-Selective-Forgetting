from functools import reduce
from typing import Any, Callable, List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
#--------
import src.utils as utils
import torch
from sklearn.manifold import TSNE
from torch import Tensor, autograd, nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.functional import avg_pool2d, relu
from src.utils_BC import BiasLayer



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_mul(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        layers: List[int] = [2,2,2,2],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        lmd_weight_loss: float = 40,
        num_task: int =2
    ) -> None:
        super(ResNet_mul, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.r = 0.5
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


        # bias layer for BiC (see CVPR19)
        self.bias_layers = []
        self.output_size = num_classes
        self.lmd_weight_loss = lmd_weight_loss

        # self.w2_new     = nn.Linear(512 * block.expansion, num_classes, bias=False)

        layer = [nn.Linear(512 * block.expansion, self.output_size[i], bias=True) for i in range(num_task)]
        self.w2_mul = nn.ModuleList(layer)
        

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)



    @property
    def name(self):
        return ('ResNet-out{output_size}').format(output_size=self.output_size)

    def forward(self, x: Tensor, task: int):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        logits = torch.flatten(x, 1)
        logits = F.normalize(logits, p=2, dim=1)
        self.feats = logits.detach().clone()

        w2_weight = self.w2_mul[task]
        w2_norm = w2_weight.weight / torch.norm(w2_weight.weight, dim=1, keepdim=True)
        logits = torch.matmul(logits, w2_norm.T)


        # x = self.fc(x)

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


    def ewc_loss_selective(self, cuda, ewc_rev, task):
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
                    fisher_diag_weight = torch.clamp( fisher_pos+ewc_rev*fisher_neg, min=0)
                    fisher_diag_weight = fisher_diag_weight.cuda()
                    losses.append(( fisher_diag_weight * (p-mean)**2).sum())
            return (self.lmd_weight_loss/2)*sum(losses)
        except AttributeError:
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

    def __UNUSED__estimate_targeted_fisher(self, dataset, sample_size, 
            batch_size,
            delete_class,
            flg_preserve,
            task,
            is_task_wise=1,
            num_class_per_task=10,
            ):

        class_label_offset = 0
        if is_task_wise==0: class_label_offset = num_class_per_task*task
        # gen keep class
        if flg_preserve == 1:
            y_list = list(range(self.output_size))
            y_keep_class = y_list.copy()
            for k in y_list:
                for t in delete_class:
                    if k==t:
                        y_keep_class.remove(k)
            target_class = y_keep_class
            del y_keep_class
        else:
            target_class = delete_class
        
        data_loader = utils.get_data_loader(dataset, batch_size)

        num_smp, flg, num_smp_batch = 0, 0, 0
        #--------------------------------
        # sample_batch_size = 32
        #-------------------------------
        print("num:", end="")
        for x, y in dataset:

            x = x.cuda()
            y_cpu = y.data.to('cpu').detach().numpy().copy()

            idxs_x = [k for k, y_ in enumerate(y_cpu) for d in target_class if y_ == d ]
            idxs_y = [ y_cpu[x_] for x_ in idxs_x ]

            # logits_keep, _, _ = self(x, task)
            logits_keep, _ = self(x, task)
            loglikelihoods_trgt=  [ F.log_softmax(logits_keep, dim=1)[xi, yi-class_label_offset] for xi,yi in zip(idxs_x, idxs_y) ] 
            
            num_smp_batch = len(loglikelihoods_trgt)

            if num_smp_batch > 0:
                num_smp += num_smp_batch

                loglikelihoods_trgt = tuple( [item for item in loglikelihoods_trgt] )

                loglikelihood_grads_trgt_tmp = []
                for i, l in enumerate(loglikelihoods_trgt, 1):
                    grad_val = autograd.grad( l, self.parameters(), allow_unused=True, retain_graph=(i < len(loglikelihoods_trgt)) ) 
                    grad_val = tuple( item if item is not None else torch.tensor(0) for item in grad_val  )
                    loglikelihood_grads_trgt_tmp.append(grad_val)

                loglikelihood_grads_trgt = zip(*(loglikelihood_grads_trgt_tmp))
                del x, loglikelihoods_trgt
                torch.cuda.empty_cache()

                loglikelihood_grads_trgt = [torch.stack(gs) for gs in loglikelihood_grads_trgt]

                fisher_diagonals_trgt = [ ((gp ** 2).sum(0)).to('cpu').detach() for gp in loglikelihood_grads_trgt]

                del loglikelihood_grads_trgt

                if flg > 0:
                    for k in range(len(fisher_diagonals_trgt_sum)):
                        fisher_diagonals_trgt_sum[k] += fisher_diagonals_trgt[k].clone() 
                        # print(k)
                else:
                    fisher_diagonals_trgt_sum = fisher_diagonals_trgt.copy()
                    flg = 1

                del fisher_diagonals_trgt
                torch.cuda.empty_cache()
                    
                if num_smp > sample_size: break

        if num_smp == 0:
            for x, y in data_loader:
                x = x.cuda()
                y_cpu = y.data.to('cpu').detach().numpy().copy()
                idxs_x = [k for k, y in enumerate(y_cpu)]
                idxs_y = [ y_cpu[x_] for x_ in idxs_x ]
                logits_keep, _ = self(x, task)
                loglikelihoods_trgt=  [ F.log_softmax(logits_keep, dim=1)[xi, yi-class_label_offset] for xi,yi in zip(idxs_x, idxs_y) ]                 
                num_smp_batch = len(loglikelihoods_trgt)
                if num_smp_batch > 0:
                    num_smp += num_smp_batch
                    loglikelihoods_trgt = tuple( [item for item in loglikelihoods_trgt] )
                    loglikelihood_grads_trgt_tmp = []
                    for i, l in enumerate(loglikelihoods_trgt, 1):
                        grad_val = autograd.grad( l, self.parameters(), allow_unused=True, retain_graph=(i < len(loglikelihoods_trgt)) ) 
                        grad_val = tuple( item if item is not None else torch.tensor(0) for item in grad_val  )
                        loglikelihood_grads_trgt_tmp.append(grad_val)
                    loglikelihood_grads_trgt = zip(*(loglikelihood_grads_trgt_tmp))
                    del x, loglikelihoods_trgt
                    torch.cuda.empty_cache()
                    loglikelihood_grads_trgt = [torch.stack(gs) for gs in loglikelihood_grads_trgt]
                    fisher_diagonals_trgt = [ ((gp ** 2).sum(0)).to('cpu').detach() for gp in loglikelihood_grads_trgt]
                    del loglikelihood_grads_trgt
                    fisher_diagonals_trgt_sum = fisher_diagonals_trgt.copy()
                    del fisher_diagonals_trgt
                    torch.cuda.empty_cache()                        
                    if num_smp > 0: break
            for k in range(len(fisher_diagonals_trgt_sum)):
                fisher_diagonals_trgt_sum[k] = (0*fisher_diagonals_trgt_sum[k])/num_smp
        else:
            for k in range(len(fisher_diagonals_trgt_sum)):
                fisher_diagonals_trgt_sum[k] = fisher_diagonals_trgt_sum[k]/num_smp

        fisher_diagonals = [fisher_diagonals_trgt_sum[k] for k in range(len(fisher_diagonals_trgt_sum)) ]
            
        param_names = [ n.replace('.', '__') for n, p in self.named_parameters()]

        return {n: f.detach().cuda() for n, f in zip(param_names, fisher_diagonals)}       

    def __UNUSED__estimate_targeted_fisher_MAS(self, dataset, sample_size, 
            batch_size,
            delete_class,
            flg_preserve,
            task,
            is_task_wise=1,
            num_class_per_task=10,
            ):

        fisher = {}
        for n, p in self.named_parameters():
            fisher[n] = 0*p.data
        
        data_loader = utils.get_data_loader(dataset, batch_size)

        self.train()
        count = 0
        for x, y in dataset:
            count += 1

            x = x.cuda()

            # logits_keep, _, _ = self(x, task)
            logits_keep, _ = self(x, task)
            loss = torch.sum(torch.norm(logits_keep, 2, dim=1))/float(logits_keep.size(0))/float(logits_keep.size(1))
            loss.backward()

            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

        for n, _ in self.named_parameters():
            fisher[n] = fisher[n]/float(count)
            fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)

            
        fisher_diagonals = [ fisher[n] for n, _ in self.named_parameters()]    
        param_names = [ n.replace('.', '__') for n, p in self.named_parameters()]

        return {n: f.detach().cuda() for n, f in zip(param_names, fisher_diagonals)}       

class ResNet_inc3(nn.Module):
    def __init__(self, input_size, output_size, lmd_weight_loss=40, alpha=1.0, scale=30, input_ch=1, dim_fea=128, num_task=4, \
        block = BasicBlock, num_block =  [3, 4, 6, 3]):
        # [2,2,2,2]=res18

        super(ResNet_inc3, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.w2_mul = nn.Linear(512 * block.expansion, output_size, bias=False)
        
        self.input_size = input_size
        self.output_size = output_size
        self.lmd_weight_loss = lmd_weight_loss
        self.scale = scale
        self.dim_fea = 512 * block.expansion

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    @property
    def name(self):
        return ('ResNet-in{input_size}').format(input_size=self.input_size)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)        
        logits = output.view(output.size(0), -1)

        logits = F.normalize(logits, p=2, dim=1)
        self.feats = logits.detach().clone()

        w2_weight = self.w2_mul
        w2_norm = w2_weight.weight / torch.norm(w2_weight.weight, dim=1, keepdim=True)
        logits = torch.matmul(logits, w2_norm.T)

        return logits, self.feats

    def evl_acc(self, logits, labels): 
        self.eval()
        _, y_pred = logits.max(1)
        precision = (y_pred == labels).sum().float() / len(logits)
        self.train()
        return precision

    def re_init_weight(self, pname='w2_mul__weight_0', prev_cls = 50, task = 0):
        mean = getattr(self, '{}_{}_mean'.format(pname, task))
        fpos = getattr(self, '{}_{}_fisher_pos'.format(pname, task))
        fneg = getattr(self, '{}_{}_fisher_neg'.format(pname, task))

        # self.w2_mul = nn.Linear(512 * block.expansion, output_size, bias=False)
        mean[prev_cls:, :] = 0
        fpos[prev_cls:, :] = 0
        fneg[prev_cls:, :] = 0

        self.register_buffer('{}_{}_mean'.format(pname, task), mean )
        self.register_buffer('{}_{}_fisher_pos'.format(pname, task), fpos )
        self.register_buffer('{}_{}_fisher_neg'.format(pname, task), fneg )

    def consolidate_selective(self, fisher_pos, fisher_neg, task):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_{}_mean'.format(n,task), p.data.clone())
            self.register_buffer('{}_{}_fisher_pos'.format(n,task), fisher_pos[n].data.clone())
            self.register_buffer('{}_{}_fisher_neg'.format(n,task), fisher_neg[n].data.clone())

    def ewc_loss_selective(self, cuda, ewc_rev, task):
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
                    fisher_diag_weight = torch.clamp( fisher_pos+ewc_rev*fisher_neg, min=0)
                    fisher_diag_weight = fisher_diag_weight.cuda()
                    losses.append(( fisher_diag_weight * (p-mean)**2).sum())
            return (self.lmd_weight_loss/2)*sum(losses)
        except AttributeError:
            return Variable(torch.zeros(1)).cuda()
            
    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def final_layer_param(self):
        optim_params = \
            [{'params': f2.parameters()},{'params': fc1.parameters()}]
        return optim_params

    def estimate_targeted_fisher(self, dataset, sample_size, 
            batch_size,
            delete_class,
            flg_preserve,
            task,
            is_task_wise=1,
            num_class_per_task=10,
            ):

        class_label_offset = 0
        if is_task_wise==0: class_label_offset = num_class_per_task*task
        # gen keep class
        if flg_preserve == 1:
            y_list = list(range(self.output_size))
            y_keep_class = y_list.copy()
            for k in y_list:
                for t in delete_class:
                    if k==t:
                        y_keep_class.remove(k)
            target_class = y_keep_class
            del y_keep_class
        else:
            target_class = delete_class
        
        data_loader = utils.get_data_loader(dataset, batch_size)

        num_smp, flg, num_smp_batch = 0, 0, 0
        #--------------------------------
        # sample_batch_size = 32
        #-------------------------------
        print("num:", end="")
        for x, y in data_loader:

            x = x.cuda()
            y_cpu = y.data.to('cpu').detach().numpy().copy()

            idxs_x = [k for k, y_ in enumerate(y_cpu) for d in target_class if y_ == d ]
            idxs_y = [ y_cpu[x_] for x_ in idxs_x ]

            # logits_keep, _, _ = self(x, task)
            logits_keep, _ = self(x)
            loglikelihoods_trgt=  [ F.log_softmax(logits_keep, dim=1)[xi, yi-class_label_offset] for xi,yi in zip(idxs_x, idxs_y) ] 
            
            num_smp_batch = len(loglikelihoods_trgt)

            if num_smp_batch > 0:
                num_smp += num_smp_batch

                loglikelihoods_trgt = tuple( [item for item in loglikelihoods_trgt] )

                loglikelihood_grads_trgt_tmp = []
                
                for i, l in enumerate(loglikelihoods_trgt, 1):
                    grad_val = autograd.grad( l, self.parameters(), allow_unused=True, retain_graph=(i < len(loglikelihoods_trgt)) ) 
                    grad_val = tuple( item if item is not None else torch.tensor(0) for item in grad_val  )
                    loglikelihood_grads_trgt_tmp.append(grad_val)

                loglikelihood_grads_trgt = zip(*(loglikelihood_grads_trgt_tmp))
                del x, loglikelihoods_trgt
                torch.cuda.empty_cache()

                loglikelihood_grads_trgt = [torch.stack(gs) for gs in loglikelihood_grads_trgt]

                fisher_diagonals_trgt = [ ((gp ** 2).sum(0)).to('cpu').detach() for gp in loglikelihood_grads_trgt]

                del loglikelihood_grads_trgt

                if flg > 0:
                    for k in range(len(fisher_diagonals_trgt_sum)):
                        fisher_diagonals_trgt_sum[k] += fisher_diagonals_trgt[k].clone() 
                        # print(k)
                else:
                    fisher_diagonals_trgt_sum = fisher_diagonals_trgt.copy()
                    flg = 1

                del fisher_diagonals_trgt
                torch.cuda.empty_cache()
                    
                if num_smp > sample_size: break

        #---------------------------------
        if num_smp == 0:
            #---------------------------------
            ## Exception handling (when num_smp=0 because there is no applicable class)
            #---------------------------------
            for x, y in data_loader:
                x = x.cuda()
                y_cpu = y.data.to('cpu').detach().numpy().copy()
                idxs_x = [k for k, y in enumerate(y_cpu)]
                idxs_y = [ y_cpu[x_] for x_ in idxs_x ]
                logits_keep, _ = self(x)
                loglikelihoods_trgt=  [ F.log_softmax(logits_keep, dim=1)[xi, yi-class_label_offset] for xi,yi in zip(idxs_x, idxs_y) ]                 
                num_smp_batch = len(loglikelihoods_trgt)
                if num_smp_batch > 0:
                    num_smp += num_smp_batch
                    loglikelihoods_trgt = tuple( [item for item in loglikelihoods_trgt] )
                    loglikelihood_grads_trgt_tmp = []
                    for i, l in enumerate(loglikelihoods_trgt, 1):
                        grad_val = autograd.grad( l, self.parameters(), allow_unused=True, retain_graph=(i < len(loglikelihoods_trgt)) ) 
                        grad_val = tuple( item if item is not None else torch.tensor(0) for item in grad_val  )
                        loglikelihood_grads_trgt_tmp.append(grad_val)
                    loglikelihood_grads_trgt = zip(*(loglikelihood_grads_trgt_tmp))
                    del x, loglikelihoods_trgt
                    torch.cuda.empty_cache()
                    loglikelihood_grads_trgt = [torch.stack(gs) for gs in loglikelihood_grads_trgt]
                    fisher_diagonals_trgt = [ ((gp ** 2).sum(0)).to('cpu').detach() for gp in loglikelihood_grads_trgt]
                    del loglikelihood_grads_trgt
                    fisher_diagonals_trgt_sum = fisher_diagonals_trgt.copy()
                    del fisher_diagonals_trgt
                    torch.cuda.empty_cache()                        
                    if num_smp > 0: break
            for k in range(len(fisher_diagonals_trgt_sum)):
                fisher_diagonals_trgt_sum[k] = (0*fisher_diagonals_trgt_sum[k])/num_smp
            #---------------------------------
        else:
            for k in range(len(fisher_diagonals_trgt_sum)):
                fisher_diagonals_trgt_sum[k] = fisher_diagonals_trgt_sum[k]/num_smp

        fisher_diagonals = [fisher_diagonals_trgt_sum[k] for k in range(len(fisher_diagonals_trgt_sum)) ]
            
        param_names = [ n.replace('.', '__') for n, p in self.named_parameters()]

        return {n: f.detach().cuda() for n, f in zip(param_names, fisher_diagonals)}           


