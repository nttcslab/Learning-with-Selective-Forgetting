from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
#------------------------------
class Distillation_Loss(nn.Module):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
    """
    def __init__(self, T=2.0, scale=1.0):
        super(Distillation_Loss, self).__init__()
        self.T = T
        self.scales = scale*scale*T*T # scale for adtive margin soft-max loss

    def forward(self, y, teacher_scores, msk, lmd, yrand):
        a = F.kl_div(F.log_softmax(y*msk / self.T, dim=1), F.softmax(teacher_scores*msk / self.T, dim=1), reduction='batchmean') + \
            F.kl_div(F.log_softmax(y*(1-msk) / self.T, dim=1), F.softmax(yrand*(1-msk) / self.T, dim=1), reduction='batchmean') 
        return a * lmd * self.scales

class Distillation_Loss_org(nn.Module):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
    """
    def __init__(self, T=2.0, scale=1.0):
        super(Distillation_Loss_org, self).__init__()
        self.T = T
        self.scales = scale*scale*T*T # scale for adtive margin soft-max loss
    def forward(self, y, teacher_scores, msk, lmd, yrand): 
        return lmd * self.scales * F.kl_div(F.log_softmax(yrand / self.T, dim=1), F.softmax(teacher_scores / self.T, dim=1), reduction='batchmean')

class Distillation_Loss_Forget(nn.Module):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
    """
    def __init__(self, T=2.0, scale=1.0):
        super(Distillation_Loss_Forget, self).__init__()
        self.T = T
        self.scales = scale*scale*T*T # scale for adtive margin soft-max loss

    def forward(self, y, teacher_scores, msk, lmd, yrand):
        a = F.kl_div(F.log_softmax(y*(1-msk) / self.T, dim=1), F.softmax(yrand*(1-msk) / self.T, dim=1), reduction='batchmean') 
        return a * lmd * self.scales

class Distillation_Loss_MSK(nn.Module):
    def __init__(self, T=2.0, scale=1.0):
        super(Distillation_Loss_MSK, self).__init__()
        self.T = T
        self.scales = scale*scale*T*T # scale for adtive margin soft-max loss
    def forward(self, y, teacher_scores, msk, lmd):
        scale_coff = y.size(-1)
        a = F.kl_div(F.log_softmax(y*msk / self.T, dim=1), F.softmax(teacher_scores*msk / self.T, dim=1), reduction='batchmean')
        return a * lmd * self.scales

class Distillation_Loss_MSK_CE(nn.Module):

    # y, teacher_scores, T, scale): 
    def __init__(self, T=2.0, scale=1.0):
        super(Distillation_Loss_MSK_CE, self).__init__()
        self.T = T
        self.scale = scale
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    def forward(self, y, teacher_scores, msk, lmd):
        return F.kl_div(F.log_softmax(y*msk / self.T, dim=1), F.softmax(teacher_scores*msk / self.T, dim=1), reduction='batchmean')*lmd*self.T*self.T

class _UNUSED_Distillation_Loss_RND(nn.Module):

    # y, teacher_scores, T, scale): 
    def __init__(self, T=2.0, scale=1.0):
        super(Distillation_Loss_RND, self).__init__()
        self.T = T
        self.scale = scale

    def forward(self, y, teacher_scores, msk, lmd, yrand):
        scale_tmp =  y.size(-1)* y.size(-1)
        a = F.kl_div(F.log_softmax(y*msk / self.T, dim=1), F.softmax(teacher_scores*msk / self.T, dim=1), reduction='batchmean') * scale_tmp * lmd + \
            F.kl_div(F.log_softmax(y*(1-msk) / self.T, dim=1), F.softmax(yrand*(1-msk) / self.T, dim=1), reduction='batchmean') * scale_tmp * lmd
        return a


class Distillation_Loss_RND_CE(nn.Module):
    def __init__(self, T=2.0, scale=1.0):
        super(Distillation_Loss_RND_CE, self).__init__()
        self.T = T
        # self.scale = scale
    def forward(self, y, teacher_scores, msk, lmd, yrand):
        scale_tmp = y.size(-1)
        a = F.kl_div(F.log_softmax(y*msk / self.T, dim=1), F.softmax(teacher_scores*msk / self.T, dim=1), reduction='batchmean') * lmd * self.T * self.T + \
            F.kl_div(F.log_softmax(y*(1-msk) / self.T, dim=1), F.softmax(yrand*(1-msk) / self.T, dim=1), reduction='batchmean')  * lmd * self.T * self.T
        return a
#-----------------------------------------
class AdditiveMarginLoss(nn.Module):
    def __init__(self, scale, margin):
        super(AdditiveMarginLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits, y, lmd=1.0):
        y = y.type(torch.float32)
        cos_theta = logits  # /(norm_W+1e-10)
        xi = cos_theta - self.margin

        a = torch.sum(torch.exp(self.scale * xi) * y, dim=1,  keepdim=True)
        rev_y = 1. - y

        sum_c = torch.sum(rev_y * torch.exp(self.scale * cos_theta), dim=1,
                          keepdim=True)
        b = a + sum_c

        l_ams = - torch.mean(torch.log(a/b + 1e-20))
        
        return lmd*l_ams


class AdditiveMarginLoss_MSK(nn.Module):
    def __init__(self, scale, margin):
        super(AdditiveMarginLoss_MSK, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits, y, msk, lmd=1.0):
        y = y.type(torch.float32)
        cos_theta = logits  # /(norm_W+1e-10)
        xi = cos_theta - self.margin

        a = torch.sum(torch.exp(self.scale * xi) * y, dim=1,  keepdim=True)
        rev_y = 1. - y

        sum_c = torch.sum(rev_y * torch.exp(self.scale * cos_theta), dim=1,
                          keepdim=True)
        b = a + sum_c

        l_ams = - torch.mean(torch.log(a/b + 1e-20))
        
        return lmd*l_ams

#-----------------------------------------
class MultiLandMarkLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(MultiLandMarkLoss, self).__init__()

    def forward(self, x, labels, key_centers, key_labels, lmd_cnt):
        num_key = len(key_labels)
        batch_size = x.size(0)
        distmat = torch.pow( x, 2).sum(dim=1, keepdim=True).expand(batch_size, num_key) + \
                  torch.pow( key_centers, 2).sum(dim=1, keepdim=True).expand(num_key, batch_size).t()
        distmat.addmm_(1, -2, x,  key_centers.t() )

        labels_mlt = labels.unsqueeze(1).expand(batch_size, num_key)

        set_key_labels = key_labels.cuda()
        
        mask = (labels_mlt.eq(set_key_labels.expand(batch_size, num_key))).float()

        dist = distmat *mask

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return lmd_cnt*loss

class Attention_Distillation_Loss(nn.Module):
    def __init__(self, use_gpu=True):
        super(Attention_Distillation_Loss, self).__init__()

    def forward(self, attention_map1, attention_map2):
        """Calculates the attention distillation loss"""
        attention_map1 = torch.norm(attention_map1, p=2, dim=1)
        attention_map2 = torch.norm(attention_map2, p=2, dim=1)
        return torch.norm(attention_map2 - attention_map1, p=1, dim=1).sum(dim=1).mean()

#-----------------------------------------
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None