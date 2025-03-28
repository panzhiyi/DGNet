import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from openpoints.utils import registry
import logging
from torch_scatter import scatter
from IPython.core.debugger import set_trace

LOSS = registry.Registry('loss')
LOSS.register_module(name='CrossEntropy', module=CrossEntropyLoss)
LOSS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
LOSS.register_module(name='BCEWithLogitsLoss', module=BCEWithLogitsLoss)

@LOSS.register_module()
class tCE(torch.nn.Module):
    def __init__(self, thr=0.2):
        super(tCE, self).__init__()
        self.thr = thr
        
    def forward(self, logit, target):
        logit = F.softmax(logit, dim=1)
        logit = torch.clamp(logit, min=eps, max=1.0)
        label_one_hot = F.one_hot(target, self.num_classes).float().to(logit.device)
        return loss.mean()
        return loss

@LOSS.register_module()
class vMFLoss(torch.nn.Module):
    def __init__(self, label_smoothing=0.2,
                 ignore_index=255,
                 start_epoch=51,
                 nonexact_loss = None,
                 ka = 10,
                 iter_num = 10,
                 loss_type = 'tce_vmf_consist_contrast',
                 ablation = 'avmf_uvmf',
                 init_u = 'part',
                 beta = 0.8,
                 weight = None
                ):
        super(vMFLoss, self).__init__()
        self.ignore_index = ignore_index
        self.start_epoch = start_epoch
        self.nonexact_loss = nonexact_loss
        self.ka = ka
        self.iter_num = iter_num
        self.loss_type = loss_type
        self.ablation = ablation
        self.init_u = init_u
        self.beta = beta
        self.creterion = CrossEntropyLoss(reduction = 'mean', ignore_index=self.ignore_index)
        self.creterion_none = CrossEntropyLoss(reduction = 'none', ignore_index=self.ignore_index)
        self.creterion_BCE = BCEWithLogitsLoss() 
    def forward(self, logit, target, feat, epoch, prototypes_dataset):
        # feat: B * C * P, C is channel number
        # logit: B * K * P , K is class number
        # target: B * P
        B, K, P = logit.shape
        C = feat.shape[1]
        index = target.clone()
        index[index == self.ignore_index] = K

        if self.nonexact_loss is None:
            if epoch >= self.start_epoch:
                feat_norm = F.normalize(feat,p=2,dim=1)

                prototypes = scatter(feat, index.unsqueeze(1), dim = -1, reduce='sum') # B * C * (K+1)
                counts = scatter(torch.ones_like(index), index, dim = 1, reduce='sum') # B * (K+1)
                counts_batch = torch.sum(counts,0)
                prototypes_sum = torch.sum(prototypes,0)
                prototypes_sum = prototypes_sum[:,:K] # C * K
                counts_batch =  counts_batch[:K]
                u_vMF_init = prototypes_sum/(counts_batch + 1e-6)
                u_vMF_init = F.normalize(u_vMF_init,p=2,dim=0)
                if prototypes_dataset is not None and 'part' in self.init_u.lower():
                    u_vMF = u_vMF_init.clone()
                    u_vMF[:, counts_batch == 0] = prototypes_dataset[:, counts_batch == 0]
                elif prototypes_dataset is not None and 'all' in self.init_u.lower():
                    u_vMF = prototypes_dataset
                elif 'none' in self.init_u.lower():
                    u_vMF = u_vMF_init.clone()
                    
                a_vMF = torch.ones(1,1,K).cuda() / K
                u_vMF = F.normalize(u_vMF,p=2,dim=0)
                dis = torch.bmm(feat_norm.transpose(1,2), torch.unsqueeze(u_vMF, 0).repeat(B,1,1)) # B * P * K
                
                exp_dis = a_vMF * torch.exp(self.ka * dis) # B * P * K 
                p_vMF = exp_dis/torch.sum(exp_dis,-1, keepdim=True) # normalized without alpha = softmax(ka * dis,-1) B * P * K
                
                ### EM process ###
                for i in range(self.iter_num):
                    if 'avmf' in self.ablation.lower():
                        a_vMF = torch.mean(p_vMF,1,keepdim=True) # B * 1 * K
                        a_vMF = torch.mean(p_vMF,0,keepdim=True) # 1 * 1 * K
                    if 'uvmf' in self.ablation.lower():
                        u_vMF = torch.mean(torch.bmm(feat_norm, p_vMF),dim=0) # C * K
                        #if prototypes_dataset is not None:
                        #    u_vMF[:, counts_batch == 0] = prototypes_dataset[:, counts_batch == 0]
                        u_vMF = F.normalize(u_vMF,p=2,dim=0)
                        dis = torch.bmm(feat_norm.transpose(1,2), torch.unsqueeze(u_vMF, 0).repeat(B,1,1))
                    exp_dis = a_vMF * torch.exp(self.ka * dis) # B * P * K 
                    p_vMF = exp_dis/torch.sum(exp_dis, -1, keepdim=True) # normalized without alpha = softmax(ka * dis,-1) B * P * K
                
                vMF = torch.argmax(p_vMF, -1) # B * P 
                vMF_hard_loss = torch.mean(vMF.view(B,P,-1) * dis) # hard vMF
                
                vMF_loss = torch.mean(- p_vMF * (torch.log(a_vMF + 1e-6) +  self.ka * dis)) / np.log(K) # soft vMF
                 
                consist_loss = - p_vMF.transpose(1,2) * F.log_softmax(logit, dim=1)
                consist_loss = consist_loss.mean()
                #contrast_loss = (torch.sum(torch.mm(u_vMF.transpose(0,1), u_vMF)) - K)/(K * (K-1))
                u_vMF_init = u_vMF_init[:, counts_batch != 0]
                K_ = torch.sum(counts_batch != 0)
                contrast_loss = (torch.sum(torch.mm(u_vMF_init.transpose(0,1), u_vMF_init)) - K_)/(K_ * (K_-1))
                if 'pce' in self.loss_type.lower():
                    loss = self.creterion(logit, target)
                elif 'tce' in self.loss_type.lower():
                    loss = self.creterion_none(logit, target)
                    mask = (target!=self.ignore_index)
                    loss = torch.sum(torch.clamp(loss, min=-np.log(self.beta))*mask)/(torch.sum(mask)+1e-6)
                if 'vmf' in self.loss_type.lower():
                    loss = loss + vMF_loss
                if 'hard' in self.loss_type.lower():
                    loss = loss + vMF_hard_loss
                if 'contrast' in self.loss_type.lower():
                    loss = loss + contrast_loss
                if 'consist' in self.loss_type.lower():
                    loss = loss + consist_loss
            else:
                prototypes = scatter(feat, index.unsqueeze(1), dim = -1, reduce='sum') # B * C * (K+1)
                counts = scatter(torch.ones_like(index), index, dim = 1, reduce='sum') # B * (K+1)
                counts_batch = torch.sum(counts,0)
                prototypes_sum = torch.sum(prototypes,0)
                prototypes_sum = prototypes_sum[:,:K] # C * K
                counts_batch =  counts_batch[:K]
                if 'pce' in self.loss_type.lower():
                    loss = self.creterion(logit, target)
                elif 'tce' in self.loss_type.lower():
                    loss = self.creterion_none(logit, target)
                    mask = (target!=self.ignore_index)
                    loss = torch.sum(torch.clamp(loss, min=-np.log(self.beta))*mask)/(torch.sum(mask)+1e-6)

        return loss, prototypes_sum.detach(), counts_batch.detach()
                
@LOSS.register_module()
class AGMM_prototype(torch.nn.Module):
    def __init__(self, label_smoothing=0.2,
                 ignore_index=255,
                 theta=0.5,
                 weight = None
                ):
        super(AGMM_prototype, self).__init__()
        self.ignore_index = ignore_index
        self.theta = theta
        self.creterion = CrossEntropyLoss(reduction = 'mean', ignore_index=self.ignore_index, label_smoothing=label_smoothing)
        self.creterion_BCE = BCEWithLogitsLoss()
        self.nonexact_loss = False

    def forward(self, logit, target, feat, epoch, prototypes_pre):
        # feat: B * C * P, C is channel number
        # logit: B * K * P , K is class number
        # target: B * P
        B, K, P = logit.shape
        C = feat.shape[1]
        index = target.clone()
        index[index == self.ignore_index] = K
        if not self.nonexact_loss:

            prototypes_ = scatter(feat, index.unsqueeze(1), dim = -1, reduce='sum') # B * C * (K+1)
            counts = scatter(torch.ones_like(index), index, dim = 1, reduce='sum') # B * (K+1)
            prototypes_ = prototypes_[:,:,:K]
            counts = counts[:,:K]
            
            if prototypes_pre is not None:
                prototypes = prototypes_pre
            else:
                prototypes = torch.sum(prototypes_,0)/(torch.sum(counts,0)+1e-6)
                prototypes = prototypes.view(C,K,1).repeat(1,1,10).reshape(C,-1)
            #prototypes = prototypes[:,:K] # C * K
            dis = torch.abs(prototypes.view(1, C, 10 * K, -1) - feat.view(B, C, -1, P)).mean(1)
            
            Gmm = torch.exp(-(dis * dis))# we set sigma^2 = 1
            Gmm = torch.sum(Gmm.reshape(B,K,10,P),dim=2)
            Gmm = F.normalize(Gmm, p=1, dim=1)
            loss1 = - (Gmm * F.log_softmax(logit, dim=1) + (1 - Gmm) * torch.log(1 - F.softmax(logit, dim=1) + 1e-6)) 
            loss1 = loss1.mean() + self.creterion(Gmm, target)

            mask = scatter(torch.ones_like(index).float(), index, dim = 1, reduce='mean') # B * (K+1)
            mask = torch.max(mask, 0)[0]
            mask = mask[:K]
            mask = torch.mm(mask.view(K,1),mask.view(1,K))
            mask = mask - torch.diag_embed(torch.diagonal(mask,dim1=0,dim2=1))
            
            #prototypes_norm = F.normalize(prototypes,p=2,dim=0)#do we need norm?
            prototypes_norm = F.normalize(torch.sum(prototypes_,0)/(torch.sum(counts,0)+1e-6),p=2,dim=0)
            negetive_matrix = torch.mm(prototypes_norm.transpose(1,0), prototypes_norm)
            loss1 = loss1 + torch.sum(negetive_matrix*mask)/torch.sum(mask)

            loss = (1 - np.power(epoch,self.theta)/2)*self.creterion(logit, target) + np.power(epoch,self.theta)/2 * loss1
        return loss, torch.sum(prototypes_,0).detach(), torch.sum(counts,0)                
        

@LOSS.register_module()
class AsynchronousCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2,
                 ignore_index=255,
                 start_epoch = 50,
                 step = 5,
                 weight_entropy = 0.01,
                 weight = None
                ):
        super(AsynchronousCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.strat_epoch = start_epoch
        self.step = step
        self.weight_entropy = weight_entropy
        self.creterion = CrossEntropyLoss(reduction = 'none', ignore_index=self.ignore_index, label_smoothing=label_smoothing)
        
    def forward(self, logit, target, epoch):
        loss_crossentropy = self.creterion(logit, target)
        
        if epoch > self.strat_epoch and (epoch - self.strat_epoch - 1) % (2 * self.step) < self.step:
            loss_entropy = -torch.sum(F.softmax(logit, dim = 1) * F.log_softmax(logit, dim = 1), dim = 1)
            loss_entropy[target==self.ignore_index] = 0
            loss = loss_crossentropy + self.weight_entropy * loss_entropy
        else:
            '''
            loss_entropy = -torch.sum(F.softmax(logit, dim = 1) * F.log_softmax(logit, dim = 1), dim = 1)
            loss_entropy[target==self.ignore_index] = 0
            loss = loss_crossentropy + self.weight_entropy * loss_entropy
            '''
            '''
            with torch.no_grad():
                inverse_density = 1.0 / density
            loss = inverse_density * loss_crossentropy
            '''

            with torch.no_grad():
                loss_entropy = -torch.sum(F.softmax(logit, dim = 1) * F.log_softmax(logit, dim = 1), dim = 1)
            loss = loss_entropy * loss_crossentropy

        loss = torch.mean(loss)
        return loss

@LOSS.register_module()
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=255, 
                 num_classes=None, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
        logging.info(f"I use ignore and is {str(self.ignore_index)}")
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            print(f"I use ignore and is {str(self.ignore_index)}")
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss


@LOSS.register_module()
class MaskedCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2):
        super(MaskedCrossEntropy, self).__init__()
        self.creterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logit, target, mask):
        logit = logit.transpose(1, 2).reshape(-1, logit.shape[1])
        target = target.flatten()
        mask = mask.flatten()
        idx = mask == 1
        loss = self.creterion(logit[idx], target[idx])
        return loss

@LOSS.register_module()
class BCELogits(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = BCEWithLogitsLoss(**kwargs)
        
    def forward(self, logits, targets):
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        targets = targets.contiguous().view(-1)
        num_clsses = logits.shape[-1]
        targets_onehot = F.one_hot(targets, num_classes=num_clsses).to(device=logits.device,dtype=logits.dtype)
        return self.criterion(logits, targets_onehot)

@LOSS.register_module()
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logit, target):
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)  # N,C,H,W => N,C,H*W
            logit = logit.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logit = logit.contiguous().view(-1, logit.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(logit)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != logit.data.type():
                self.alpha = self.alpha.type_as(logit.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()




@LOSS.register_module()
class Poly1CrossEntropyLoss(torch.nn.Module):
    def __init__(self,
                 num_classes: int =50,
                 epsilon: float = 1.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        labels = labels.contiguous().view(-1)

        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


@LOSS.register_module()
class Poly1FocalLoss(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

@LOSS.register_module()
class MultiShapeCrossEntropy(torch.nn.Module):
    def __init__(self, criterion_args, **kwargs):
        super(MultiShapeCrossEntropy, self).__init__()
        self.criterion = build_criterion_from_cfg(criterion_args)

    def forward(self, logits_all_shapes, points_labels, shape_labels):
        batch_size = shape_labels.shape[0]
        losses = 0
        for i in range(batch_size):
            sl = shape_labels[i]
            logits = torch.unsqueeze(logits_all_shapes[sl][i], 0)
            pl = torch.unsqueeze(points_labels[i], 0)
            loss = self.criterion(logits, pl)
            losses += loss
        return losses / batch_size

def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT): 
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return LOSS.build(cfg, **kwargs)