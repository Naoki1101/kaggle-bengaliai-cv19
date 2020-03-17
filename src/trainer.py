import gc
import time
import logging
from fastprogress import master_bar, progress_bar
import numpy as np

from sklearn.metrics import recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import BengaliDataset
from models import ResNet, ResNetD, DenseNet, SeNet, ENet, Mish
from losses import FocalLoss, SmoothCrossEntropyLoss, ClassBalancedLoss, OhemLoss
from scheduler import CosineAnnealingWarmUpRestarts

def train_model(x_trn, x_val, config, num_classes, weights, device):

    y_gr_val = x_val['grapheme_root']
    y_vo_val = x_val['vowel_diacritic']
    y_co_val = x_val['consonant_diacritic']

    model_params = config.model_params

    train_dataset = BengaliDataset(x_trn, n_channels=model_params.n_channels, img_size=config.img_size, transforms=config.augmentation)
    valid_dataset = BengaliDataset(x_val, n_channels=model_params.n_channels, img_size=config.img_size, transforms=None)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=3)

    del train_dataset, valid_dataset; gc.collect()

    if 'se_resnext' in model_params.model_name:
        model = SeNet(model_name=model_params.model_name,
                      n_channels=model_params.n_channels,
                      n_classes=model_params.n_classes,
                      pretrained=model_params.pretrained).to(device)
    
    elif 'resnetd' in model_params.model_name:
        model = ResNetD(model_name=model_params.model_name,
                        n_channels=model_params.n_channels,
                        n_classes=model_params.n_classes).to(device)

    elif 'resne' in model_params.model_name:
        model = ResNet(model_name=model_params.model_name,
                       n_channels=model_params.n_channels,
                       n_classes=model_params.n_classes,
                       pretrained=model_params.pretrained).to(device)

    elif 'densenet' in model_params.model_name:
        model = DenseNet(model_name=model_params.model_name,
                         n_channels=model_params.n_channels,
                         n_classes=model_params.n_classes,
                         pretrained=model_params.pretrained).to(device)

    elif 'efficient' in model_params.model_name:
        model = ENet(model_name=model_params.model_name,
                     n_channels=model_params.n_channels,
                     n_classes=model_params.n_classes,
                     pretrained=model_params.pretrained).to(device)

    if config.model_state_fname is not None:
        model.load_state_dict(torch.load(f'../logs/{config.model_state_fname}/weight_best.pt'))
    
    # relu_replace(model)
    # bn_replace(model)

    weights_gr = torch.from_numpy(weights['grapheme_root']).cuda()
    weights_vo = torch.from_numpy(weights['vowel_diacritic']).cuda()
    weights_co = torch.from_numpy(weights['consonant_diacritic']).cuda()
    if config.loss == 'CrossEntropyLoss':
        # criterion_gr = nn.CrossEntropyLoss(weight=weights_gr)
        # criterion_vo = nn.CrossEntropyLoss(weight=weights_vo)
        # criterion_co = nn.CrossEntropyLoss(weight=weights_co)
        criterion_gr = nn.CrossEntropyLoss()
        criterion_vo = nn.CrossEntropyLoss()
        criterion_co = nn.CrossEntropyLoss()
    elif config.loss == 'SmoothCrossEntropyLoss':
        criterion_gr = SmoothCrossEntropyLoss()
        criterion_vo = SmoothCrossEntropyLoss()
        criterion_co = SmoothCrossEntropyLoss()
    elif config.loss == 'FocalLoss':
        criterion_gr = FocalLoss()
        criterion_vo = FocalLoss()
        criterion_co = FocalLoss()
    elif config.loss == 'ClassBalancedLoss':
        criterion_gr = ClassBalancedLoss(samples_per_cls=weights_gr, no_of_classes=num_classes[0], loss_type='focal', beta=0.999, gamma=2.0)
        criterion_vo = ClassBalancedLoss(samples_per_cls=weights_vo, no_of_classes=num_classes[1], loss_type='focal', beta=0.999, gamma=2.0)
        criterion_co = ClassBalancedLoss(samples_per_cls=weights_co, no_of_classes=num_classes[2], loss_type='focal', beta=0.999, gamma=2.0)
    elif config.loss == 'OhemLoss':
        criterion_gr = OhemLoss(rate=1.0)
        criterion_vo = OhemLoss(rate=1.0)
        criterion_co = OhemLoss(rate=1.0)

    if config.optimizer.type == 'Adam':
        optimizer = Adam(params=model.parameters(), lr=config.optimizer.lr, amsgrad=False, weight_decay=1e-4)
    elif config.optimizer.type == 'SGD':
        optimizer = SGD(params=model.parameters(), lr=config.optimizer.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    scheduler_flg = False
    if config.scheduler.type == 'cosine':
        scheduler_flg = True
        scheduler = CosineAnnealingLR(optimizer, 
                                      T_max=config.scheduler.t_max, 
                                      eta_min=config.scheduler.eta_min)
    elif config.scheduler.type == 'cosine-warmup':
        scheduler_flg = True
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, 
                                                  T_0=config.scheduler.t_0,
                                                  T_mult=config.scheduler.t_mult,
                                                  eta_max=config.scheduler.eta_max,
                                                  T_up=config.scheduler.t_up, 
                                                  gamma=config.scheduler.gamma)
    elif config.scheduler.type == 'step':
        scheduler_flg = True
        scheduler = StepLR(optimizer, 
                           step_size=config.scheduler.step_size, 
                           gamma=config.scheduler.gamma)
    elif config.scheduler.type == 'reduce':
        scheduler_flg = True
        scheduler = ReduceLROnPlateau(optimizer, 
                                      factor=config.scheduler.factor, 
                                      patience=config.scheduler.patience,
                                      min_lr=config.scheduler.min_lr)


    best_epoch = -1
    best_val_score = -np.inf
    mb = master_bar(range(config.epochs))

    train_loss_list = []
    val_loss_list = []
    val_score_list = []

    counter = 0

    for epoch in mb:
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for images, labels_gr, labels_vo, labels_co in progress_bar(train_loader, parent=mb):
            images = Variable(images).to(device)
            labels_gr = Variable(labels_gr).to(device)
            labels_vo = Variable(labels_vo).to(device)
            labels_co = Variable(labels_co).to(device)

            if config.loss == 'OhemLoss':
                if epoch < config.epochs * 0.2:
                    new_rate = 1.0
                elif epoch < config.epochs * 0.4:
                    new_rate = 0.8
                elif epoch < config.epochs * 0.6:
                    new_rate = 0.75
                elif epoch < config.epochs * 0.8:
                    new_rate = 0.7
                else:
                    new_rate = 0.6

                criterion_gr.update_rate(new_rate)
                criterion_vo.update_rate(new_rate)
                criterion_co.update_rate(new_rate)

            r = np.random.rand()
            mix_params = config.augmentation.mix_params
            if r < mix_params.mixup:
                images, targets = mixup(images, labels_gr, labels_vo, labels_co, 1.0)
                preds_gr, preds_vo, preds_co = model(images)
                loss = mixup_criterion(preds_gr, preds_vo, preds_co, targets, criterion_gr, criterion_vo, criterion_co)
            elif r < (mix_params.mixup + mix_params.cutmix):
                images, targets = cutmix(images, labels_gr, labels_vo, labels_co, 1.0)
                preds_gr, preds_vo, preds_co = model(images)
                loss = cutmix_criterion(preds_gr, preds_vo, preds_co, targets, criterion_gr, criterion_vo, criterion_co)
            else:
                preds_gr, preds_vo, preds_co = model(images.float())
                loss = criterion_gr(preds_gr, labels_gr) \
                       + criterion_vo(preds_vo, labels_vo) \
                       + criterion_co(preds_co, labels_co)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        train_loss_list.append(avg_loss)

        model.eval()
        valid_gr_preds = np.zeros((len(valid_loader.dataset), num_classes[0]))
        valid_vo_preds = np.zeros((len(valid_loader.dataset), num_classes[1]))
        valid_co_preds = np.zeros((len(valid_loader.dataset), num_classes[2]))
        avg_val_loss = 0.

        for i, (images, labels_gr, labels_vo, labels_co) in enumerate(valid_loader):
            images = Variable(images).to(device)
            labels_gr = Variable(labels_gr).to(device)
            labels_vo = Variable(labels_vo).to(device)
            labels_co = Variable(labels_co).to(device)

            preds_gr, preds_vo, preds_co = model(images.float())

            loss_gr = criterion_gr(preds_gr, labels_gr)
            loss_vo = criterion_vo(preds_vo, labels_vo)
            loss_co = criterion_co(preds_co, labels_co)

            valid_gr_preds[i * config.batch_size: (i + 1) * config.batch_size] = preds_gr.cpu().detach().numpy()
            valid_vo_preds[i * config.batch_size: (i + 1) * config.batch_size] = preds_vo.cpu().detach().numpy()
            valid_co_preds[i * config.batch_size: (i + 1) * config.batch_size] = preds_co.cpu().detach().numpy()
            avg_val_loss += (loss_gr.item() + loss_vo.item() + loss_co.item()) / len(valid_loader)

        recall_gr = recall_score(y_gr_val, np.argmax(valid_gr_preds, axis=1), average='macro')
        recall_vo = recall_score(y_vo_val, np.argmax(valid_vo_preds, axis=1), average='macro')
        recall_co = recall_score(y_co_val, np.argmax(valid_co_preds, axis=1), average='macro')

        val_score = np.average([recall_gr, recall_vo, recall_co], weights=[2,1,1])

        val_loss_list.append(avg_val_loss)
        val_score_list.append(val_score)

        if scheduler_flg and config.scheduler.type != 'reduce':
            scheduler.step()
        elif scheduler_flg and config.scheduler.type == 'reduce':
            scheduler.step(avg_val_loss)
        
        elapsed = time.time() - start_time
        mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} val_score: {val_score:.4f} val_gr_score: {recall_gr:.4f} val_vo_score: {recall_vo:.4f} val_co_score: {recall_co:.4f} time: {elapsed:.0f}s')
        logging.debug(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} val_score: {val_score:.4f} val_gr_score: {recall_gr:.4f} val_vo_score: {recall_vo:.4f} val_co_score: {recall_co:.4f} time: {elapsed:.0f}s')

        if best_val_score < val_score:
            best_epoch = epoch + 1
            best_val_score = val_score
            best_recall_gr = recall_gr
            best_recall_vo = recall_vo
            best_recall_co = recall_co
            best_valid_gr_preds = valid_gr_preds
            best_valid_vo_preds = valid_vo_preds
            best_valid_co_preds = valid_co_preds
            best_model = model.state_dict()
            counter = 0
        
        counter += 1
        if counter == config.early_stopping:
            break

    print('\n\n===================================\n')
    print(f'CV: {best_val_score}\n')
    print(f'BEST EPOCH: {best_epoch}')
    print(f'BEST RECALL GR: {best_recall_gr}')
    print(f'BEST RECALL VO: {best_recall_vo}')
    print(f'BEST RECALL CO: {best_recall_co}')
    logging.debug(f'\n\nCV: {best_val_score}\n')
    logging.debug(f'BEST EPOCH: {best_epoch}')
    logging.debug(f'BEST RECALL GR: {best_recall_gr}')
    logging.debug(f'BEST RECALL VO: {best_recall_vo}')
    logging.debug(f'BEST RECALL CO: {best_recall_co}\n\n')
    print('\n===================================\n\n')
    
    return best_model, [best_valid_gr_preds, best_valid_vo_preds, best_valid_co_preds], best_val_score, train_loss_list, val_loss_list, val_score_list


def relu_replace(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Sequential):
            for child_name2, child2 in child.named_children():
                if isinstance(child2, nn.ReLU):
                    setattr(model.child_name, child_name2, Mish())
                else:
                    relu_replace(child2)
        else:
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, Mish())
            else:
                relu_replace(child)


def relu_replace(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Sequential):
            for child_name2, child2 in child.named_children():
                if isinstance(child2, nn.ReLU):
                    setattr(model.child_name, child_name2, Mish())
                else:
                    relu_replace(child2)
        else:
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, Mish())
            else:
                relu_replace(child)


def bn_replace(model):
    for child_name, child in model.named_children():
        if child_name != 'downsample':
            if isinstance(child, nn.Sequential):
                for child_name2, child2 in child.named_children():
                    if isinstance(child2, nn.BatchNorm2d):
                        setattr(model.child_name, child_name2, nn.GroupNorm(num_channels=child2.num_features, 
                                                                                                           num_groups=32))
                    else:
                        bn_replace(child2)
            else:
                if isinstance(child, nn.BatchNorm2d):
                    setattr(model, child_name, nn.GroupNorm(num_channels=child.num_features, 
                                                                                   num_groups=32))
                else:
                    bn_replace(child)
        else:
            bn = child[1]
            child[1] = nn.GroupNorm(num_channels=bn.num_features, 
                                                  num_groups=32)


def save_png(run_name, config, train_loss_list, val_loss_list, val_score_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax1.plot(range(len(train_loss_list)), train_loss_list, color='blue', linestyle='-', label='train_loss')
    ax1.plot(range(len(val_loss_list)), val_loss_list, color='green', linestyle='-', label='val_loss')
    ax1.legend()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title(f'Training and validation {config.loss}')
    ax1.grid()

    ax2.plot(range(len(val_score_list)), val_score_list, color='blue', linestyle='-', label='val_score')
    ax2.legend()
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('score')
    ax2.set_title('Training and validation score')
    ax2.grid()

    plt.savefig(f'../logs/{run_name}/learning_curve.png')


# https://www.kaggle.com/c/bengaliai-cv19/discussion/126504
# https://www.kaggle.com/c/bengaliai-cv19/discussion/128637
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets


def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets


def cutmix_criterion(preds1,preds2,preds3, targets, criterion1, criterion2, criterion3):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    return lam * criterion1(preds1, targets1) + (1 - lam) * criterion1(preds1, targets2) \
           + lam * criterion2(preds2, targets3) + (1 - lam) * criterion2(preds2, targets4) \
           + lam * criterion3(preds3, targets5) + (1 - lam) * criterion3(preds3, targets6)


def mixup_criterion(preds1,preds2,preds3, targets, criterion1, criterion2, criterion3):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    return lam * criterion1(preds1, targets1) + (1 - lam) * criterion1(preds1, targets2) \
           + lam * criterion2(preds2, targets3) + (1 - lam) * criterion2(preds2, targets4) \
           + lam * criterion3(preds3, targets5) + (1 - lam) * criterion3(preds3, targets6)


# def ohem_loss(rate, cls_pred, cls_target):
#     batch_size = cls_pred.size(0) 
#     ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

#     sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
#     keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
#     if keep_num < sorted_ohem_loss.size()[0]:
#         keep_idx_cuda = idx[:keep_num]
#         ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
#     cls_loss = ohem_cls_loss.sum() / keep_num
#     return cls_loss