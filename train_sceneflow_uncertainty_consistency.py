import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
# from util import InputPadder
from core.utils.utils import InputPadder
from core.monster_uncertainty import Monster
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
import h5py
import gc
from pathlib import Path
from scipy.stats import pearsonr
from filelock import FileLock

def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, accelerator, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    #init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
    #disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    
    valid_mask = valid.bool() & torch.isfinite(disp_init_pred) & torch.isfinite(disp_gt)
    if valid_mask.sum() == 0:
        disp_loss = disp_loss + 0
    else:
        disp_loss = disp_loss + 1.0 * F.smooth_l1_loss(disp_init_pred[valid_mask], disp_gt[valid_mask], reduction='mean')
    
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        mask = valid.bool() & torch.isfinite(i_loss)
        if mask.sum() == 0:
            disp_loss = disp_loss + 0
        else:
            disp_loss = disp_loss + i_weight * i_loss[mask].mean()
        #disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    epe = epe[torch.isfinite(epe)]

    '''if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }'''

    def safe_mean(tensor):
        #return tensor.mean() if tensor.numel() > 0 else torch.tensor(0.0, device=accelerator.device)
        return tensor.mean() if tensor.numel() > 0 else torch.Tensor([0.0]).cuda()

    metrics = {
        'epe': safe_mean(epe),
        '1px': safe_mean((epe < 1).float()),
        '3px': safe_mean((epe < 3).float()),
        '5px': safe_mean((epe < 5).float()),
    }

    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0}, 
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    '''for group in optimizer.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = group['lr']/25
        if 'max_lr' not in group:
            group['max_lr'] = group['lr']
        if 'min_lr' not in group:
            group['min_lr'] = group['lr']/250000

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear', last_epoch=69999)'''

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def criterion_uncertainty(u, la, alpha, beta, y, mask):
    weight_reg = 0.05
    max_disp = 192
    mag = torch.sum(y**2, dim=1).sqrt()
    valid = (mask >= 0.5) & (mag < max_disp).unsqueeze(1)
    valid_mask = valid.bool() & torch.isfinite(u) & torch.isfinite(la) & torch.isfinite(alpha) & torch.isfinite(beta) & torch.isfinite(y)

    #if valid_mask.sum() == 0:
    if torch.sum(valid_mask == True) == 0:
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

    # our loss function
    om = 2 * beta * (1 + la)
    om = torch.clamp(om, min=1e-10)
    # len(u): size
    '''loss = torch.sum(
        (0.5 * torch.log(np.pi / la) - alpha * torch.log(om) +
         (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) +
         torch.lgamma(alpha) - torch.lgamma(alpha+0.5))[valid_mask]
    ) / torch.sum(valid_mask == True)

    lossr = weight_reg * (torch.sum((torch.abs(u - y) * (2 * la + alpha))
                                             [valid_mask])) / torch.sum(valid_mask == True)'''

    NLL_loss = 0.5 * torch.log(np.pi / la) - alpha * torch.log(om) + \
        (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) + \
            torch.lgamma(alpha) - torch.lgamma(alpha+0.5)
    loss = NLL_loss[valid_mask.bool() & torch.isfinite(NLL_loss)].mean()

    R_loss = torch.abs(u - y) * (2 * la + alpha)
    lossr = weight_reg * R_loss[valid_mask.bool() & torch.isfinite(R_loss)].mean()

    loss = loss + lossr
    return loss

def CRL_loss_base(con, score, iter_i):
    permi = torch.randperm(len(con))
    print('len:', len(con))
    print('con:', con)
    print('con_max:', torch.max(con))
    print('score:', score)
    all_CRL_loss = 0

    con = con[permi]
    score = score[permi]
    for i in range(iter_i):
        con1 = torch.roll(con, i + 1)
        score1 = torch.roll(score, i + 1)
        for_see = torch.nn.functional.relu(-torch.sign(con1 - con) *
                                           (score1 - score) + torch.abs(con1 - con))

        all_CRL_loss = all_CRL_loss + torch.sum(for_see) / len(con)
        print('all_CRL_loss:', all_CRL_loss)
    
    return all_CRL_loss / iter_i


# 获得训练一致性和最大softmax输出，将其输入到CRL_loss_base函数中计算loss
def rank_loss_function_constant(model, sample, path, cfg, accelerator):
    model.eval()
    #model.train()
    #model.freeze_bn()
    #model.module.freeze_bn()

    img1 = sample['img1']
    img2 = sample['img2']
    padder = InputPadder(img1.shape, divis_by=32)
    img1, img2 = padder.pad(img1, img2)

    gt_disp = sample["disp"]
    mag = torch.sum(gt_disp**2, dim=1).sqrt()
    mask = (gt_disp >= 0.5) & (mag < cfg.max_disp).unsqueeze(1)
    mask = mask.squeeze(1)

    with accelerator.autocast():
        disp_pr, confidence = model(img1, img2, iters=cfg.valid_iters, test_mode=True)

    disp_pr = padder.unpad(disp_pr)
    confidence['la'] = padder.unpad(confidence['la'])
    confidence['alpha'] = padder.unpad(confidence['alpha'])
    confidence['beta'] = padder.unpad(confidence['beta'])
    confidence['aleatoric'] = padder.unpad(confidence['aleatoric'])
    confidence['epistemic'] = padder.unpad(confidence['epistemic'])

    constant_path = os.path.join(path, 'constant_change.h5')
    constant_batch = []

    lock_file3 = f"{constant_path}.lock"
    lock3 = FileLock(lock_file3)
    
    with lock3:
        with h5py.File(constant_path, 'r') as f3:
            for k in range(len(sample["img1_path"])):
                para = f3[sample["img1_path"][k]]
                constant_batch.append(torch.from_numpy(para[()]))
                #constant_batch.append(f1[sample["img1_path"][k]])

    #constant_batch = torch.stack(constant_batch).unsqueeze(1).to(accelerator.device)
    constant_batch1 = torch.stack(constant_batch)
    print('constant_batch_shape1:', constant_batch1.shape)
    #constant_batch1 = constant_batch1.view(disp_pr.shape)
    constant_batch1 = constant_batch1.to(accelerator.device)
    '''if len(constant_batch1.shape) != 4:
        constant_batch1 = constant_batch1.unsqueeze(1)
    constant_batch1 = constant_batch1.to(accelerator.device)'''

    print('constant_batch_shape1:', constant_batch1.shape)
    score_batch = confidence['epistemic'].squeeze(1)

    valid1 = mask.bool()&torch.isfinite(score_batch)
    print('valid1_shape1:', valid1.shape)

    constant_batch1 = constant_batch1.view(score_batch.shape)

    print('constant_batch_shape1:', constant_batch1.shape)
    print('score_batch_shape1:', score_batch.shape)

    print('test1:', score_batch[valid1].shape)
    print('test2:', constant_batch1[valid1].shape)
    constant_batch1 = constant_batch1[valid1]
    score_batch1 = score_batch[valid1]

    if constant_batch1.numel() == 0 or score_batch1.numel() == 0:
        CRL_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
        return CRL_loss, disp_pr, confidence, gt_disp

    '''score_batch1 = torch.where(valid1, score_batch, torch.tensor(0.0, device=accelerator.device))
    print('score_batch1_shape1:', score_batch1.shape)
    constant_batch1 = torch.where(valid1, constant_batch1, torch.tensor(0.0, device=accelerator.device))
    print('constant_batch_shape2:', constant_batch1.shape)'''

    #constant_batch1 = constant_batch1.flatten()

    constant_batch1 = (constant_batch1 - torch.min(constant_batch1)) / (
                torch.max(constant_batch1) - torch.min(constant_batch1) + 1e-7)

    #pearson = pearson_correlation_coefficient(score_batch2.detach(), pred_disp, gt_disp.detach())
    #score_batch1 = score_batch1.flatten()

    #del img1, img2, pred_disp, para
    del img1, img2, para

    #return CRL_loss_base(constant_batch, score_batch1, 1), pearson, disp_pr, confidence, gt_disp
    return CRL_loss_base(constant_batch1, score_batch1, 1), disp_pr, confidence, gt_disp


def constant_for_dataset(model, train_loader, epoch, path, threshold, cfg, accelerator):
    model.eval()

    constant_path = os.path.join(path, 'constant_change.h5')
    out_path = os.path.join(path, 'out_temp.h5')

    lock_file1 = f"{constant_path}.lock"
    lock_file2 = f"{out_path}.lock"
    lock1 = FileLock(lock_file1)
    lock2 = FileLock(lock_file2)

    count = 0
    for sample in tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
        print('count:', count)
        img1 = sample['img1']
        img2 = sample['img2']
        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)
        #img1 = sample['img1'].to(torch.device("cuda"))
        #img2 = sample['img2'].to(torch.device("cuda"))
        #_, iter_preds, confidence = model(img1, img2, iters=cfg.train_iters)
        with accelerator.autocast():
            disp_pr, confidence = model(img1, img2, iters=cfg.valid_iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr)
        epistemic = padder.unpad(confidence['epistemic'])
        print('epistemic:', epistemic.shape)
        print('pred_disp:', disp_pr.shape)
        print('gt_disp', sample["disp"].shape)

        # 分别存放一致性和上一轮disparity图

        if epoch == 0:
            # 同时打开一致性和disparity
            # w写模式，文件不存在则创建，文件存在则清空文件内容
            for k in range(len(sample["img1_path"])):
                key = sample["img1_path"][k]
                mask_invalid = 0
                with lock1:
                    with h5py.File(constant_path, 'a') as f1:
                        if key in f1:
                            del f1[key]
                        f1[sample["img1_path"][k]] = torch.zeros(sample["disp"].squeeze(1).shape[1:]).float()
                        #f1[sample["img1_path"][k]] = torch.zeros(sample["disp"].shape[1:]).float()
                with lock2:
                    with h5py.File(out_path, 'a') as f2:
                        if key in f2:
                            del f2[key]
                        current_disp = disp_pr[k].detach().cpu()
                        mask_invalid = ~torch.isfinite(current_disp)
                        assert current_disp.shape == mask_invalid.shape
                        if mask_invalid.any():
                            current_disp[mask_invalid] = 0.0

                        f2[sample["img1_path"][k]] = current_disp
                        #f2[sample["img1_path"][k]] = disp_pr[k].detach().cpu()
                del key
            #torch.cuda.empty_cache()
        else:
            # a追加模型，可以修改或增加文件内数据，不删除其余的
            for k in range(len(sample["img1_path"])):
                key = sample["img1_path"][k]
                f1_temp = 0
                f2_temp = 0
                mask_invalid = 0
                with lock2:
                    with h5py.File(out_path, 'a') as f2:
                        if key in f2:
                            f2_temp = f2[sample["img1_path"][k]][()]
                            del f2[key]
                        
                        current_disp = disp_pr[k].detach().cpu()
                        mask_invalid = ~torch.isfinite(current_disp)
                        assert current_disp.shape == mask_invalid.shape
                        if mask_invalid.any():
                            #current_disp[mask_invalid] = torch.from_numpy(f2_temp)[mask_invalid]
                            current_disp = torch.where(mask_invalid, torch.from_numpy(f2_temp).to(current_disp.device), current_disp)

                        f2[sample["img1_path"][k]] = current_disp
                with lock1:
                    with h5py.File(constant_path, 'a') as f1:
                        #print('tc:', ((torch.from_numpy(f2_temp) - iter_preds[-1][k].cpu()).abs() / torch.from_numpy(f2_temp)))
                        #print('tc:', ((torch.from_numpy(f2_temp) - output["disp"][k].cpu()).abs() / torch.from_numpy(f2_temp)))
                        #constant_or = torch.where((torch.from_numpy(f2_temp) - output["disp"][k].cpu()).abs() < threshold, 1, 0)
                        constant_or = torch.where(((torch.from_numpy(f2_temp) - disp_pr[k].cpu()).abs() / (torch.from_numpy(f2_temp) + 1e-6)) < threshold, 1, 0)
                        if mask_invalid.any():
                            constant_or[mask_invalid] = 0
                        #constant_or = torch.where(((torch.from_numpy(f2_temp) - output["disp"][k].cpu()).abs() / (torch.from_numpy(f2_temp) + 1e-6)) < threshold, 1, 0)
                        if key in f1:
                            f1_temp = f1[sample["img1_path"][k]]
                            del f1[key]
                        f1[sample["img1_path"][k]] = torch.from_numpy(f1_temp[()]) + constant_or
                del key
            #torch.cuda.empty_cache()
        gc.collect()
        count += 1

    del img1, img2, disp_pr, confidence

# pearson
def pearson_correlation_coefficient(confidence, predicted_disp, true_disp):
    """
    计算整个批次的皮尔逊相关系数。
    
    参数:
    confidence -- 置信度图 (torch.Tensor) [B, H, W]
    predicted_disp -- 预测的视差图 (torch.Tensor) [B, H, W]
    true_disp -- 真实的视差图 (torch.Tensor) [B, H, W]
    epsilon -- 一个小常数，用于避免分母为零 (float)
    
    返回:
    avg_r -- 皮尔逊相关系数的平均值 (标量)
    """
    # 检查输入数据的形状是否一致
    if confidence.shape != predicted_disp.shape or predicted_disp.shape != true_disp.shape:
        raise ValueError("置信度图、预测视差图和真实视差图的形状必须相同")
    
    # 计算相对误差
    error = torch.abs(predicted_disp - true_disp) / (true_disp + 1e-6)
    
    # 初始化相关系数列表
    correlation_coefficients = []
    
    # 遍历批次中的每张图片
    for i in range(confidence.shape[0]):
        # 展平当前图片的误差和置信度图
        error_flat = error[i].view(-1)
        confidence_flat = confidence[i].view(-1)
        
        # 计算均值
        mean_error = torch.mean(error_flat)
        mean_confidence = torch.mean(confidence_flat)
        
        # 计算分子
        numerator = torch.sum((error_flat - mean_error) * (confidence_flat - mean_confidence))
        
        # 计算分母
        denominator = torch.sqrt(torch.sum((error_flat - mean_error) ** 2)) * torch.sqrt(torch.sum((confidence_flat - mean_confidence) ** 2))
        
        # 计算皮尔逊相关系数
        if denominator == 0:
            print("分母为零，无法计算相关系数")
            r = torch.tensor(0)
            return r
        
        r = numerator / denominator
        
        # 将当前图片的相关系数添加到列表中
        correlation_coefficients.append(r)
    
    # 计算相关系数的平均值
    avg_r = torch.mean(torch.stack(correlation_coefficients))
    print('pearson:', avg_r)
    
    return avg_r

@hydra.main(version_base=None, config_path='config', config_name='train_sceneflow')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    if cfg.num_gpu == 1:
        # 单卡
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    else:
        # 多卡
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    accelerator = Accelerator(mixed_precision='fp16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb', kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs={'wandb': cfg.wandb})

    cfg.whole_dataset = True
    cfg.sampler = True
    whole_dataset, train_sampler = datasets.fetch_dataloader(cfg)
    whole_loader = torch.utils.data.DataLoader(whole_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=True, sampler=train_sampler)
    cfg.whole_dataset = False
    cfg.sampler = False

    train_dataset = datasets.fetch_dataloader(cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=True, sampler=train_sampler)

    aug_params = {}
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=8, drop_last=False)

    model = Monster(cfg)

    if not cfg.restore_ckpt.endswith("None"):
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]

        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, whole_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, whole_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device)

    '''for sample1, sample2 in zip(train_loader, whole_loader):
        print(sample1["img1"].shape)
        print(sample2["img1"].shape)
        print('train_loader:', sample1["img1_path"])
        print('whole_loader:', sample2["img1_path"])
        break'''

    total_step = 0
    epoch = 0
    should_keep_training = True

    path = './checkpoints/sceneflow/consistency1/'
    threshold = 0.15
    
    while epoch != 10:
        active_train_loader = train_loader

        model.train()
        if cfg.num_gpu == 1:
            model.freeze_bn()
        else:
            model.module.freeze_bn()

        train_sampler.set_epoch(epoch)

        # 获得每个epoch的一致性
        #if epoch != 0:
        constant_for_dataset(model, whole_loader, epoch, path, threshold, cfg, accelerator)

        for data1, data2 in tqdm(zip(active_train_loader, whole_loader), dynamic_ncols=True, disable=not accelerator.is_main_process):

            model.train()
            if cfg.num_gpu == 1:
                model.freeze_bn()
            else:
                model.module.freeze_bn()

            left = data1['img1']
            right = data1['img2']
            disp_gt = data1['disp']
            valid = data1['valid']

            with accelerator.autocast():
                disp_init_pred, disp_preds, depth_mono, confidence = model(left, right, iters=cfg.train_iters)
            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, accelerator, max_disp=cfg.max_disp)

            # 一致性排名损失
            #constant_loss, pearson_sample, whole_disp_preds, whole_confidence, whole_disp_gt = rank_loss_function_constant(model, data2, path, cfg, accelerator)
            constant_loss, whole_disp_preds, whole_confidence, whole_disp_gt = rank_loss_function_constant(model, data2, path, cfg, accelerator)

            #model.train()
            #model.freeze_bn()
            #model.module.freeze_bn()

            # 不确定性损失
            '''invalid_mask = torch.logical_or(whole_disp_preds <= 0.0, whole_disp_preds >= cfg.max_disp)
            la = whole_confidence['la']
            alpha = whole_confidence['alpha']
            beta = whole_confidence['beta']
            uncertainty_loss = criterion_uncertainty(whole_disp_preds, la, alpha, beta, whole_disp_gt, ~invalid_mask)'''

            invalid_mask = torch.logical_or(disp_preds[-1] <= 0.0, disp_preds[-1] >= cfg.max_disp)
            la = confidence['la']
            alpha = confidence['alpha']
            beta = confidence['beta']
            uncertainty_loss = criterion_uncertainty(disp_preds[-1], la, alpha, beta, disp_gt, ~invalid_mask)

            # 
            tqdm.write(f"Epoch: {epoch}, total_steps: {total_step}, epe: {metrics['epe']}, loss: {loss}, uncertainty_loss: {uncertainty_loss}, constant_loss: {constant_loss}")
            #print('Epoch:',epoch, 'total_steps:',total_steps, 'epe:',metrics['epe'], 'loss:',loss, 'uncertainty_loss:',uncertainty_loss)

            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, dtype=torch.float16, device=accelerator.device)

            if not isinstance(uncertainty_loss, torch.Tensor):
                uncertainty_loss = torch.tensor(uncertainty_loss, dtype=torch.float16, device=accelerator.device)

            if not isinstance(constant_loss, torch.Tensor):
                constant_loss = torch.tensor(constant_loss, dtype=torch.float16, device=accelerator.device)

            if loss.item() != 0.0:
                loss = loss + uncertainty_loss + 5 * constant_loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_step += 1
                loss = accelerator.reduce(loss.detach(), reduction='mean')
                metrics = accelerator.reduce(metrics, reduction='mean')
                accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
                accelerator.log(metrics, total_step)
            else:
                total_step += 1

            ####visualize the depth_mono and disp_preds
            if total_step % 20 == 0 and accelerator.is_main_process:
                image1_np = left[0].squeeze().cpu().numpy()
                image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                image1_np = image1_np.astype(np.uint8)
                image1_np = np.transpose(image1_np, (1, 2, 0))

                image2_np = right[0].squeeze().cpu().numpy()
                image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                image2_np = image2_np.astype(np.uint8)
                image2_np = np.transpose(image2_np, (1, 2, 0))


                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
                
                accelerator.log({"disp_pred": wandb.Image(disp_preds_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"disp_gt": wandb.Image(disp_gt_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"depth_mono": wandb.Image(depth_mono_np, caption="step:{}".format(total_step))}, total_step)

            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step + 1))
                    model_save = accelerator.unwrap_model(model)
                    #torch.save(model_save.state_dict(), save_path)
                    torch.save({
                            "state_dict": model_save.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict()
                            }, save_path)
                    del model_save
        
            if (total_step > 0) and (total_step % cfg.val_frequency == 0):

                model.eval()
                #elem_num, total_epe, total_out = 0, 0, 0
                out_list, epe_list = [], []
                pearson_list = []
                count = 0

                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                    #_, left, right, disp_gt, valid = [x for x in data]

                    tqdm.write(f"count: {count}")
                    count += 1

                    left = data['img1']
                    right = data['img2']
                    disp_gt = data['disp']
                    valid_gt = data['valid'].cpu()

                    padder = InputPadder(left.shape, divis_by=32)
                    left, right = padder.pad(left, right)

                    with accelerator.autocast():
                        disp_pr, confidence = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    disp_pr = padder.unpad(disp_pr).detach().cpu()
                    conf = padder.unpad(confidence['epistemic']).detach().cpu()
                    disp_gt = disp_gt.cpu()
                    assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
                    epe = torch.abs(disp_pr - disp_gt)

                    max_disp = 192
                    mag = torch.sum(disp_gt**2, dim=1).sqrt()
                    mask = ((valid_gt >= 0.5) & (mag < max_disp)).unsqueeze(1)
                    pearson_batch = []

                    for i in range(conf.shape[0]):
                        #epe_np = epe[i][valid[i]].numpy()
                        #conf_np = conf[i][valid[i]].numpy()
                        valid_mask = mask[i].bool() & torch.isfinite(epe[i]) & torch.isfinite(conf[i])

                        if valid_mask.sum() < 2:
                            # 统计 NaN 和 Inf 数量
                            epe_nan = torch.isnan(epe[i]).sum().item()
                            epe_inf = torch.isinf(epe[i]).sum().item()
                            conf_nan = torch.isnan(conf[i]).sum().item()
                            conf_inf = torch.isinf(conf[i]).sum().item()

                            # 写入调试日志
                            with open('./checkpoints/sceneflow/consistency1/pearson_debug.txt', 'a') as log_file1:
                                log_file1.write(f"count: {count}\n")
                                if mask[i].bool().sum() == 0:
                                    log_file1.write("  WARNING: mask[i] is all zeros\n")
                                log_file1.write(f"  epe NaN: {epe_nan}, Inf: {epe_inf}\n")
                                log_file1.write(f"  conf NaN: {conf_nan}, Inf: {conf_inf}\n")
                            continue

                        epe_np = epe[i][valid_mask].numpy()
                        conf_np = conf[i][valid_mask].numpy()
                        r_batch, _ = pearsonr(conf_np, epe_np)
                        pearson_batch.append(r_batch)
        
                    if not pearson_batch:
                        r = 0
                    else:
                        r = np.mean(pearson_batch)
                        pearson_list.append(r)
                    #r = np.mean(pearson_batch)
                    tqdm.write(f"pearson_batch: {r}")
                    #print('pearson_batch:', r)
                    #pearson_list.append(r)

                    epe = epe.flatten()
                    val = (disp_gt.abs().flatten() < 192)
                    if(np.isnan(epe[val].mean().item())):
                        continue

                    out = (epe > 3.0)
                    epe_list.append(epe[val].mean().item())
                    out_list.append(out[val].cpu().numpy())

                epe_list = np.array(epe_list)
                out_list = np.concatenate(out_list)

                epe = np.mean(epe_list)
                d1 = 100 * np.mean(out_list)
                pearson_avg = np.mean(pearson_list)

                with open('./checkpoints/sceneflow/consistency1/test_sceneflow.txt', 'a') as log_file2:
                    log_file2.write(f"Validation Scene Flow: {epe}, {d1}, {pearson_avg}\n")

                #print("Validation Scene Flow: %f, %f" % (epe, d1))
                tqdm.write(f"Validation Scene Flow: EPE={epe}, D1={d1}%, Pearson={pearson_avg}")
                print("Validation Scene Flow: %f, %f, %f" % (epe, d1, pearson_avg))

                #model.train()
                #model.freeze_bn()
                #model.module.freeze_bn()

            if total_step == cfg.total_step:
                should_keep_training = False
                break
                
        if accelerator.is_main_process:
            save_path = Path(cfg.save_path + '/%d_%d.pth' % (epoch, total_step))
            model_save = accelerator.unwrap_model(model)
            #torch.save(model_save.state_dict(), save_path)
            torch.save({
                    "state_dict": model_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict()
                    }, save_path)
            del model_save
        
        epoch += 1

    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        #torch.save(model_save.state_dict(), save_path)
        torch.save({
                "state_dict": model_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict()
                }, save_path)
        del model_save
    
    accelerator.end_training()

if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    main()
