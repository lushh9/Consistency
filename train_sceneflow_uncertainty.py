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


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pathlib import Path
from scipy.stats import pearsonr

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
        disp_loss += 0
    else:
        disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid_mask], disp_gt[valid_mask], reduction='mean')
    
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        mask = valid.bool() & torch.isfinite(i_loss)
        if mask.sum() == 0:
            disp_loss += 0
        else:
            disp_loss += i_weight * i_loss[mask].mean()
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
        return tensor.mean() if tensor.numel() > 0 else torch.tensor(0.0, device=accelerator.device)

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

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')


    return optimizer, scheduler

def criterion_uncertainty(u, la, alpha, beta, y, mask):
    weight_reg = 0.05
    valid_mask = mask.bool() & torch.isfinite(u) & torch.isfinite(la) & torch.isfinite(alpha) & torch.isfinite(beta) & torch.isfinite(y)

    if valid_mask.sum() == 0:
        loss = 0
        return loss

    # our loss function
    om = 2 * beta * (1 + la)
    # len(u): size
    loss = torch.sum(
        (0.5 * torch.log(np.pi / la) - alpha * torch.log(om) +
         (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) +
         torch.lgamma(alpha) - torch.lgamma(alpha+0.5))[valid_mask]
    ) / torch.sum(valid_mask == True)

    lossr = weight_reg * (torch.sum((torch.abs(u - y) * (2 * la + alpha))
                                             [valid_mask])) / torch.sum(valid_mask == True)
    loss = loss + lossr
    return loss

@hydra.main(version_base=None, config_path='config', config_name='train_sceneflow')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='fp16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb', kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs={'wandb': cfg.wandb})

    train_dataset = datasets.fetch_dataloader(cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=True, num_workers=int(8), drop_last=True)

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
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device)

    total_step = 0
    epoch = 0
    should_keep_training = True
    
    while should_keep_training:
        active_train_loader = train_loader

        model.train()
        model.freeze_bn()
        #model.module.freeze_bn()
        for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            #_, left, right, disp_gt, valid = [x for x in data]

            left = data['img1']
            right = data['img2']
            disp_gt = data['disp']
            valid = data['valid']

            with accelerator.autocast():
                disp_init_pred, disp_preds, depth_mono, confidence = model(left, right, iters=cfg.train_iters)
            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, accelerator, max_disp=cfg.max_disp)
            
            # 不确定性损失
            #print('disp_preds:', disp_preds[-1].shape)
            invalid_mask = torch.logical_or(disp_preds[-1] <= 0.0, disp_preds[-1] >= cfg.max_disp)
            la = confidence['la']
            alpha = confidence['alpha']
            beta = confidence['beta']
            uncertainty_loss = criterion_uncertainty(disp_preds[-1], la, alpha, beta, disp_gt, ~invalid_mask)

            # 
            tqdm.write(f"Epoch: {epoch}, total_steps: {total_step}, epe: {metrics['epe']:.4f}, loss: {loss:.6f}, uncertainty_loss: {uncertainty_loss:.6f}")
            #print('Epoch:',epoch, 'total_steps:',total_steps, 'epe:',metrics['epe'], 'loss:',loss, 'uncertainty_loss:',uncertainty_loss)

            loss = loss + uncertainty_loss

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

            if (total_step > 0) and (total_step % cfg.save_frequency == cfg.save_frequency - 1):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step + 1))
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    del model_save
        
            if (total_step > 0) and (total_step % cfg.val_frequency == cfg.val_frequency - 1):

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
                            with open('./checkpoints/sceneflow/uncertainty1/pearson_debug.txt', 'a') as log_file1:
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
                    val = (disp_gt.abs().flatten() < 768)
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

                with open('./checkpoints/sceneflow/uncertainty1/test_sceneflow.txt', 'a') as log_file2:
                    log_file2.write(f"Validation Scene Flow: {epe}, {d1}, {pearson_avg}\n")

                #print("Validation Scene Flow: %f, %f" % (epe, d1))
                tqdm.write(f"Validation Scene Flow: EPE={epe}, D1={d1}%, Pearson={pearson_avg}")
                print("Validation Scene Flow: %f, %f, %f" % (epe, d1, pearson_avg))

                '''padder = InputPadder(left.shape, divis_by=32)
                left, right = padder.pad(left, right)
                with torch.no_grad():
                    disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                disp_pred = padder.unpad(disp_pred)
                assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                epe = torch.abs(disp_pred - disp_gt)
                out = (epe > 1.0).float()
                epe = torch.squeeze(epe, dim=1)
                out = torch.squeeze(out, dim=1)
                disp_gt = torch.squeeze(disp_gt, dim=1)
                epe, out = accelerator.gather_for_metrics((epe[(valid >= 0.5) & (disp_gt.abs() < 192)].mean(), out[(valid >= 0.5) & (disp_gt.abs() < 192)].mean()))
                elem_num += epe.shape[0]
                for i in range(epe.shape[0]):
                    total_epe += epe[i]
                    total_out += out[i]
                accelerator.log({'val/epe': total_epe / elem_num, 'val/d1': 100 * total_out / elem_num}, total_step)'''

                model.train()
                model.freeze_bn()
                #model.module.freeze_bn()

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save
    
    accelerator.end_training()

if __name__ == '__main__':
    main()
