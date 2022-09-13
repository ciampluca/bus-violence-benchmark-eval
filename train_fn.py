# -*- coding: utf-8 -*-
from functools import partial
import logging
from pathlib import Path

import hydra
import torch
from tqdm import tqdm
import pandas as pd

tqdm = partial(tqdm, dynamic_ncols=True)

# creating logger
log = logging.getLogger(__name__)


def train_one_epoch(dataloader, model, optimizer, device, writer, epoch, cfg):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    criterion = hydra.utils.instantiate(cfg.optim.loss)

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        frames, labels, _ = sample
        frames, labels = frames.to(device), labels.to(device)

        # computing outputs
        preds = model(frames)

        # computing loss and backwarding it
        loss = criterion(preds, labels)
        loss.backward()

        batch_metrics = {'loss': loss.item()}
        metrics.append(batch_metrics)

        postfix = {metric: f'{value:.3f}' for metric, value in batch_metrics.items()}
        progress.set_postfix(postfix)

        if (i + 1) % cfg.optim.batch_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % cfg.optim.log_every == 0:
            batch_metrics.update({'lr': optimizer.param_groups[0]['lr']})
            n_iter = epoch * n_batches + i
            for metric, value in batch_metrics.items():
                writer.add_scalar(f'train/{metric}', value, n_iter)

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()
    
    return metrics


def _save_debug_metrics(metrics, epoch):
    
    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)
    
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(debug_dir / Path('validation_metrics_epoch_{}.csv'.format(epoch)), index=False)


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device
    criterion = hydra.utils.instantiate(cfg.optim.loss)

    metrics, debug_metrics = [], []
    n_videos = len(dataloader)
    progress = tqdm(dataloader, total=n_videos, desc='EVAL', leave=False)
    for i, sample in enumerate(progress):
        b_frames, b_labels, b_ids = sample
        
        # Un-batching
        # TODO not efficient, should be done in parallel
        for frames, label, video_id in zip(b_frames, b_labels, b_ids):
            frames, label = torch.unsqueeze(frames, dim=0).to(validation_device), torch.unsqueeze(label, dim=0).to(validation_device)

            # Computing pred
            pred = model(frames)

            # Computing loss
            loss = criterion(pred, label)

            # Accumulate video metric for debugging
            pred_prob = (torch.sigmoid(pred)).item() if criterion.__class__.__name__.endswith("WithLogitsLoss") else pred.item()
            debug_metrics.append({
                'video_id': video_id,
                'bce_loss': loss.item(),
                'pred_prob': pred_prob,
                'pred_label': int(pred_prob > 0.5),
                'target_label': int(label.item()),
            })
            
            # Accumulate video metric
            metrics.append({
                'accuracy': int((pred_prob > 0.5) == label.item()),
                'bce_loss': loss.item(),
            })

            if cfg.optim.debug and epoch % cfg.optim.debug_freq == 0:
                # TODO eventually save video/frames or other
                pass

            progress.set_description('EVAL')
            
    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()
    
    if cfg.optim.debug:
        _save_debug_metrics(debug_metrics, epoch)
    
    return metrics


@torch.no_grad()
def predict(dataloader, model, device, cfg, outdir, debug=0, csv_file_name='preds.csv'):
    """ Make predictions on data. """
    model.eval()
    criterion = hydra.utils.instantiate(cfg.optim.loss)
    
    metrics = []
    n_videos = len(dataloader)
    progress = tqdm(dataloader, total=n_videos, desc='PRED', leave=False)
    for i, sample in enumerate(progress):
        b_frames, b_labels, b_ids = sample
        
        # Un-batching
        # TODO not efficient, should be done in parallel
        for frames, label, video_id in zip(b_frames, b_labels, b_ids):
            frames, label = torch.unsqueeze(frames, dim=0).to(device), torch.unsqueeze(label, dim=0).to(device)
            
            # Computing pred
            pred = model(frames)
            
            # Accumulate video metric
            pred_prob = (torch.sigmoid(pred)).item() if criterion.__class__.__name__.endswith("WithLogitsLoss") else pred.item()
            metrics.append({
                'video_id': video_id,
                'pred_prob': pred_prob,
                'pred_label': int(pred_prob > 0.5),
                'target_label': int(label.item()),
            })
            
            if outdir and debug:
                # TODO eventually save video/frames or other
                pass        
            
            progress.set_description('PRED')
            
    metrics = pd.DataFrame(metrics)       

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(outdir / csv_file_name)