import os
import logging
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from train_fn import train_one_epoch, validate
from utils import CheckpointManager, seed_everything, seed_worker

log = logging.getLogger(__name__)
tqdm = partial(tqdm, dynamic_ncols=True)
trange = partial(trange, dynamic_ncols=True)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    from omegaconf import OmegaConf; print(OmegaConf.to_yaml(cfg))
    
    log.info(f"Run path: {Path.cwd()}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device(f'cuda' if cfg.gpu is not None else 'cpu')
    log.info(f"Use device {device} for training")

    # Reproducibility
    seed_everything(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # create tensorboard writer
    writer = SummaryWriter()
    
    # training dataset and dataloader
    g = torch.Generator()
    g.manual_seed(cfg.seed)     # To guarantee reproducibility
    cfg.data.train.batch_size = cfg.optim.batch_size
    if cfg.data.train.in_memory:
        log.info(f'Loading training dataset in memory')
    train_dataset = hydra.utils.instantiate(cfg.data.train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.optim.batch_size, shuffle=True, num_workers=cfg.optim.num_workers, worker_init_fn=seed_worker, generator=g)
    log.info(f'[TRAIN] {train_dataset}')

    # validation dataset and dataloader
    g = torch.Generator()
    g.manual_seed(cfg.seed)     # To guarantee reproducibility
    cfg.data.validation.batch_size = cfg.optim.val_batch_size
    if cfg.data.validation.in_memory:
        log.info(f'Loading validation dataset in memory')
    valid_dataset = hydra.utils.instantiate(cfg.data.validation)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.optim.val_batch_size, num_workers=cfg.optim.num_workers, worker_init_fn=seed_worker, generator=g)
    log.info(f'[VALID] {valid_dataset}')

    # create model
    torch.hub.set_dir(cfg.model.cache_folder)
    skip_weights_loading = cfg.optim.resume or cfg.model.pretrained
    model = hydra.utils.instantiate(cfg.model.module, skip_weights_loading=skip_weights_loading)
    # move model to device
    model.to(device)
    model_param_string = ', '.join(f'{k}={v}' for k, v in cfg.model.module.items() if not k.startswith('_'))
    log.info(f"[MODEL] {cfg.model.name}({model_param_string})")

    # build the optimizer
    optimizer = hydra.utils.instantiate(cfg.optim.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.optim.lr_scheduler, optimizer)

    # optionally load pre-trained weights
    if cfg.model.pretrained and cfg.model.resume is not None:
        if cfg.model.pretrained.startswith('http://') or cfg.model.pretrained.startswith('https://'):
            pre_trained_model = torch.hub.load_state_dict_from_url(
                cfg.model.pretrained, map_location=device, model_dir=cfg.model.cache_folder)
        else:
            pre_trained_model = torch.load(cfg.model.pretrained, map_location=device)
        model.load_state_dict(pre_trained_model['model'])
        log.info(f"[PRETRAINED]: {cfg.model.pretrained}")
    
    start_epoch = 0
    best_metrics = {}

    train_log_path = 'train_log.csv'
    valid_log_path = 'valid_log.csv'

    train_log = pd.DataFrame()
    valid_log = pd.DataFrame()

    # optionally resume from a saved checkpoint
    if cfg.optim.resume:
        assert Path('last.pth').exists(), 'Cannot find checkpoint for resuming.'
        checkpoint = torch.load('last.pth', map_location=device)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])

        start_epoch = checkpoint['epoch'] + 1
        best_metrics = checkpoint['best_metrics']

        train_log = pd.read_csv(train_log_path, index_col=0, header=[0,1])
        valid_log = pd.read_csv(valid_log_path, index_col=0, header=[0,1])
        log.info(f"[RESUME] Resuming from epoch {start_epoch}")
    
    # checkpoint manager
    ckpt_dir = Path('best_models')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_manager = CheckpointManager(ckpt_dir, current_best=best_metrics)

    # Train loop
    log.info(f"Training ...")
    progress = trange(start_epoch, cfg.optim.epochs, initial=start_epoch)
    for epoch in progress:
        # train
        train_metrics = train_one_epoch(train_loader, model, optimizer, device, writer, epoch, cfg)
        scheduler.step()  # update lr scheduler

        # convert for pandas
        train_metrics = pd.DataFrame(train_metrics, index=[epoch]).rename_axis('epoch')
        train_log = pd.concat([train_log, train_metrics])
        train_log.to_csv(train_log_path)

        # evaluation
        if (epoch + 1) % cfg.optim.val_freq == 0:
            valid_metrics = validate(valid_loader, model, device, epoch, cfg)
            
            for metric, value in valid_metrics.items():  # log to tensorboard
                writer.add_scalar(f'valid/{metric}', value, epoch)
                
            # save only if best on some metric (via CheckpointManager)
            best_metrics = ckpt_manager.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': valid_metrics
            }, valid_metrics, epoch)

            # save last checkpoint for resuming
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_metrics': best_metrics,
            }, 'last.pth')
                
            # convert for pandas
            valid_metrics = pd.DataFrame(valid_metrics, index=[epoch]).rename_axis('epoch')
            valid_log = pd.concat([valid_log, valid_metrics])
            valid_log.to_csv(valid_log_path)

    log.info("Training ended. Exiting....")


if __name__ == "__main__":
    main()