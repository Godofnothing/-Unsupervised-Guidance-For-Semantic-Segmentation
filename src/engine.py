import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable, Dict, Optional, Callable
from collections import OrderedDict


def train_epoch(
    model : nn.Module, 
    data_loader : Iterable,
    criterion : nn.Module,
    metrics: Dict[str, nn.Module],
    optimizer : torch.optim.Optimizer,
    device : torch.device
):
    assert isinstance(metrics, dict), "You have to pass dict of metrics"

    model.train()
    running_loss = 0
    running_metrics = {key : 0 for key in metrics}
    
    for images, masks in data_loader:
        images, masks = images.to(device), masks.to(device)

        with torch.cuda.amp.autocast(enabled=False):
            # get model output
            mask_logits = model(images)
            if isinstance(mask_logits, OrderedDict):
                mask_logits = mask_logits['out'] 
            # compute loss
            loss = criterion(mask_logits, masks)
        # get num classes for further convenience
        num_classes = mask_logits.shape[1]
        # make gradient step  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # statistics
        running_loss += loss.item() * images.size(0)
        # get predictions from logits
        pred_masks = torch.argmax(mask_logits, dim=1)   
        # get one hot masks     
        masks_one_hot = F.one_hot(masks, num_classes=num_classes)
        pred_masks_one_hot =   F.one_hot(pred_masks, num_classes=num_classes)
        # update metric statistics
        for key in metrics:
            running_metrics[key] += metrics[key](pred_masks_one_hot, masks_one_hot).item()
        
    total_samples = len(data_loader.dataset)
        
    epoch_loss = running_loss / total_samples
    epoch_metrics  = {f"train/{key}" : running_metrics[key] / total_samples for key in running_metrics}

    return {"train/loss" : epoch_loss, **epoch_metrics}


@torch.no_grad()
def val_epoch(
    model : nn.Module, 
    data_loader : Iterable,
    criterion : nn.Module,
    metrics: Dict[str, nn.Module],
    device : torch.device
):
    model.eval()
    running_loss = 0
    running_metrics = {key : 0 for key in metrics}
    
    for images, masks in data_loader:
        images, masks = images.to(device), masks.to(device)

        with torch.cuda.amp.autocast(enabled=False):
            # get model output
            mask_logits = model(images)
            if isinstance(mask_logits, OrderedDict):
                mask_logits = mask_logits['out']  
            # compute loss
            loss = criterion(mask_logits, masks)  
        # get num classes for further convenience
        num_classes = mask_logits.shape[1] 
        # statistics
        running_loss += loss.item() * images.size(0)
        # get predictions from logits
        pred_masks = torch.argmax(mask_logits, dim=1)   
        # get one hot masks     
        masks_one_hot = F.one_hot(masks, num_classes=num_classes)
        pred_masks_one_hot =   F.one_hot(pred_masks, num_classes=num_classes)
        # update metric statistics
        for key in metrics:
            running_metrics[key] += metrics[key](pred_masks_one_hot, masks_one_hot).item()
        
    total_samples = len(data_loader.dataset)
        
    epoch_loss = running_loss / total_samples
    epoch_metrics  = {f"val/{key}" : running_metrics[key] / total_samples for key in running_metrics}

    return {"val/loss" : epoch_loss, **epoch_metrics}


def train(
    model : nn.Module,
    dataloaders : Dict[str, Iterable],
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    metrics: Dict[str, nn.Module],
    num_epochs: int,
    target_metric: str,
    scheduler : Optional[Callable] = None,
    device='cpu',
    save_checkpoint_dir='',
    wandb_logger=False
):
    history = {
        "train" : {"loss" : [], "acc" : []},
        "val"   : {"loss" : [], "acc" : []}
    }

    best_val_metric = 0.0

    if wandb_logger:
        # define epoch as x-axis metric
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        if dataloaders.get("val"):
            wandb.define_metric("val/*", step_metric="epoch")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # run train epoch
        train_stats = train_epoch(
            model, dataloaders["train"], criterion, metrics, optimizer,
            device=device,
        )

        log_str = ' '.join([f'{k} {v:.4f}' for k, v in train_stats.items()])
        print(log_str, flush=True)
        if wandb_logger:
            print({"epoch" : epoch, **train_stats}, flush=True)
            for key, value in train_stats.items():
                print(type(value))
            wandb.log({"epoch" : epoch, **train_stats})

        # run validation epoch
        if dataloaders.get("val"):
            val_stats = val_epoch(
                model, dataloaders["val"], criterion, metrics,
                device=device,
            )

            log_str = ' '.join([f'{k} {v:.4f}' for k, v in val_stats.items()])
            print(log_str, flush=True)
            if wandb_logger:
                print({"epoch" : epoch, **val_stats})
                wandb.log({"epoch" : epoch, **val_stats})

            if val_stats[f"val/{target_metric}"] > best_val_metric:
                best_val_metric = val_stats[f"val/{target_metric}"]
                # save best model
                if save_checkpoint_dir:
                    torch.save({
                        "epoch" : epoch,
                        "model_state_dict" : model.state_dict(),
                        "optimizer_state_dict" : optimizer.state_dict()
                    }, f"{save_checkpoint_dir}/best.pt")

        if scheduler:
            scheduler.step()

    # save last model
    if save_checkpoint_dir:
        torch.save({
            "epoch" : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()
        }, f"{save_checkpoint_dir}/last.pt")

    return history
