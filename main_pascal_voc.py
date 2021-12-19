import os
import wandb
import argparse

import torch
import torch.nn as nn
import albumentations as A

from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from src.pascal_voc import PascalVOCDataset
from src.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss
from src.losses.constants import MULTICLASS_MODE
from src.metrics import IoU, Accuracy
from src.engine import train


def get_args_parser():
    parser = argparse.ArgumentParser('CIFAR10 Gradual Pruning', add_help=False)
    # Logging
    parser.add_argument('--wandb_logger', action='store_true')
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, help='path to image dir')
    # Path to checkpoint to start from
    parser.add_argument('--from_checkpoint', type=str, default='', help='path to checkpoint to start from')
    parser.add_argument('--pretrain_n_classes', type=int, default=40, help='number of classes in pretrained model')
    # Dataloader parameters
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # Loss
    parser.add_argument('--loss', default='dice', type=str)
    # Image params
    parser.add_argument('--image_size', default=128, type=int)
    # Optimizer params
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    # Scheduler params
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--milestones', nargs='+', default=[10], type=int)
    # Save arguments
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='dir with results')
    return parser


LOSS_FUNCTIONS = {
    "dice" : DiceLoss(mode=MULTICLASS_MODE, from_logits=True),
    "ce" : nn.CrossEntropyLoss(),
    "soft_ce" : SoftCrossEntropyLoss(smooth_factor=0.1),
    "focal" : FocalLoss(mode=MULTICLASS_MODE, gamma=2)
}


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.wandb_logger:
        wandb.init(
            project="Unsupervised Semantic Segmentation",
            entity="spiridon_sun_rotator",
            name="Pascal VOC training",
            config={
                "model" : "MobileNetV3_small",
                "dataset" : "Pascal VOC"
            }
        )

    # get dataset
    PASCAL_VOC_MEAN, PASCAL_VOC_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_transforms = A.Compose([
        A.Resize(height=args.image_size, width=args.image_size),
        A.OneOf([
            A.GaussianBlur(p=0.5),
            A.GaussNoise(),
        ], p=0.5),
        A.HueSaturationValue(),
        A.CLAHE(),
        A.ColorJitter(),
        A.Normalize(mean=PASCAL_VOC_MEAN, std=PASCAL_VOC_STD),
        ToTensorV2()
    ])

    test_transforms = A.Compose([
        A.Resize(height=args.image_size, width=args.image_size),
        A.Normalize(mean=PASCAL_VOC_MEAN, std=PASCAL_VOC_STD),
        ToTensorV2()
    ])

    train_dataset = PascalVOCDataset(root=args.data_dir, year='2012', image_set='train', transform=train_transforms)
    test_dataset = PascalVOCDataset(root=args.data_dir, year='2012', image_set='val', transform=test_transforms)
    # get train and test loader
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False)

    # get model
    model = deeplabv3_mobilenet_v3_large(
        pretrained=False, 
        pretrained_backbone=True,
        num_classes=args.pretrain_n_classes if args.from_checkpoint else 21
    )
    # load checkpoint if given
    if args.from_checkpoint:
        model.load_state_dict(torch.load(args.from_checkpoint)['model_state_dict'])
        # reset the classfier head to 21 classes
        classifier_in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(classifier_in_ch, 21, kernel_size=1)
    # load to device
    model = model.to(device)

    # create directory for checkpoints (if needed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # get criterion
    criterion = LOSS_FUNCTIONS[args.loss]
    # get metrics
    # metrics = {'IoU' : IoU(threshold=None)}
    metrics = {'Accuracy' : Accuracy(threshold=None)}
    # get optimizer
    optimizer =  torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr' : (args.lr / 10)},
        {'params': model.classifier.parameters(), 'lr': args.lr},   
    ])
    # get scheduler
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    else:
        scheduler = None
    # train the model
    train(
        model,
        {"train" :train_loader, "val" : test_loader},
        criterion,
        optimizer,
        metrics=metrics,
        target_metric="Accuracy",
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        device=device,
        save_checkpoint_dir=args.checkpoint_dir,
        wandb_logger=args.wandb_logger
    )
