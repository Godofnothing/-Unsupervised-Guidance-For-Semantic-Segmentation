import torch.nn as nn


class MultiViewLoss(nn.Module):
    
    def __init__(self, consistency_loss : nn.Module, mask_loss : nn.Module, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_loss = consistency_loss 
        self.mask_loss = mask_loss
        self.consistency_weight = consistency_weight
        
    def forward(self, mask_logits_1, mask_logits_2, gt_masks):
        mask_loss = self.mask_loss(mask_logits_1, gt_masks) + self.mask_loss(mask_logits_2, gt_masks)
        consistency_loss = self.consistency_loss(mask_logits_1, mask_logits_2)
        return mask_loss + self.consistency_weight * consistency_loss