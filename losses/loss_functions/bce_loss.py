import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn


class Loss(nn.Module):
    log_keys = ["pos", "neg", "target_density"]

    def __init__(
        self,
        src_name,
        tgt_name,
        balance_positive_negative=False,
        positive_weight=1.0,
        negative_weight=1.0,
    ):
        super().__init__()

        self.src_name = src_name
        self.tgt_name = tgt_name
        self.balance_positive_negative = balance_positive_negative
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.bce_loss_fn = torch.nn.BCELoss(reduction="none")

    def forward(self, y_pred, target):
        if isinstance(self.src_name, tuple):
            src = y_pred
            for mn in self.src_name:
                src = src[mn]
        else:
            src = y_pred[self.src_name]
        tgt = target[self.tgt_name]

        gloss_targets = torch.zeros(
            src.shape[0], src.shape[-1], dtype=src.dtype, device=src.device
        )
        for i, t in enumerate(tgt):
            for t_i in t:
                gloss_targets[i, t_i] = 1.0

        loss = self.bce_loss_fn(torch.clamp(src, 1e-8, 1 - 1e-8), gloss_targets)
        pos_mask = gloss_targets.bool()
        neg_mask = ~pos_mask
        zero = loss.new_zeros(())
        pos_loss = loss[pos_mask].mean() if pos_mask.any() else zero
        neg_loss = loss[neg_mask].mean() if neg_mask.any() else zero
        target_density = pos_mask.float().mean()

        if self.balance_positive_negative:
            total_loss = (
                self.positive_weight * pos_loss
                + self.negative_weight * neg_loss
            )
        else:
            total_loss = loss.mean()

        return total_loss, {
            "pos": pos_loss,
            "neg": neg_loss,
            "target_density": target_density,
        }


if __name__ == "__main__":
    loss_params = {}
    loss_fn = Loss(**loss_params)

    dict_preds = {
        "target": {},
        "y_pred": {},
    }

    loss = loss_fn(**dict_preds)
