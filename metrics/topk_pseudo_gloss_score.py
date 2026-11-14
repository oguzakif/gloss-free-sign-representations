from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch


def _valid_unique_targets(target_ids, num_classes, device):
    if not torch.is_tensor(target_ids):
        target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
    else:
        target_ids = target_ids.to(device=device, dtype=torch.long)
    target_ids = target_ids[(target_ids >= 0) & (target_ids < num_classes)]
    return torch.unique(target_ids)


class RecallAtK(Metric):
    def __init__(self, k, output_transform=lambda x: x):
        self.k = int(k)
        self.hit_total = None
        self.target_total = None
        super(RecallAtK, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.hit_total = torch.zeros(1)
        self.target_total = torch.zeros(1)
        super(RecallAtK, self).reset()

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output
        preds = preds.float().detach()
        num_classes = preds.shape[-1]
        k = min(self.k, num_classes)
        if k <= 0:
            return

        topk_ids = torch.topk(preds, k=k, dim=-1).indices
        hit_total = preds.new_zeros(())
        target_total = preds.new_zeros(())

        for i, target_ids in enumerate(targets):
            target_ids = _valid_unique_targets(target_ids, num_classes, preds.device)
            if target_ids.numel() == 0:
                continue
            hits = (topk_ids[i].unsqueeze(1) == target_ids.unsqueeze(0)).any(dim=0)
            hit_total += hits.float().sum()
            target_total += target_ids.numel()

        self.hit_total += hit_total.detach().cpu()
        self.target_total += target_total.detach().cpu()

    @sync_all_reduce("hit_total", "target_total")
    def compute(self):
        return (self.hit_total / (self.target_total + 1e-8)).item()


class PrecisionAtK(Metric):
    def __init__(self, k, output_transform=lambda x: x):
        self.k = int(k)
        self.hit_total = None
        self.pred_total = None
        super(PrecisionAtK, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.hit_total = torch.zeros(1)
        self.pred_total = torch.zeros(1)
        super(PrecisionAtK, self).reset()

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output
        preds = preds.float().detach()
        num_classes = preds.shape[-1]
        k = min(self.k, num_classes)
        if k <= 0:
            return

        topk_ids = torch.topk(preds, k=k, dim=-1).indices
        hit_total = preds.new_zeros(())
        pred_total = preds.new_zeros(())

        for i, target_ids in enumerate(targets):
            target_ids = _valid_unique_targets(target_ids, num_classes, preds.device)
            if target_ids.numel() == 0:
                continue
            hits = (topk_ids[i].unsqueeze(1) == target_ids.unsqueeze(0)).any(dim=1)
            hit_total += hits.float().sum()
            pred_total += k

        self.hit_total += hit_total.detach().cpu()
        self.pred_total += pred_total.detach().cpu()

    @sync_all_reduce("hit_total", "pred_total")
    def compute(self):
        return (self.hit_total / (self.pred_total + 1e-8)).item()


class MeanPositiveRank(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.rank_total = None
        self.target_total = None
        super(MeanPositiveRank, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.rank_total = torch.zeros(1)
        self.target_total = torch.zeros(1)
        super(MeanPositiveRank, self).reset()

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output
        preds = preds.float().detach()
        num_classes = preds.shape[-1]
        rank_total = preds.new_zeros(())
        target_total = preds.new_zeros(())

        rank_values = torch.arange(1, num_classes + 1, device=preds.device)
        for i, target_ids in enumerate(targets):
            target_ids = _valid_unique_targets(target_ids, num_classes, preds.device)
            if target_ids.numel() == 0:
                continue
            sorted_ids = torch.argsort(preds[i], descending=True)
            ranks = torch.empty_like(sorted_ids)
            ranks[sorted_ids] = rank_values
            rank_total += ranks[target_ids].float().sum()
            target_total += target_ids.numel()

        self.rank_total += rank_total.detach().cpu()
        self.target_total += target_total.detach().cpu()

    @sync_all_reduce("rank_total", "target_total")
    def compute(self):
        return (self.rank_total / (self.target_total + 1e-8)).item()
