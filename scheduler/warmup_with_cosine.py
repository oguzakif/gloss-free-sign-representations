from ignite.contrib.handlers import (
    LinearCyclicalScheduler,
    CosineAnnealingScheduler,
    ConcatScheduler,
)
import math
import copy


class WarmupCosineHoldScheduler:
    def __init__(
        self,
        optimizer,
        *,
        warmup_steps,
        decay_steps,
        start_factor,
        min_factor,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(int(warmup_steps), 1)
        self.decay_steps = max(int(decay_steps), self.warmup_steps + 1)
        self.start_factor = float(start_factor)
        self.min_factor = float(min_factor)
        self.event_index = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def _factor(self):
        if self.event_index <= self.warmup_steps:
            progress = self.event_index / self.warmup_steps
            return self.start_factor + (1.0 - self.start_factor) * progress

        if self.event_index >= self.decay_steps:
            return self.min_factor

        progress = (self.event_index - self.warmup_steps) / (
            self.decay_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine

    def __call__(self, engine):
        factor = self._factor()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor
        self.event_index += 1

    def state_dict(self):
        return {
            "event_index": self.event_index,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.event_index = state_dict.get("event_index", self.event_index)
        self.base_lrs = state_dict.get("base_lrs", self.base_lrs)


def get_warmup_with_cosine(args, optimizer, dl):
    lr = args.lr
    if "lr_scale_factor" in args.lr_scheduler_params:
        lr_scale_factor = args.lr_scheduler_params["lr_scale_factor"]
        start_value = lr * lr_scale_factor
    else:
        start_value = args.lr_scheduler_params["start_lr"]
    num_cycles = args.lr_scheduler_params["num_cycles"]
    start_value_mult = args.lr_scheduler_params["start_value_mult"]
    end_value_mult = args.lr_scheduler_params["end_value_mult"]
    warmup_epochs = args.lr_scheduler_params["warmup_epochs"]

    if args.train_length:
        micro_epoch_length = args.train_length
    else:
        micro_epoch_length = len(dl)
    accum = max(int(args.accum), 1) if "accum" in args else 1
    epoch_length = math.ceil(micro_epoch_length / accum)
    num_epochs = args.max_epochs
    if "decay_epochs" in args.lr_scheduler_params:
        decay_epochs = min(int(args.lr_scheduler_params["decay_epochs"]), num_epochs)
        return WarmupCosineHoldScheduler(
            optimizer,
            warmup_steps=epoch_length * warmup_epochs,
            decay_steps=epoch_length * decay_epochs,
            start_factor=start_value / lr,
            min_factor=start_value / lr,
        )

    scheduler_1 = LinearCyclicalScheduler(
        optimizer,
        "lr",
        start_value=start_value,
        end_value=lr,
        cycle_size=epoch_length * warmup_epochs * 2,
    )
    scheduler_2 = CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value=lr,
        end_value=start_value,
        cycle_size=math.ceil(
            ((num_epochs - 1 * warmup_epochs) * epoch_length) / num_cycles
        ),
        start_value_mult=start_value_mult,
        end_value_mult=end_value_mult,
    )
    durations = [
        epoch_length * warmup_epochs,
    ]
    scheduler = ConcatScheduler(
        schedulers=[scheduler_1, scheduler_2], durations=durations
    )
    return scheduler
