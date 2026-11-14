# Adapted for sign representation training and evaluation.
import numbers

from ignite.metrics import Average
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
import torch

try:
    from ignite.handlers.checkpoint import BaseSaveHandler
except ImportError:
    BaseSaveHandler = None

from train_utils.seed_helpers import setup_seed
from augmentation.get_aug import get_aug
from dataloaders.get_dataset import get_dataset
from losses.get_loss import get_loss
from optimizer.get_optimizer import get_optim


class DisplayScoreCheckpoint(Checkpoint):
    def __init__(self, *args, display_score_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_score_function = display_score_function

    def _should_store(self, priority):
        if self._check_lt_n_saved():
            return True
        if hasattr(self, "_compare_fn"):
            return self._compare_fn(priority)
        return self._saved[0].priority < priority

    def _get_filename_pattern_compat(self, global_step):
        if hasattr(self, "_get_filename_pattern"):
            return self._get_filename_pattern(global_step)
        if self.filename_pattern is not None:
            return self.filename_pattern
        return self.setup_filename_pattern(
            with_prefix=len(self.filename_prefix) > 0,
            with_score=self.score_function is not None,
            with_score_name=self.score_name is not None,
            with_global_step=global_step is not None,
        )

    def _format_score(self, score):
        if isinstance(score, numbers.Integral):
            return f"{score}"
        return f"{score:.4f}"

    def __call__(self, engine):
        global_step = None
        if self.global_step_transform is not None:
            global_step = self.global_step_transform(engine, engine.last_event_name)

        if self.score_function is not None:
            priority = self.score_function(engine)
            if not isinstance(priority, numbers.Number):
                raise ValueError("Output of score_function should be a number")
        else:
            if global_step is None:
                global_step = engine.state.get_event_attrib_value(
                    Events.ITERATION_COMPLETED
                )
            priority = global_step

        if self._should_store(priority):
            display_score = priority
            if self.score_function is not None and self.display_score_function is not None:
                display_score = self.display_score_function(engine)
                if not isinstance(display_score, numbers.Number):
                    raise ValueError(
                        "Output of display_score_function should be a number"
                    )

            checkpoint = self._setup_checkpoint()

            name = "checkpoint"
            if len(checkpoint) == 1:
                for k in checkpoint:
                    name = k
                checkpoint = checkpoint[name]

            filename_pattern = self._get_filename_pattern_compat(global_step)
            filename_dict = {
                "filename_prefix": self.filename_prefix,
                "ext": self.ext,
                "name": name,
                "score_name": self.score_name,
                "score": (
                    self._format_score(display_score)
                    if self.score_function is not None
                    else None
                ),
                "global_step": global_step,
            }
            filename = filename_pattern.format(**filename_dict)
            if any(item.filename == filename for item in self._saved):
                return

            metadata = {
                "basename": f"{self.filename_prefix}{'_' * int(len(self.filename_prefix) > 0)}{name}",
                "score_name": self.score_name,
                "priority": priority,
            }
            if not self._check_lt_n_saved():
                item = self._saved.pop(0)
                if BaseSaveHandler is not None and isinstance(self.save_handler, BaseSaveHandler):
                    self.save_handler.remove(item.filename)
                elif hasattr(self.save_handler, "remove"):
                    self.save_handler.remove(item.filename)

            self._saved.append(Checkpoint.Item(priority, filename))
            self._saved.sort(key=lambda it: it[0])
            if self.include_self:
                checkpoint["checkpointer"] = self.state_dict()
            try:
                self.save_handler(checkpoint, filename, metadata)
            except TypeError:
                self.save_handler(checkpoint, filename)


class BaseTrainer:
    def __init__(self, args):
        cfg = args[0]
        self.cfg = cfg
        setup_seed(cfg.seed)

        self.support_bfloat = torch.cuda.is_bf16_supported()
        self.dtype = torch.bfloat16 if self.support_bfloat else torch.float16
        self.accum = max(int(cfg.accum), 1) if "accum" in cfg else 1

    def get_gradient_accumulation_state(self, engine):
        epoch_length = engine.state.epoch_length or 1
        iteration = max(int(engine.state.iteration), 1)
        batch_in_epoch = ((iteration - 1) % epoch_length) + 1
        group_start = ((batch_in_epoch - 1) // self.accum) * self.accum + 1
        group_end = min(group_start + self.accum - 1, epoch_length)
        accum_steps = group_end - group_start + 1

        should_zero_grad = (batch_in_epoch - 1) % self.accum == 0
        should_step = batch_in_epoch % self.accum == 0 or batch_in_epoch == epoch_length

        return should_zero_grad, should_step, accum_steps

    def get_dataloaders(self, cfg):
        transform = get_aug(cfg.aug_name, cfg.aug_params)
        self.transform = transform
        if "run" in cfg and cfg.get("evaluation_split"):
            split = cfg.evaluation_split
            if split not in {"dev", "test"}:
                raise ValueError(f"Unsupported evaluation split: {split}")
            params = cfg.valid_ds_params if split == "dev" else cfg.test_ds_params
            dataset_name = cfg.valid_ds_name if split == "dev" else cfg.test_ds_name
            loader, metadata = get_dataset(dataset_name, params, transform=transform)
            return loader, loader, loader, metadata, metadata, metadata
        train_dl, train_dict = get_dataset(
            cfg.train_ds_name, cfg.train_ds_params, transform=transform
        )
        valid_dl, valid_dict = get_dataset(
            cfg.valid_ds_name, cfg.valid_ds_params, transform=transform
        )
        test_dl, test_dict = get_dataset(
            cfg.test_ds_name, cfg.test_ds_params, transform=transform
        )
        return train_dl, valid_dl, test_dl, train_dict, valid_dict, test_dict

    def loss_fn(self, y_pred, target):
        loss, dict_losses = self.criterion(y_pred, target)
        return loss, dict_losses

    def init_criterion(self, cfg):
        criterion = get_loss(cfg.criterion_name, cfg.criterion_params)
        self.criterion = criterion.to(idist.device())

    def init_optimizer(self, cfg):
        self.optimizer = get_optim(cfg.optimizer_name, cfg.optimizer_params, self.model)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("MODEL:", count_parameters(self.model))

        if not self.support_bfloat:
            self.scaler = torch.cuda.amp.GradScaler()  # growth_interval=500)
        else:
            self.scaler = None
        self.grad_clip_norm = cfg.grad_clip_norm if ("grad_clip_norm" in cfg) else None
        self.grad_clip_value = (
            cfg.grad_clip_value if ("grad_clip_value" in cfg) else None
        )

    def get_only_engines(self):
        trainer = Engine(self.train_step)
        evaluator = Engine(self.eval_step)
        valid_tester = Engine(self.test_step)
        test_tester = Engine(self.test_step)
        return trainer, evaluator, valid_tester, test_tester

    def get_only_dataloaders(self):
        (
            train_dl,
            valid_dl,
            test_dl,
            train_dict,
            valid_dict,
            test_dict,
        ) = self.get_dataloaders(self.cfg)
        return train_dl, valid_dl, test_dl

    def dict_metric_from_list(
        self, engine_type, list_of_metrics, dict_metrics, **kwargs
    ):
        return dict_metrics

    def init_metrics(self, engine, engine_type, dict_metrics={}, additional=True):
        if additional:

            def loss_display(k):
                def loss_fn(x):
                    return x["losses"][k]

                return loss_fn

            for k in ["loss"] + [f"loss_{comp}" for comp in self.criterion.crit_keys]:
                dict_metrics[f"{engine_type}/avg_{k}"] = Average(
                    output_transform=loss_display(k)
                )

        for k, v in dict_metrics.items():
            v.attach(engine, k)

    def prep_scheduler(self, cfg, train_dl, optimizer, trainer, evaluator):
        from scheduler.get_scheduler import prep_scheduler

        scheduler, engine_name, event_name = prep_scheduler(
            cfg.lr_scheduler, cfg, train_dl, optimizer, trainer, evaluator
        )

        if scheduler:
            def add_handler(engine, scheduler_handler):
                if (
                    engine is trainer
                    and event_name == Events.ITERATION_STARTED
                    and self.accum > 1
                ):
                    def run_on_accum_start(active_engine):
                        should_zero_grad, _, _ = self.get_gradient_accumulation_state(
                            active_engine
                        )
                        if should_zero_grad:
                            scheduler_handler(active_engine)

                    engine.add_event_handler(event_name, run_on_accum_start)
                else:
                    engine.add_event_handler(event_name, scheduler_handler)

            if isinstance(scheduler, list):
                for sch in scheduler:
                    if engine_name == "trainer":
                        add_handler(trainer, sch)
                    elif engine_name == "evaluator":
                        add_handler(evaluator, sch)
            else:
                if engine_name == "trainer":
                    add_handler(trainer, scheduler)
                elif engine_name == "evaluator":
                    add_handler(evaluator, scheduler)
        return scheduler

    def get_checkpoint_monitor_settings(self, cfg):
        metric_name = (
            cfg.checkpoint_metric if "checkpoint_metric" in cfg else cfg.score_name
        )
        metric_key = (
            cfg.checkpoint_metric_key if "checkpoint_metric_key" in cfg else None
        )
        explicit_monitor = any(
            key in cfg
            for key in ("checkpoint_metric", "checkpoint_metric_key", "checkpoint_mode")
        )

        if "checkpoint_mode" in cfg:
            mode = str(cfg.checkpoint_mode).lower()
        else:
            score_factor = cfg.score_factor if "score_factor" in cfg else 1
            mode = "min" if score_factor < 0 else "max"

        if mode not in {"min", "max"}:
            raise ValueError(
                f"Unsupported checkpoint mode '{mode}'. Use 'min' or 'max'."
            )

        if metric_key == "":
            metric_key = None

        return metric_name, metric_key, mode, explicit_monitor

    def get_checkpoint_monitors(self, cfg):
        if "checkpoint_monitors" in cfg:
            monitors = []
            seen_prefixes = set()
            for filename_prefix, raw_monitor in cfg.checkpoint_monitors.items():
                monitor = (
                    raw_monitor.to_dict()
                    if hasattr(raw_monitor, "to_dict")
                    else dict(raw_monitor)
                )
                metric_name = (
                    monitor["metric"] if "metric" in monitor else cfg.score_name
                )
                metric_key = monitor["metric_key"] if "metric_key" in monitor else None
                mode = str(monitor["mode"] if "mode" in monitor else "max").lower()

                if not filename_prefix:
                    raise ValueError("Checkpoint monitor names must be non-empty.")
                if filename_prefix in seen_prefixes:
                    raise ValueError(
                        f"Duplicate checkpoint monitor name '{filename_prefix}'."
                    )
                if mode not in {"min", "max"}:
                    raise ValueError(
                        f"Unsupported checkpoint mode '{mode}' for "
                        f"checkpoint monitor '{filename_prefix}'."
                    )
                if metric_key == "":
                    metric_key = None

                seen_prefixes.add(filename_prefix)
                monitors.append(
                    {
                        "filename_prefix": filename_prefix,
                        "metric_name": metric_name,
                        "metric_key": metric_key,
                        "mode": mode,
                        "use_legacy_score_factor": False,
                    }
                )

            if len(monitors) == 0:
                raise ValueError("cfg.checkpoint_monitors must define at least one entry.")

            return monitors

        metric_name, metric_key, mode, explicit_monitor = (
            self.get_checkpoint_monitor_settings(cfg)
        )
        return [
            {
                "filename_prefix": "best_result",
                "metric_name": metric_name,
                "metric_key": metric_key,
                "mode": mode,
                "use_legacy_score_factor": not explicit_monitor,
                "score_factor": cfg.score_factor if "score_factor" in cfg else 1,
            }
        ]

    def get_primary_checkpoint_prefix(self, cfg):
        monitors = self.get_checkpoint_monitors(cfg)
        prefixes = [monitor["filename_prefix"] for monitor in monitors]
        return "best_result" if "best_result" in prefixes else prefixes[0]

    def resolve_best_checkpoint_prefix(self, cfg, load_from_ckpt):
        if load_from_ckpt == "best":
            return self.get_primary_checkpoint_prefix(cfg)

        monitor_prefixes = {
            monitor["filename_prefix"] for monitor in self.get_checkpoint_monitors(cfg)
        }
        if load_from_ckpt in monitor_prefixes:
            return load_from_ckpt
        return None

    def get_checkpoint_metric_value(self, engine, metric_name, metric_key=None):

        if metric_name not in engine.state.metrics:
            raise KeyError(
                f"Checkpoint metric '{metric_name}' was not found. "
                f"Available metrics: {sorted(engine.state.metrics.keys())}"
            )

        metric_value = engine.state.metrics[metric_name]

        if metric_key is not None:
            if not isinstance(metric_value, dict):
                raise TypeError(
                    f"Checkpoint metric '{metric_name}' is not a dict, so "
                    f"checkpoint_metric_key='{metric_key}' cannot be used."
                )
            if metric_key not in metric_value:
                raise KeyError(
                    f"Checkpoint metric key '{metric_key}' was not found in "
                    f"'{metric_name}'. Available keys: {sorted(metric_value.keys())}"
                )
            metric_value = metric_value[metric_key]
        elif isinstance(metric_value, dict):
            if "bleu4" in metric_value:
                metric_value = metric_value["bleu4"]
            else:
                raise TypeError(
                    f"Checkpoint metric '{metric_name}' returned a dict. "
                    "Set cfg.checkpoint_metric_key to select a scalar value."
                )

        return float(metric_value)

    def build_checkpoint_score_function(self, monitor):
        metric_name = monitor["metric_name"]
        metric_key = monitor["metric_key"]
        mode = monitor["mode"]
        use_legacy_score_factor = monitor.get("use_legacy_score_factor", False)
        score_factor = monitor.get("score_factor", 1)

        def score_function(engine):
            metric_value = self.get_checkpoint_metric_value(
                engine, metric_name, metric_key
            )
            if use_legacy_score_factor:
                return score_factor * metric_value

            return metric_value if mode == "max" else -metric_value

        return score_function

    def build_checkpoint_display_score_function(self, monitor):
        metric_name = monitor["metric_name"]
        metric_key = monitor["metric_key"]

        def display_score_function(engine):
            return self.get_checkpoint_metric_value(engine, metric_name, metric_key)

        return display_score_function

    def save_checkpoints(self, cfg, trainer, evaluator, best_only=False, to_save={}):
        if not cfg.save_ckpt:
            return
        if to_save == {}:
            to_save = {
                "trainer": trainer,
                "optimizer": self.optimizer,
                "model": self.model,
            }
        if self.scaler is not None:
            to_save["scaler"] = self.scaler

        if self.scheduler is not None:
            to_save["scheduler"] = self.scheduler

        checkpoint_monitors = self.get_checkpoint_monitors(cfg)
        best_checkpoints = []
        for monitor in checkpoint_monitors:
            best_checkpoints.append(
                DisplayScoreCheckpoint(
                    to_save,
                    DiskSaver(dirname=cfg.save_dir, require_empty=False),
                    n_saved=1,
                    filename_prefix=monitor["filename_prefix"],
                    score_function=self.build_checkpoint_score_function(monitor),
                    display_score_function=self.build_checkpoint_display_score_function(
                        monitor
                    ),
                    score_name=None,
                    global_step_transform=global_step_from_engine(trainer),
                    greater_or_equal=True,
                )
            )

        latest_checkpoint = Checkpoint(
            to_save,
            DiskSaver(dirname=cfg.save_dir, require_empty=False),
            n_saved=1,
            filename_prefix=f"latest_epoch",
        )

        def before_latest_checkpoint(engine):
            engine.state.output = None
            engine.state.batch = None

        if cfg.save_ckpt:
            for checkpoint in best_checkpoints:
                evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpoint)
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=1), before_latest_checkpoint
            )
            if not best_only:
                trainer.add_event_handler(
                    Events.EPOCH_COMPLETED(every=1), latest_checkpoint
                )

    def load_checkpoints(self, cfg, trainer, objects_to_load={}):
        if objects_to_load == {}:
            objects_to_load = {
                "model": self.model,
                "optimizer": self.optimizer,
                "trainer": trainer,
            }
            if self.scaler is not None:
                objects_to_load["scaler"] = self.scaler

        if cfg.model_checkpoint_dir != "":
            print("LOADING MODEL FROM:", cfg.model_checkpoint_dir)
            objects_to_load = {"model": self.model}
            Checkpoint.load_objects(
                to_load=objects_to_load,
                # checkpoint=cfg.model_checkpoint_dir,
                checkpoint=torch.load(
                    cfg.model_checkpoint_dir, map_location="cpu"
                ),
                strict=True,
            )

        if cfg.resume:
            from train_utils.checkpoint_helpers import (
                get_latest_saved_file,
                get_best_checkpoint_details,
            )

            latest_checkpoint_from_resume = get_latest_saved_file(
                cfg.save_dir, extension="pt", name_latest="latest_epoch"
            )
            if latest_checkpoint_from_resume[0]:
                if self.scheduler:
                    objects_to_load["scheduler"] = self.scheduler

                Checkpoint.load_objects(
                    to_load=objects_to_load,
                    checkpoint=torch.load(
                        latest_checkpoint_from_resume[0], map_location="cpu"
                    ),
                )

    def cleaning_with_progress(self, trainer, evaluator, cfg, train_dl):
        if cfg.train_length is not None:
            tstep = cfg.train_length // 10
        else:
            tstep = len(train_dl) // 10
        from datetime import datetime

        def progress_train(engine):
            if idist.get_rank() == 0:
                compl = "{0:.2f}".format(
                    (engine.state.iteration % engine.state.epoch_length)
                    / engine.state.epoch_length
                )
                n = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{n}] Epoch {engine.state.epoch}: ({compl}) - {(engine.state.iteration % engine.state.epoch_length)}/{engine.state.epoch_length}"
                )

        # self.logger.apply_gradient_log(trainer, self.model, tstep, self.scaler)

        if cfg.train_length is not None:
            tstep = max(cfg.train_length // 5, 50)
        else:
            tstep = max(len(train_dl) // 5, 50)

        if "pbar" in cfg.logger_name:
            if idist.get_rank() == 0:
                pbar = ProgressBar()
                pbar.attach(evaluator)

            if idist.get_rank() == 0:
                pbar = ProgressBar()
                pbar.attach(trainer)
        else:
            trainer.add_event_handler(
                Events.ITERATION_COMPLETED(every=tstep), progress_train
            )
        # -------------------------------------- [END] LOGGING HANDLER --------------------------------------
        from ignite.contrib.engines.common import empty_cuda_cache

        def clear_states_engines(engine):
            trainer.state.output = None
            trainer.state.batch = None
            engine.state.output = None
            engine.state.batch = None

        trainer.add_event_handler(Events.EPOCH_STARTED, clear_states_engines)
        evaluator.add_event_handler(Events.EPOCH_STARTED, clear_states_engines)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

        if idist.get_world_size() > 1:

            def set_epoch(engine):
                train_dl.sampler.set_epoch(engine.state.epoch)

            trainer.add_event_handler(Events.EPOCH_STARTED, set_epoch)
