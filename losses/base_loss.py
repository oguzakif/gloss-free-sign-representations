import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


class Loss(nn.Module):
    def __init__(self, dict_of_loss_params={}):
        super().__init__()
        self.dict_loss_functions = {}
        self.weighting = {}
        dict_mods = {}
        for key, dict_value in dict_of_loss_params.items():

            dict_mods[key] = importlib.import_module(dict_value["cls_name"], package=None)

            self.dict_loss_functions[key] = dict_mods[key].Loss(**dict_value["loss_params"])

            self.weighting[key] = dict_value["weight"]

        self.crit_keys = list(dict_of_loss_params.keys())
        for key, loss_fn in self.dict_loss_functions.items():
            for log_key in getattr(loss_fn, "log_keys", []):
                self.crit_keys.append(f"{key}_{log_key}")

    def forward(self, y_pred, target):
        loss = 0
        dict_loss = {}
        for key, loss_fn in self.dict_loss_functions.items():
            loss_k = loss_fn(y_pred, target)
            extra_losses = {}
            if isinstance(loss_k, tuple):
                loss_k, extra_losses = loss_k
            dict_loss[f"loss_{key}"]=loss_k
            for extra_key, extra_value in extra_losses.items():
                dict_loss[f"loss_{key}_{extra_key}"] = extra_value
            loss+=self.weighting[key]*loss_k

        return loss, dict_loss
