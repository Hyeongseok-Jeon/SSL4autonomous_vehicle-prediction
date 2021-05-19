import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from importlib import import_module
from LaneGCN.utils import gpu, to_long, Optimizer, StepLR
import os


def get_model(args):
    downstream = import_module('SSL_downstream')

    config, config_enc, Dataset, collate_fn, model, loss, _, post_process = downstream.get_model(args)
    model = model.cuda()

    params = model.parameters()
    opt = Optimizer(params, config)

    return config, config_enc, Dataset, collate_fn, model, loss, opt, post_process
