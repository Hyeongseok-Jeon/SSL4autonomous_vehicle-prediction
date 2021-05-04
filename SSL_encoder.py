import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from importlib import import_module
from LaneGCN.utils import gpu, to_long, Optimizer, StepLR
import numpy as np
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import NTXentLoss
import copy
import os

file_path = os.path.abspath(__file__)
# file_path = os.getcwd() + '/LaneGCN/lanegcn.py'
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

### config ###
config_enc = dict()
config_action_emb = dict()
"""Train"""
config_action_emb["output_size"] = 128
config_action_emb["num_channels"] = [128, 128, 128]
config_action_emb["kernel_size"] = 5
config_action_emb["dropout"] = 0.2
config_action_emb["n_hid"] = 128
config_enc['action_emb'] = config_action_emb
config_enc['auxiliary'] = True
config_enc['pre_trained'] = True
config_enc['pre_trained_weight'] = os.path.join(root_path, 'results', 'SSL_encoder', '0.000.ckpt')

if "save_dir" not in config_enc:
    config_enc["save_dir"] = os.path.join(
        root_path, "results", model_name
    )


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output).double()
        return self.sig(output)


class SSL_encoder(nn.Module):
    def __init__(self, config, base_model):
        super(SSL_encoder, self).__init__()
        self.relu = nn.ReLU()
        self.base_net = base_model.Net(config)
        self.action_emb = TCN(input_size=102,
                              output_size=config_action_emb["output_size"],
                              num_channels=config_action_emb["num_channels"],
                              kernel_size=config_action_emb["kernel_size"],
                              dropout=config_action_emb["dropout"]).cuda()
        self.out = nn.Linear(config_action_emb["output_size"] * 2, config_action_emb["n_hid"]).double()
        self.auxiliary = nn.Linear(config_action_emb["n_hid"], config_action_emb["n_hid"]).double()

    def forward(self, data):
        actors, veh_in_batch = self.base_net(data)
        batch_num = len(veh_in_batch)
        veh_num_in_batch = sum(veh_in_batch)
        ego_idx = [0] + [sum(veh_in_batch[:i + 1]) for i in range(batch_num - 1)]
        target_idx = [1] + [sum(veh_in_batch[:i + 1]) + 1 for i in range(batch_num - 1)]

        positive_idx = [np.random.randint(1, data['action'][i].shape[1]) for i in range(batch_num)]
        action_original = torch.cat([gpu(data['action'][i][0:1, 0, :, :]) for i in range(batch_num)])
        action_augmented = torch.cat([gpu(data['action'][i][0:1, positive_idx[i], :, :]) for i in range(batch_num)])

        actions = torch.cat([action_original, action_augmented])
        hid_act = self.action_emb(actions)[:, -1, :]
        hid_act_original = hid_act[:int(hid_act.shape[0] / 2)]
        hid_act_augmented = hid_act[int(hid_act.shape[0] / 2):]
        idx_mask = torch.arange(0, hid_act_original.shape[0])

        sample_original = torch.cat([hid_act_original, actors[target_idx]], dim=1)
        sample_augmented = torch.cat([hid_act_augmented, actors[target_idx]], dim=1)

        positive_samples = sample_augmented
        anchor_sample = sample_original

        samples = torch.cat([positive_samples, anchor_sample])
        hid_tmp = self.relu(self.out(samples))
        hid_positive = torch.cat([hid_tmp[i].unsqueeze(0) for i in range(batch_num)])
        hid_anchor = torch.cat([hid_tmp[i + batch_num].unsqueeze(0) for i in range(batch_num)])
        if config_enc['auxiliary']:
            hid = [hid_positive, hid_anchor]
            hid_aux = self.auxiliary(hid_tmp)
            hid_positive = torch.cat([hid_aux[i].unsqueeze(0) for i in range(batch_num)])
            hid_anchor = torch.cat([hid_aux[i + batch_num].unsqueeze(0) for i in range(batch_num)])
            hid_aux = [hid_positive, hid_anchor]
            return [hid, hid_aux]

        hid = [hid_positive, hid_anchor]
        return hid


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config

    def forward(self, hid):
        if isinstance(hid[0], list):
            hid = hid[1]
        batch_num = hid[0].shape[0]
        hid_positive = hid[0]
        hid_anchor = hid[1]

        samples = torch.zeros_like(torch.cat([hid_anchor, hid_positive]))
        anc_idx = torch.arange(batch_num) * 2
        pos_idx = torch.arange(batch_num) * 2 + 1
        samples[anc_idx] = hid_anchor
        samples[pos_idx] = hid_positive
        labels = torch.arange(2 * batch_num)
        labels[anc_idx] = labels[pos_idx]

        infoNCE_loss = infoNCELoss(samples, labels)

        return infoNCE_loss


def infoNCELoss(samples, labels):
    batch_num = int(len(labels) / 2)
    label_uni = torch.unique(labels)
    loss_tot = 0
    for i in range(batch_num):
        label = label_uni[i]
        pos_pair = samples[labels == label]
        neg_pairs = torch.cat([pos_pair[0:1], samples[labels != label]])

        num = consine_similarity(pos_pair)
        den = consine_similarity(neg_pairs) + num
        loss = num / den
        loss_tot = loss_tot + loss

    return -torch.log(loss_tot / batch_num)


def consine_similarity(pair):
    num = torch.sum(pair[0:1] * pair[1:], dim=1)
    den = torch.norm(pair[0:1]) * torch.norm(pair[1:], dim=1)
    sim = num / den
    sim = torch.clamp(sim, -1, 1)

    return torch.sum(1 - (torch.arccos(sim) / np.pi))


def get_model(base_model_name):
    base_model = import_module(base_model_name + '_backbone')
    config = base_model.config
    Dataset = base_model.ArgoDataset
    collate_fn = base_model.collate_fn

    encoder = SSL_encoder(config, base_model)
    if config_enc['pre_trained'] == True:
        pre_trained_weight = torch.load("LaneGCN/pre_trained" + '/36.000.ckpt')
        print('pretrained weight is loaded from "LaneGCN/pre_trained/36.0000.ckpt"')
        pretrained_dict = pre_trained_weight['state_dict']
        new_model_dict = encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        encoder.load_state_dict(new_model_dict)
    encoder = encoder.cuda()
    loss = Loss(config).cuda()

    params = encoder.parameters()
    opt = Optimizer(params, config)

    return config, config_enc, Dataset, collate_fn, encoder, loss, opt
