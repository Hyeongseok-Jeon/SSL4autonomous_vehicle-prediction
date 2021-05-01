import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from importlib import import_module
from LaneGCN.utils import gpu, to_long, Optimizer, StepLR
import numpy as np
import copy

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
config_enc['auxiliary'] = False

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
                              dropout=config_action_emb["dropout"])
        self.out = nn.Linear(config_action_emb["output_size"]*2, config_action_emb["n_hid"]).double().cuda()
        self.auxiliary = nn.Linear(config_action_emb["n_hid"], config_action_emb["n_hid"]).double().cuda()

    def forward(self, data):
        actors, veh_in_batch = base_net(data)
        batch_num = len(veh_in_batch)
        veh_num_in_batch = sum(veh_in_batch)
        ego_idx = [0] + [sum(veh_in_batch[:i+1]) for i in range(batch_num-1)]
        target_idx = [1] + [sum(veh_in_batch[:i+1])+1 for i in range(batch_num-1)]

        positive_idx = [np.random.randint(1, data['action'][i].shape[1]) for i in range(batch_num)]
        action_original = torch.cat([gpu(data['action'][i][:, 0, :, :]) for i in range(batch_num)])
        action_augmented = torch.cat([gpu(data['action'][i][:, positive_idx[i], :, :]) for i in range(batch_num)])
        actions = torch.cat([action_original, action_augmented])
        hid_act = action_emb(actions)[:,-1,:]
        hid_act_original = hid_act[:int(hid_act.shape[0]/2)]
        hid_act_augmented = hid_act[int(hid_act.shape[0]/2):]
        idx_mask = torch.arange(0, hid_act_original.shape[0])

        sample_pos = []
        sample_anchor = []
        sample_neg = []
        for i in range(batch_num):
            # positive_samples = action_augmented + target actirs
            # anchor = action_original + target actors
            # negative_samples = action_original + sur actors
            idxs = idx_mask[idx_mask!=ego_idx[i]]
            idxs = idxs[idxs!=target_idx[i]]

            positive_samples = torch.cat([hid_act_augmented[target_idx[i]], actors[target_idx[i]]]).unsqueeze(0)
            anchor = torch.cat([hid_act_original[target_idx[i]], actors[target_idx[i]]]).unsqueeze(0)
            negative_samples = torch.cat([hid_act_original[idxs], actors[idxs]], dim=1)

            sample_pos.append(positive_samples)
            sample_anchor.append(anchor)
            sample_neg.append(negative_samples)
        sample_pos = torch.cat(sample_pos)
        sample_anchor = torch.cat(sample_anchor)
        sample_neg = torch.cat(sample_neg)

        samples = torch.cat([sample_pos, sample_anchor, sample_neg])
        hid = relu(out(samples))
        if config_enc['auxiliary']:
            hid = auxiliary(hid)

        hid_positive = torch.cat([hid[i].unsqueeze(0) for i in range(batch_num)])
        hid_anchor = torch.cat([hid[i + batch_num].unsqueeze(0) for i in range(batch_num)])
        hid_negatvie = [hid[2*batch_num+(veh_num_in_batch-2)*(i):2*batch_num+(veh_num_in_batch-2)*(i+1)] for i in range(batch_num)]

        hid = [hid_positive, hid_anchor, hid_negatvie]
        return hid


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.cosine_sim = cosine_similarity

    def forward(self, hid):
        hid_positive = hid[0]
        hid_anchor = hid[1]
        hid_negative = hid[2]

        cos_sim_positive = cosine_sim(hid_positive, hid_anchor)
        cos_sim_negative = [torch.sum(cosine_sim(hid_negative[i], torch.repeat_interleave(hid_anchor[i].unsqueeze(0), hid_negative[i].shape[0], dim=0))) for i in range(hid_positive.shape[0])]

        return loss_out

def cosine_similarity(sample, anchor):
    return torch.sum(sample * anchor, dim=1)/(torch.norm(sample, dim=1) * torch.norm(anchor, dim=1))


def get_model(base_model_name):
    base_model = import_module(base_model_name + '_backbone')
    config = base_model.config
    Dataset = base_model.ArgoDataset
    collate_fn = base_model.collate_fn

    encoder = SSL_encoder(config, base_model)
    encoder = encoder.cuda()

    loss = Loss(config).cuda()

    params = encoder.parameters()
    opt = Optimizer(params, config)

    return config, Dataset, collate_fn, encoder, loss, opt
