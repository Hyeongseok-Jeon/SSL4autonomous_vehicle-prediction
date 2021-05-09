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
config_action_emb["num_channels"] = [128, 128, 128, 128]
config_action_emb["kernel_size"] = 2
config_action_emb["dropout"] = 0.2
config_action_emb["n_hid"] = 128
config_enc['action_emb'] = config_action_emb
config_enc['auxiliary'] = True
config_enc['pre_trained'] = True
config_enc['pre_trained_weight'] = os.path.join(root_path, 'results', 'SSL_encoder_initialize_with_pretrained_lanegcn_and_freeze', '300.000.ckpt')

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
        output = self.linear(output)
        return self.sig(output)


class SSL_encoder(nn.Module):
    def __init__(self, config, base_model):
        super(SSL_encoder, self).__init__()
        self.config = config
        self.relu6 = nn.ReLU6()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.base_net = base_model.Net(config)
        self.action_emb = TCN(input_size=102,
                              output_size=config_action_emb["output_size"],
                              num_channels=config_action_emb["num_channels"],
                              kernel_size=config_action_emb["kernel_size"],
                              dropout=config_action_emb["dropout"])
        self.out = nn.Linear(config_action_emb["output_size"] * 2, config_action_emb["n_hid"])
        self.auxiliary = nn.Linear(config_action_emb["n_hid"], config_action_emb["n_hid"])

    def forward(self, data):
        if 'backbone' in self.config['freeze']:
            with torch.no_grad():
                actors, veh_in_batch = self.base_net(data)
        else:
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
        hid_tmp = self.tanh(self.out(samples))
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
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.infonce = nn.CosineEmbeddingLoss()

    def forward(self, hid):
        if isinstance(hid[0], list):
            hid = hid[1]
        batch_num = hid[0].shape[0]
        hid_positive = hid[0]
        hid_anchor = hid[1]

        anchor_batch = torch.repeat_interleave(hid_anchor,batch_num, dim=0)
        sample_batch = torch.cat([torch.cat([hid_positive[i:i+1], torch.cat([hid_positive[:i,:], hid_positive[i+1:,:]])]) for i in range(batch_num)])
        idx = -torch.ones_like(anchor_batch[:,0])
        for i in range(batch_num):
            idx[batch_num*i] = 1

        loss_out = self.infonce(anchor_batch, sample_batch, idx)

        #
        # cos_sim_out = cos_sim(anchor_batch, sample_batch)
        # cos_sim_out = 1-torch.acos(cos_sim_out)/np.pi
        # loss_tot = 0
        # for i in range(batch_num):
        #     num = cos_sim_out[batch_num*i]
        #     den = sum(cos_sim_out[batch_num*i+1 : batch_num*(i+1)]) + num
        #     loss_batch = -torch.log(num/den)
        #     loss_tot = loss_tot + loss_batch
        # loss_out = loss_tot/batch_num
        # samples = torch.zeros_like(torch.cat([hid_anchor, hid_positive]))
        # anc_idx = torch.arange(batch_num) * 2
        # pos_idx = torch.arange(batch_num) * 2 + 1
        # samples[anc_idx] = hid_anchor
        # samples[pos_idx] = hid_positive
        # labels = torch.arange(2 * batch_num)
        # labels[anc_idx] = labels[pos_idx]

        # infoNCE_loss = infoNCELoss(samples, labels)
        # infoNCE_loss = self.l1loss(hid_anchor, hid_positive)
        return loss_out



def get_model(args):
    base_model_name = args.base_model
    base_model = import_module(base_model_name + '_backbone')
    config = base_model.config
    Dataset = base_model.ArgoDataset
    collate_fn = base_model.collate_fn

    config['freeze'] = args.freeze
    encoder = SSL_encoder(config, base_model)
    if 'backbone' in args.transfer:
        pre_trained_weight = torch.load("LaneGCN/pre_trained" + '/36.000.ckpt')
        print('backbone is transferred')
        pretrained_dict = pre_trained_weight['state_dict']
        new_model_dict = encoder.base_net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        encoder.base_net.load_state_dict(new_model_dict)

    encoder = encoder.cuda()
    loss = Loss(config).cuda()

    if 'backbone' in args.freeze:
        print('backbone is freezed')
        params_wrap = [(name, param) for name, param in encoder.action_emb.named_parameters()]
        params_out = [(name, param) for name, param in encoder.out.named_parameters()]
        params_aux = [(name, param) for name, param in encoder.auxiliary.named_parameters()]

        params_wrap = [p for n, p in params_wrap]
        params_out = [p for n, p in params_out]
        params_aux = [p for n, p in params_aux]

        params = params_wrap + params_aux + params_out
        opt = Optimizer(params, config)
    else:
        params = encoder.parameters()
        opt = Optimizer(params, config)

    return config, config_enc, Dataset, collate_fn, encoder, loss, opt
