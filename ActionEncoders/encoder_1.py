# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys

sys.path.extend(['/home/jhs/Desktop/SSL4autonomous_vehicle-prediction/LaneGCN'])
sys.path.extend(['/home/user/data/HyeongseokJeon/SSL4autonomous_vehicle-prediction/LaneGCN'])

import torch.nn as nn
from torch.nn.utils import weight_norm
from LaneGCN.utils import gpu, to_long, Optimizer, StepLR
import torch

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
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.tanh1, self.dropout1,
                                 self.conv2, self.chomp2, self.tanh2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.tanh(out + res)


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
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)

        return self.tanh(output)


class encoder(nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        self.config = config
        self.relu = nn.LeakyReLU(inplace=True)

        self.action_emb = TCN(input_size=102,
                              output_size=config_action_emb["output_size"],
                              num_channels=config_action_emb["num_channels"],
                              kernel_size=config_action_emb["kernel_size"],
                              dropout=config_action_emb["dropout"])
        self.out = nn.Linear(config_action_emb["output_size"] * 2, config_action_emb["n_hid"])

    def forward(self, actors, actor_idcs, data):
        batch_num = len(data['city'])
        action_original = torch.cat([gpu(data['action'][i][0:1, 0, :, :]) for i in range(batch_num)])

        target_idx = torch.cat([x[1].unsqueeze(dim=0) for x in actor_idcs])
        actors_target = actors[target_idx]
        hid_act = self.action_emb(action_original)[:, -1, :]
        sample = torch.cat([hid_act, actors_target], dim=1)

        out = self.out(sample)
        action_conditional_hid = self.relu(out)
        return action_conditional_hid, torch.norm(actors_target), torch.norm(hid_act)
