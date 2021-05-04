# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd

from torch.utils.data.distributed import DistributedSampler

from LaneGCN.utils import Logger, load_pretrain

from mpi4py import MPI

comm = MPI.COMM_WORLD
hvd.init()
torch.cuda.set_device(hvd.local_rank())

# root_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.getcwd()
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="SSL_encoder", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--base_model", default="LaneGCN.lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--memo", default="_initialize_with_pretrained_lanegcn"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)


def main():
    seed = hvd.rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, config_enc, Dataset, collate_fn, net, loss, opt = model.get_model(args.base_model)

    if config["horovod"]:
        opt.opt = hvd.DistributedOptimizer(
            opt.opt, named_parameters=net.named_parameters()
        )

    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(net, ckpt["state_dict"])
        if args.resume:
            config["epoch"] = ckpt["epoch"]
            opt.load_state_dict(ckpt["opt_state"])

    if args.eval:
        # Data loader for evaluation
        dataset = Dataset(config["val_split"], config, train=False)
        val_sampler = DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        val_loader = DataLoader(
            dataset,
            batch_size=config["val_batch_size"],
            num_workers=config["val_workers"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        val(config, config_enc, val_loader, net, loss, 999)
        return

    # Create log and copy all code
    save_dir = config_enc["save_dir"] + args.memo
    log = os.path.join(save_dir, "log")
    if hvd.rank() == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sys.stdout = Logger(log)

        src_dirs = [root_path]
        dst_dirs = [os.path.join(save_dir, "files")]
        for src_dir, dst_dir in zip(src_dirs, dst_dirs):
            files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        src_dirs = [os.path.join(root_path, 'LaneGCN')]
        dst_dirs = [os.path.join(save_dir, "files", 'LaneGCN')]
        for src_dir, dst_dir in zip(src_dirs, dst_dirs):
            files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    train_sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    config["display_iters"] = len(train_loader.dataset.split)
    config["val_iters"] = len(train_loader.dataset.split) * 2

    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt.opt, root_rank=0)

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    if hvd.rank() == 0:
        print('logging directory :  ' + save_dir)
    for i in range(remaining_epochs):
        check = train(epoch + i, config, config_enc, train_loader, net, loss, opt, val_loader)
        if check == 0:
            break


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, config_enc, train_loader, net, loss, opt, val_loader=None):
    train_loader.sampler.set_epoch(int(epoch))
    net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (hvd.size() * config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (hvd.size() * config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    loss_tot = 0
    loss_calc = 0
    for i, data in tqdm(enumerate(train_loader), disable=hvd.rank()):
        epoch += epoch_per_batch
        data = dict(data)

        output = net(data)
        loss_out = loss(output)

        opt.zero_grad()
        loss_out.backward()
        loss_tot = loss_tot + loss_out.item()
        loss_calc = loss_calc + 1
        lr = opt.step(epoch)

        if torch.isnan(loss_out):
            print('output')
            print(output)
            hid = output
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
            print('nan loss')
            return 0

        num_iters = int(np.round(epoch * num_batches))
        if hvd.rank() == 0 and (
                num_iters % save_iters == 0 or epoch >= config["num_epochs"]
        ):
            save_ckpt(net, opt, config_enc["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            if hvd.rank() == 0:
                print(
                    "infoNCE loss  = %2.4f, time = %2.4f"
                    % (loss_tot / loss_calc, dt)
                )
            start_time = time.time()
            loss_tot = 0
            loss_calc = 0

        if num_iters % val_iters == 0:
            val(config, config_enc, val_loader, net, loss, epoch)

        if epoch >= config["num_epochs"]:
            val(config, config_enc, val_loader, net, loss, epoch)
            return 1


def val(config, config_enc, data_loader, net, loss, epoch):
    net.eval()

    start_time = time.time()
    loss_tot = 0
    loss_calc = 0
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output)
            loss_tot = loss_tot + loss_out.item()
            loss_calc = loss_calc + 1
    dt = time.time() - start_time
    if hvd.rank() == 0:
        print(
            "validation infoNCE loss  = %2.4f, time = %2.4f"
            % (loss_tot / loss_calc, dt)
        )
    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


def sync(data):
    data_list = comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data


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
    print(den)
    return torch.sum(1 - (torch.arccos(sim) / np.pi))

if __name__ == "__main__":
    main()
