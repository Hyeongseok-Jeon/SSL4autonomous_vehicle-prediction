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
import pickle

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

    train_mod(config, train_loader)
    val_mod(config, val_loader)


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train_mod(config, train_loader):
    store = train_loader.dataset.split
    get_idx = []
    for i in range(len(store)):
        if not (store[i]['action'] == 'error'):
            if not (np.sum(np.isnan(store[i]['ego_aug']['traj'])) > 0):
                get_idx.append(i)
    new_store = [store[i] for i in get_idx]
    f = open(os.path.join(root_path, 'preprocess', config['preprocess_train']), 'wb')
    pickle.dump(new_store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def val_mod(config, val_loader):
    store = val_loader.dataset.split
    get_idx = []
    for i in range(len(store)):
        if not (store[i]['action'] == 'error'):
            if not (np.sum(np.isnan(store[i]['ego_aug']['traj'])) > 0):
                get_idx.append(i)
    new_store = [store[i] for i in get_idx]
    f = open(os.path.join(root_path, 'preprocess', config['preprocess_val']), 'wb')
    pickle.dump(new_store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


if __name__ == "__main__":
    main()
