
import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import sys
from importlib import import_module
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd
from torch.utils.data.distributed import DistributedSampler

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
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, _, Dataset, collate_fn, _, _, _ = model.get_model(args.base_model)

    config['val_data_analysis'] = 'Data_analysis/val_data.csv'
    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=config["workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    data_anal(train_loader)
    data_anal(val_loader)

def data_anal(data_loader):
    for i in range(len(data_loader.dataset.split)):
        data = data_loader.dataset.split[i]
