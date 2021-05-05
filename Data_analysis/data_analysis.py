import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import sys
from importlib import import_module
from torch.utils.data import Sampler, DataLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import csv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

am = ArgoverseMap()


def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, _, Dataset, collate_fn, _, _, _ = model.get_model(args.base_model)

    config['val_data_analysis'] = 'Data_analysis/val_data.csv'
    config['train_data_analysis'] = 'Data_analysis/train_data.csv'
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

    data_anal(train_loader, config['train_data_analysis'])
    data_anal(val_loader, config['val_data_analysis'])


def data_anal(data_loader, save_dir):
    f = open(save_dir, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['file name', 'ego maneuver', 'ego augmented maneuver', 'target maneuver', 'minimum distance btw ego and target'])

    for i in range(len(data_loader.dataset.split)):
        print(i)
        data = data_loader.dataset.split[i]
        file_name = data['file_name']
        ego_maneuver = data['ego_maneuver']
        ego_augmented_maneuver = data['ego_aug']['relation']
        target_maneuver = get_reaction_maneuver_class(data)
        min_dist = get_min_dist(data)
        wr.writerow([file_name, ego_maneuver, ego_augmented_maneuver, target_maneuver, min_dist])
    f.close()


def get_min_dist(data):
    ego_traj = np.concatenate([data['gt_hists'][0], data['gt_preds'][0]])
    sur_traj = np.concatenate([data['gt_hists'][1], data['gt_preds'][1]])
    min_dist = []
    for i in range(ego_traj.shape[0]):
        ego_pos = ego_traj[i]
        min_dist.append(np.min(np.linalg.norm(ego_pos - sur_traj, axis=1)))
    return min(min_dist)


def get_reaction_maneuver_class(data):
    sur_hists = data['gt_hists']
    sur_futs = data['gt_preds']

    i = 1
    hist_traj = sur_hists[i]
    try:
        path_cands = am.get_candidate_centerlines_for_traj(hist_traj, data['city'], viz=False)
        seg_lists = []
        for j in range(len(path_cands[1])):
            seg_lists = seg_lists + path_cands[1][j]
        seg_lists = list(dict.fromkeys(seg_lists))

        closest_lane_obj, conf, dense_centerline, nearby_lane_ids, per_lane_dists = am.get_nearest_centerline(hist_traj[0], data['city'], visualize=False)
        lane_scores = sorted(per_lane_dists)
        for j in range(len(lane_scores)):
            idx = np.where(per_lane_dists == lane_scores[j])[0][0]
            lane_id = nearby_lane_ids[idx]
            if lane_id in seg_lists:
                break
        ego_start_lane_obj = am.city_lane_centerlines_dict[data['city']][lane_id]

        fut_traj = sur_futs[i]
        path_cands = am.get_candidate_centerlines_for_traj(fut_traj, data['city'], viz=False)

        seg_lists = []
        for j in range(len(path_cands[1])):
            seg_lists = seg_lists + path_cands[1][j]
        seg_lists = list(dict.fromkeys(seg_lists))

        closest_lane_obj, conf, dense_centerline, nearby_lane_ids, per_lane_dists = am.get_nearest_centerline(fut_traj[-1], data['city'], visualize=False)
        lane_scores = sorted(per_lane_dists)
        for j in range(len(lane_scores)):
            idx = np.where(per_lane_dists == lane_scores[j])[0][0]
            lane_id = nearby_lane_ids[idx]
            if lane_id in seg_lists:
                break
        ego_fut_lane_obj = am.city_lane_centerlines_dict[data['city']][lane_id]

        if ego_start_lane_obj.turn_direction == 'NONE':
            maneuver = 0
            connected_ids = am.dfs(ego_start_lane_obj.id, data['city'])
            seg_seq = 0
            for j in range(len(connected_ids)):
                if ego_fut_lane_obj.id in connected_ids[j]:
                    seg_seq = connected_ids[j]
                    break
            if seg_seq != 0:
                init_idx = seg_seq.index(ego_start_lane_obj.id)
                end_idx = seg_seq.index(ego_fut_lane_obj.id)
                maneuver = 'go_straight'
                for k in range(init_idx + 1, end_idx + 1):
                    if am.city_lane_centerlines_dict[data['city']][seg_seq[k]].turn_direction != 'NONE':
                        maneuver = am.city_lane_centerlines_dict[data['city']][seg_seq[k]].turn_direction
            else:
                init_left_id, init_right_id = ego_start_lane_obj.l_neighbor_id, ego_start_lane_obj.r_neighbor_id
                if init_left_id != None:
                    left_connected_ids = am.dfs(init_left_id, data['city'])
                    seg_seq = 0
                    for j in range(len(left_connected_ids)):
                        if ego_fut_lane_obj.id in left_connected_ids[j]:
                            seg_seq = left_connected_ids[j]
                            break
                    if seg_seq == 0:
                        maneuver = 'not_defined'
                    else:
                        maneuver = 'left_lane_change'
                if init_right_id != None:
                    right_connected_ids = am.dfs(init_right_id, data['city'])
                    seg_seq = 0
                    for j in range(len(right_connected_ids)):
                        if ego_fut_lane_obj.id in right_connected_ids[j]:
                            seg_seq = right_connected_ids[j]
                            break
                    if seg_seq == 0:
                        maneuver = 'not_defined'
                    else:
                        maneuver = 'right_lane_change'

        else:
            maneuver = ego_start_lane_obj.turn_direction
    except:
        maneuver = 'error: no nearby lanes found'
    return maneuver


if __name__ == "__main__":
    main()