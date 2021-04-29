# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail


import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate

'''
additional data list
1. gt_hists
2. file_name
3. ref_path
4. ego_aug
5. action
6. ego_maneuver

TDL
5. maneuver class output (leftward with acc, leftward with dcc, straight with acc, straight with dcc, rightward with acc, rightward with dcc)
6. 
7. 
'''


class ArgoDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train
        self.split = split
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
            else:
                self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split)
            self.avl.seq_list = sorted(self.avl.seq_list)
            self.am = ArgoverseMap()

        if 'raster' in config and config['raster']:
            # TODO: DELETE
            self.map_query = MapQuery(config['map_scale'])

    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]

            if self.train and self.config['rot_aug']:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds', 'gt_hists', 'file_name', 'ref_path', 'ego_aug', 'action', 'ego_maneuver']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']  # np.pi * 2.0
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                graph = dict()
                for key in ['num_nodes', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'left_pairs', 'right_pairs', 'left', 'right']:
                    graph[key] = ref_copy(data['graph'][key])
                graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
                graph['feats'] = np.matmul(data['graph']['feats'], rot)
                new_data['graph'] = graph
                data = new_data
            else:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph', 'gt_hists', 'file_name', 'ref_path', 'ego_aug', 'action','ego_maneuver']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                data = new_data

            if 'raster' in self.config and self.config['raster']:
                data.pop('graph')
                x_min, x_max, y_min, y_max = self.config['pred_range']
                cx, cy = data['orig']

                region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
                raster = self.map_query.query(region, data['theta'], data['city'])

                data['raster'] = raster
            return data

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['idx'] = idx
        data = self.get_ref_path_agent(data)
        data = self.get_ego_augmentation(data)
        data = self.get_action_representation(data)

        if 'raster' in self.config and self.config['raster']:
            x_min, x_max, y_min, y_max = self.config['pred_range']
            cx, cy = data['orig']

            region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
            raster = self.map_query.query(region, data['theta'], data['city'])

            data['raster'] = raster
            return data

        data['graph'] = self.get_lane_graph(data)
        return data

    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)

    def read_argo_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        df = copy.deepcopy(self.avl[idx].seq_df)

        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)

        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        av_idx = obj_type.index('AV')

        idcs = objs[keys[agt_idx]]
        av_idcs = objs[keys[av_idx]]

        agt_traj = trajs[idcs]
        agt_step = steps[idcs]
        av_traj = trajs[av_idcs]
        av_step = steps[av_idcs]
        file_name = self.avl[idx].current_seq
        file_name = os.path.basename(file_name).split('/')[0]
        del keys[agt_idx]
        del keys[av_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [av_traj] + [agt_traj] + ctx_trajs
        data['steps'] = [av_step] + [agt_step] + ctx_steps
        data['file_name'] = file_name
        return data

    def get_obj_feats(self, data):
        orig = data['trajs'][0][19].copy().astype(np.float32)

        if self.train and self.config['rot_aug']:
            theta = np.random.rand() * np.pi * 2.0
        else:
            pre = data['trajs'][0][18] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])

        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, ctrs, gt_preds, has_preds, gt_hists = [], [], [], [], []
        for traj, step in zip(data['trajs'], data['steps']):
            if 19 not in step:
                continue

            gt_pred = np.zeros((30, 2), np.float32)
            has_pred = np.zeros(30, np.bool)
            future_mask = np.logical_and(step >= 20, step < 50)
            post_step = step[future_mask] - 20
            post_traj = traj[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1

            obs_mask = step < 20
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]

            for i in range(len(step)):
                if step[i] == 19 - (len(step) - 1) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((20, 3), np.float32)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2] = 1.0

            x_min, x_max, y_min, y_max = self.config['pred_range']
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue
            traj_save = np.zeros(shape=(20, 2))
            traj_save[-traj.shape[0]:, :] = traj
            gt_hists.append(traj_save)
            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        gt_hists = np.asarray(gt_hists, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data['gt_hists'] = gt_hists
        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        return data

    def get_ref_path_agent(self, data):
        # data = read_argo_data(12345)
        # data = get_obj_feats(data)
        # target_traj_agent = np.concatenate([data['gt_hists'][1], data['gt_preds'][1]], axis=0)
        # target_traj_av = np.concatenate([data['gt_hists'][0], data['gt_preds'][0]], axis=0)
        # cl_list = am.get_candidate_centerlines_for_traj(target_traj_agent, data['city'], viz=True)
        # cl_list = am.get_candidate_centerlines_for_traj(target_traj_av, data['city'], viz=True)
        ref_path_array = np.zeros(shape=(data['gt_hists'].shape[0], 100, 2))
        for k in range(data['gt_hists'].shape[0]):
            hist_traj = data['gt_hists'][k][np.nonzero(data['gt_hists'][k][:, 0]), :][0]
            if hist_traj.shape[0] == 1:
                hist_traj = np.concatenate([hist_traj, hist_traj], axis=0)
            cl_list = self.am.get_candidate_centerlines_for_traj(hist_traj, data['city'], viz=False)
            cl_dense_list = []
            dist_to_ref = []
            for i in range(len(cl_list[0])):
                cl = cl_list[0][i]
                cl_cands_mod = sparse_wp(cl)
                dist = 0
                for j in range(hist_traj.shape[0]):
                    dist = dist + np.min(np.linalg.norm(cl_cands_mod - hist_traj[j], axis=1))
                dist_to_ref.append(dist)
                cl_dense_list.append(cl_cands_mod)
            ref_path = cl_dense_list[np.argmin(dist_to_ref)]
            cur_idx = np.argmin(np.linalg.norm(ref_path - hist_traj[-1], axis=1))
            ref_path_final = ref_path[cur_idx:cur_idx + 100]
            try:
                ref_path_array[k, :, :] = ref_path_final
            except:
                ref_path_array[k, :, :] = -1
        data['ref_path'] = ref_path_array

        return data

    def get_ego_augmentation(self, data):
        ego_end_point_original = data['gt_preds'][0][-1]
        path_cands = self.am.get_candidate_centerlines_for_traj(data['gt_hists'][0], data['city'], viz=False)

        seg_lists = []
        for i in range(len(path_cands[1])):
            seg_lists = seg_lists + path_cands[1][i]
        seg_lists = list(dict.fromkeys(seg_lists))

        closest_lane_obj, conf, dense_centerline, nearby_lane_ids, per_lane_dists = self.am.get_nearest_centerline(ego_end_point_original, data['city'], visualize=False)
        original_dir = self.am.get_lane_direction(data['gt_hists'][0][-1], data['city'])
        original_dir = np.rad2deg(np.arctan2(original_dir[0][1], original_dir[0][0]))
        final_pos_cands = []
        final_pos_segments = []
        for i in range(len(nearby_lane_ids)):
            if nearby_lane_ids[i] in seg_lists:
                final_pos_cands.append(self.am.get_cl_from_lane_seq([[nearby_lane_ids[i]]], data['city'])[0])
                final_pos_segments.append(nearby_lane_ids[i])
        final_pos_cands = np.concatenate(final_pos_cands)

        val_idx = []
        for i in range(final_pos_cands.shape[0]):
            check = 1
            cands = final_pos_cands[i]
            # directionality check
            direction = cands - data['gt_hists'][0][-1]
            direction = np.rad2deg(np.arctan2(direction[1], direction[0]))
            if direction * original_dir < 0:
                check = 0

            # diversity check
            if (cands == closest_lane_obj.centerline).any():
                if np.linalg.norm(cands - ego_end_point_original) < 10:
                    check = 0

            if np.linalg.norm(cands - ego_end_point_original) > 15:
                check = 0

            if check == 1:
                val_idx.append(i)

        regen = True
        ego_aug = dict()
        ego_aug['traj'] = []
        ego_aug['relation'] = []
        aug_pos = []
        regen_check = [0, 0, 0, 0, 0, 0, 0, 0]
        regen_trial = 0
        if len(val_idx) == 0:
            aug_pos.append(data['gt_preds'][0][-1])
            ego_aug['relation'].append('no_augmentations')
        else:
            while regen:
                regen_trial = regen_trial + 1
                idx_cand = np.random.randint(len(val_idx))
                aus_pos_cand = final_pos_cands[val_idx[idx_cand]]
                closest_lane_obj_aug, _, _, _, _ = self.am.get_nearest_centerline(aus_pos_cand, data['city'], visualize=False)
                if closest_lane_obj_aug.id in seg_lists:
                    if closest_lane_obj_aug.l_neighbor_id is not None:
                        if closest_lane_obj_aug.l_neighbor_id == closest_lane_obj.id and regen_check[0] == 0:
                            ego_aug['relation'].append('right')
                            aug_pos.append(aus_pos_cand)
                            regen_check[0] = 1
                    if closest_lane_obj_aug.r_neighbor_id is not None:
                        if closest_lane_obj_aug.r_neighbor_id == closest_lane_obj.id and regen_check[1] == 0:
                            ego_aug['relation'].append('left')
                            aug_pos.append(aus_pos_cand)
                            regen_check[1] = 1
                    if closest_lane_obj_aug.successors is not None:
                        if closest_lane_obj.id in closest_lane_obj_aug.successors and regen_check[2] == 0:
                            ego_aug['relation'].append('brake')
                            aug_pos.append(aus_pos_cand)
                            regen_check[2] = 1
                    if closest_lane_obj_aug.predecessors is not None:
                        if closest_lane_obj.id in closest_lane_obj_aug.predecessors and regen_check[3] == 0:
                            ego_aug['relation'].append('accelerate')
                            aug_pos.append(aus_pos_cand)
                            regen_check[3] = 1
                    if closest_lane_obj_aug.successors is not None and closest_lane_obj.l_neighbor_id is not None:
                        if closest_lane_obj.l_neighbor_id in closest_lane_obj_aug.successors and regen_check[4] == 0:
                            ego_aug['relation'].append('left_brake')
                            aug_pos.append(aus_pos_cand)
                            regen_check[4] = 1
                    if closest_lane_obj_aug.predecessors is not None and closest_lane_obj.l_neighbor_id is not None:
                        if closest_lane_obj.l_neighbor_id in closest_lane_obj_aug.predecessors and regen_check[5] == 0:
                            ego_aug['relation'].append('left_accelerate')
                            aug_pos.append(aus_pos_cand)
                            regen_check[5] = 1
                    if closest_lane_obj_aug.successors is not None and closest_lane_obj.r_neighbor_id is not None:
                        if closest_lane_obj.r_neighbor_id in closest_lane_obj_aug.successors and regen_check[6] == 0:
                            ego_aug['relation'].append('right_brake')
                            aug_pos.append(aus_pos_cand)
                            regen_check[6] = 1
                    if closest_lane_obj_aug.predecessors is not None and closest_lane_obj.r_neighbor_id is not None:
                        if closest_lane_obj.r_neighbor_id in closest_lane_obj_aug.predecessors and regen_check[7] == 0:
                            ego_aug['relation'].append('right_accelerate')
                            aug_pos.append(aus_pos_cand)
                            regen_check[7] = 1
                    else:
                        pass
                if regen_trial > 2 * len(val_idx):
                    regen = False

        if len(aug_pos) == 0:
            aug_pos.append(data['gt_preds'][0][-1])
            ego_aug['relation'].append('no_augmentations')

        vel_list_prev_x = []
        vel_list_next_x = []
        vel_list_end_x = []
        vel_list_prev_y = []
        vel_list_next_y = []
        vel_list_end_y = []
        for i in range(5):
            vel_list_prev_x.append(np.abs(data['gt_hists'][0][-i - 1, 0] - data['gt_hists'][0][-i - 2, 0]) / 0.1)
            vel_list_prev_y.append(np.abs(data['gt_hists'][0][-i - 1, 1] - data['gt_hists'][0][-i - 2, 1]) / 0.1)

            vel_list_next_x.append(np.abs(data['gt_preds'][0][i, 0] - data['gt_preds'][0][i + 1, 0]) / 0.1)
            vel_list_next_y.append(np.abs(data['gt_preds'][0][i, 1] - data['gt_preds'][0][i + 1, 1]) / 0.1)

            vel_list_end_x.append(np.abs(data['gt_preds'][0][-i - 1, 0] - data['gt_preds'][0][-i - 2, 0]) / 0.1)
            vel_list_end_y.append(np.abs(data['gt_preds'][0][-i - 1, 1] - data['gt_preds'][0][-i - 2, 1]) / 0.1)

        prev_vel_x = np.mean(vel_list_prev_x)
        next_vel_x = np.mean(vel_list_next_x)
        end_vel_x = np.mean(vel_list_end_x)
        prev_vel_y = np.mean(vel_list_prev_y)
        next_vel_y = np.mean(vel_list_next_y)
        end_vel_y = np.mean(vel_list_end_y)
        vel_init_x = (prev_vel_x + next_vel_x) / 2
        vel_init_y = (prev_vel_y + next_vel_y) / 2

        disp_end = np.linalg.norm(data['gt_preds'][0][-1] - data['gt_hists'][0][-1])
        disp_end_aug = np.linalg.norm(aug_pos - data['gt_hists'][0][-1], axis=1)
        end_vel_aug_x = end_vel_x * disp_end_aug / disp_end
        end_vel_aug_y = end_vel_y * disp_end_aug / disp_end
        traj_aug = path_gen(data['gt_hists'][0][-1], [vel_init_x, vel_init_y], aug_pos, [end_vel_aug_x, end_vel_aug_y])

        ego_aug['traj'] = traj_aug[:, 1:, :]
        data['ego_aug'] = ego_aug
        return data

    def get_reaction_maneuver_class(self, data):
        sur_hists = data['gt_hists']
        sur_futs = data['gt_preds']

        i = 0
        hist_traj = sur_hists[i]
        path_cands = self.am.get_candidate_centerlines_for_traj(hist_traj, data['city'], viz=True)

        seg_lists = []
        for j in range(len(path_cands[1])):
            seg_lists = seg_lists + path_cands[1][j]
        seg_lists = list(dict.fromkeys(seg_lists))

        closest_lane_obj, conf, dense_centerline, nearby_lane_ids, per_lane_dists = self.am.get_nearest_centerline(hist_traj[0], data['city'], visualize=True)
        lane_scores = sorted(per_lane_dists)
        for j in range(len(lane_scores)):
            idx = np.where(per_lane_dists == lane_scores[j])[0][0]
            lane_id = nearby_lane_ids[idx]
            if lane_id in seg_lists:
                break
        ego_start_lane_obj = self.am.city_lane_centerlines_dict[data['city']][lane_id]

        fut_traj = sur_futs[i]
        path_cands = self.am.get_candidate_centerlines_for_traj(fut_traj, data['city'], viz=True)

        seg_lists = []
        for j in range(len(path_cands[1])):
            seg_lists = seg_lists + path_cands[1][j]
        seg_lists = list(dict.fromkeys(seg_lists))

        closest_lane_obj, conf, dense_centerline, nearby_lane_ids, per_lane_dists = am.get_nearest_centerline(fut_traj[-1], data['city'], visualize=True)
        lane_scores = sorted(per_lane_dists)
        for j in range(len(lane_scores)):
            idx = np.where(per_lane_dists == lane_scores[j])[0][0]
            lane_id = nearby_lane_ids[idx]
            if lane_id in seg_lists:
                break
        ego_fut_lane_obj = self.am.city_lane_centerlines_dict[data['city']][lane_id]

        if ego_start_lane_obj.turn_direction == 'NONE':
            connected_ids = self.am.dfs(ego_start_lane_obj.id, data['city'])
            seg_seq = 0
            for j in range(len(connected_ids)):
                if ego_fut_lane_obj.id in connected_ids[j]:
                    seg_seq = connected_ids[j]
                    break
            if seg_seq != 0:
                init_idx = seg_seq.index(ego_start_lane_obj.id)
                end_idx = seg_seq.index(ego_fut_lane_obj.id)
                maneuver = 'go_straight'
                for k in range(init_idx+1, end_idx+1):
                    if self.am.city_lane_centerlines_dict[data['city']][seg_seq[k]].turn_direction != 'NONE':
                        maneuver = self.am.city_lane_centerlines_dict[data['city']][seg_seq[k]].turn_direction
            else:
                init_left_id, init_right_id = ego_start_lane_obj.l_neighbor_id, ego_start_lane_obj.r_neighbor_id
                left_connected_ids = self.am.dfs(init_left_id, data['city'])
                right_connected_ids = self.am.dfs(init_right_id, data['city'])
                seg_seq = 0
                for j in range(len(left_connected_ids)):
                    if ego_fut_lane_obj.id in left_connected_ids[j]:
                        seg_seq = left_connected_ids[j]
                        break
                if seg_seq == 0:
                    for j in range(len(right_connected_ids)):
                        if ego_fut_lane_obj.id in right_connected_ids[j]:
                            seg_seq = right_connected_ids[j]
                            break
                    if seg_seq == 0:
                        maneuver = 'not_defined'
                    else:
                        maneuver = 'right_lane_change'
                else:
                    maneuver = 'left_lane_change'
        else:
            maneuver = ego_start_lane_obj.turn_direction
        data['ego_maneuver'] = maneuver

        for iz in range(sur_hists.shape[0]-1):
            i = iz + 1
            hist_traj = sur_hists[i]
            path_cands = am.get_candidate_centerlines_for_traj(hist_traj, data['city'], viz=True)

            seg_lists = []
            for j in range(len(path_cands[1])):
                seg_lists = seg_lists + path_cands[1][j]
            seg_lists = list(dict.fromkeys(seg_lists))

            closest_lane_obj, conf, dense_centerline, nearby_lane_ids, per_lane_dists = am.get_nearest_centerline(hist_traj[-1], data['city'], visualize=True)
            lane_scores = sorted(per_lane_dists)
            for j in range(len(lane_scores)):
                idx = np.where(per_lane_dists == lane_scores[j])[0][0]
                lane_id = nearby_lane_ids[idx]
                if lane_id in seg_lists:
                    break
            sur_cur_lane_obj = am.city_lane_centerlines_dict[data['city']][lane_id]



        return data

    def get_action_representation(self, data):
        action_aug = data['ego_aug']['traj']
        action = data['gt_preds'][0]
        ref_path_sur = data['ref_path']

        action_new = np.zeros(shape=(ref_path_sur.shape[0], action_aug.shape[0] + 1, 30, 102), dtype=np.float32)
        for i in range(ref_path_sur.shape[0]):
            ref_path_tmp = ref_path_sur[i]
            for j in range(action_aug.shape[0] + 1):
                if j == 0:
                    action_tmp = action
                else:
                    k = j - 1
                    action_tmp = action_aug[k]
                action_repre = action_input_reform(ref_path_tmp, action_tmp)
                action_new[i, j, :, :] = action_repre

        data['action'] = action_new
        '''
        action_new[i, j, k, :]
        - j th ego action (j = 0 : original ego action)
        - based on reference path of the i th vehicle (i = 0 : ego vehicle)
        - at time k
        
        '''
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]

            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])

            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs

        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)

        for key in ['pre', 'suc']:
            if 'scales' in self.config and self.config['scales']:
                # TODO: delete here
                graph[key] += dilated_nbrs2(graph[key][0], graph['num_nodes'], self.config['scales'])
            else:
                graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])
        return graph

    def test_viz(self, data):
        import pandas as pd
        from pandas import Series, DataFrame
        from argoverse.visualization.visualize_sequences import viz_sequence
        seq_path = self.split + '/' + data['file_name']
        df = self.avl.get(seq_path).seq_df
        time_seq = np.unique(np.asarray(df.TIMESTAMP))
        time_fut = time_seq[20:]

        df_aug = dict()
        df_aug['TIMESTAMP'] = np.repeat(time_fut, data['ego_aug']['traj'].shape[0])
        df_aug_X = []
        df_aug_Y = []
        df_aug_TRACK_ID = []
        df_aug_CITY = []
        df_aug_OBJECT_TYPE = []
        for i in range(30):
            for j in range(data['ego_aug']['traj'].shape[0]):
                df_aug_X.append(data['ego_aug']['traj'][j, i, 0])
                df_aug_Y.append(data['ego_aug']['traj'][j, i, 1])
                df_aug_TRACK_ID.append(str(j))
                df_aug_CITY.append(data['city'])
                df_aug_OBJECT_TYPE.append('AV_aug')
        df_aug['TRACK_ID'] = df_aug_TRACK_ID
        df_aug['OBJECT_TYPE'] = df_aug_OBJECT_TYPE
        df_aug['X'] = df_aug_X
        df_aug['Y'] = df_aug_Y
        df_aug['CITY_NAME'] = df_aug_CITY

        df_aug_df = DataFrame(df_aug)
        df_new = pd.concat([df, df_aug_df])
        viz_sequence(df_new, show=True)


class ArgoTestDataset(ArgoDataset):
    def __init__(self, split, config, train=False):

        self.config = config
        self.train = train
        split2 = config['val_split'] if split == 'val' else config['test_split']
        split = self.config['preprocess_val'] if split == 'val' else self.config['preprocess_test']

        self.avl = ArgoverseForecastingLoader(split2)
        self.avl.seq_list = sorted(self.avl.seq_list)
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(split, allow_pickle=True)
            else:
                self.split = np.load(split, allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split)
            self.am = ArgoverseMap()

    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]
            data['argo_id'] = int(self.avl.seq_list[idx].name[:-4])  # 160547

            if self.train and self.config['rot_aug']:
                # TODO: Delete Here because no rot_aug
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds']:
                    new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']  # np.pi * 2.0
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                graph = dict()
                for key in ['num_nodes', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'left_pairs', 'right_pairs']:
                    graph[key] = ref_copy(data['graph'][key])
                graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
                graph['feats'] = np.matmul(data['graph']['feats'], rot)
                new_data['graph'] = graph
                data = new_data
            else:
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph', 'argo_id', 'city']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                data = new_data
            return data

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['graph'] = self.get_lane_graph(data)
        data['idx'] = idx
        return data

    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)


class MapQuery(object):
    # TODO: DELETE HERE No used
    """[Deprecated] Query rasterized map for a given region"""

    def __init__(self, scale, autoclip=True):
        """
        scale: one meter -> num of `scale` voxels 
        """
        super(MapQuery, self).__init__()
        assert scale in (1, 2, 4, 8)
        self.scale = scale
        root_dir = '/mnt/yyz_data_1/users/ming.liang/argo/tmp/map_npy/'
        mia_map = np.load(f"{root_dir}/mia_{scale}.npy")
        pit_map = np.load(f"{root_dir}/pit_{scale}.npy")
        self.autoclip = autoclip
        self.map = dict(
            MIA=mia_map,
            PIT=pit_map
        )
        self.OFFSET = dict(
            MIA=np.array([502, -545]),
            PIT=np.array([-642, 211]),
        )
        self.SHAPE = dict(
            MIA=(3674, 1482),
            PIT=(3043, 4259)
        )

    def query(self, region, theta=0, city='MIA'):
        """
        region: [x0,x1,y0,y1]
        city: 'MIA' or 'PIT'
        theta: rotation of counter-clockwise, angel/degree likd 90,180
        return map_mask: 2D array of shape (x1-x0)*scale, (y1-y0)*scale
        """
        region = [int(x) for x in region]

        map_data = self.map[city]
        offset = self.OFFSET[city]
        shape = self.SHAPE[city]
        x0, x1, y0, y1 = region
        x0, x1 = x0 + offset[0], x1 + offset[0]
        y0, y1 = y0 + offset[1], y1 + offset[1]
        x0, x1, y0, y1 = [round(_ * self.scale) for _ in [x0, x1, y0, y1]]
        # extend the crop region to 2x -- for rotation
        H, W = y1 - y0, x1 - x0
        x0 -= int(round(W / 2))
        y0 -= int(round(H / 2))
        x1 += int(round(W / 2))
        y1 += int(round(H / 2))
        results = np.zeros([H * 2, W * 2])
        # padding of crop -- for outlier
        xstart, ystart = 0, 0
        if self.autoclip:
            if x0 < 0:
                xstart = -x0
                x0 = 0
            if y0 < 0:
                ystart = -y0
                y0 = 0
            x1 = min(x1, shape[1] * self.scale - 1)
            y1 = min(y1, shape[0] * self.scale - 1)
        map_mask = map_data[y0:y1, x0:x1]
        _H, _W = map_mask.shape
        results[ystart:ystart + _H, xstart:xstart + _W] = map_mask
        results = results[::-1]  # flip to cartesian
        # rotate and remove margin
        rot_map = rotate(results, theta, center=None, order=0)  # center None->map center
        H, W = results.shape
        outputH, outputW = round(H / 2), round(W / 2)
        startH, startW = round(H // 4), round(W // 4)
        crop_map = rot_map[startH:startH + outputH, startW:startW + outputW]
        return crop_map


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs


def dilated_nbrs2(nbr, num_nodes, scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, max(scales)):
        mat = mat * csr

        if i + 1 in scales:
            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int64)
            nbr['v'] = coo.col.astype(np.int64)
            nbrs.append(nbr)
    return nbrs


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch


def circle_line_intersection(p2, p1, center):
    if p2[1] > p1[1]:
        y2 = p2[1] - center[1]
        y1 = p1[1] - center[1]
        x2 = p2[0] - center[0]
        x1 = p1[0] - center[0]
    else:
        y2 = p1[1] - center[1]
        y1 = p2[1] - center[1]
        x2 = p1[0] - center[0]
        x1 = p2[0] - center[0]

    dx = x2 - x1
    dy = y2 - y1
    dr = np.sqrt(dx ** 2 + dy ** 2)
    D = x1 * y2 - x2 * y1
    cand1 = [(D * dy + dx * np.sqrt(0.5 ** 2 * dr ** 2 - D ** 2)) / dr ** 2 + center[0], (-D * dx + dy * np.sqrt(0.5 ** 2 * dr ** 2 - D ** 2)) / dr ** 2 + center[1]]
    cand2 = [(D * dy - dx * np.sqrt(0.5 ** 2 * dr ** 2 - D ** 2)) / dr ** 2 + center[0], (-D * dx - dy * np.sqrt(0.5 ** 2 * dr ** 2 - D ** 2)) / dr ** 2 + center[1]]

    if min(p2[0], p1[0]) <= cand1[0] <= max(p2[0], p1[0]):
        if min(p2[0], p1[0]) <= cand2[0] <= max(p2[0], p1[0]):
            min_idx = np.argmin([np.linalg.norm(p2 - cand1), np.linalg.norm(p2 - cand2)])
            if min_idx == 0:
                point = cand1
            elif min_idx == 1:
                point = cand2
        else:
            point = cand1
    elif min(p2[0], p1[0]) <= cand2[0] <= max(p2[0], p1[0]):
        point = cand2
    else:
        point = None
    return point


def sparse_wp(cl):
    val_index = np.unique(cl[:, 0:1], return_index=True)[1]
    cl_valid = np.concatenate([np.expand_dims(cl[sorted(val_index), 0:1], 1), np.expand_dims(cl[sorted(val_index), 1:2], 1)], axis=1)
    cl_mod = []
    dist = []
    i = 0
    while i < cl_valid.shape[0]:
        if i == 0:
            cl_mod.append(cl[0, :])
            i += 1
        else:
            dist.append(np.linalg.norm(cl[i, :] - cl_mod[-1]))
            if dist[-1] > 0.5:
                while dist[-1] > 0.5:
                    cl_mod.append(np.asarray(circle_line_intersection(cl[i, :], cl[i - 1, :], cl_mod[-1])))
                    dist.append(np.linalg.norm(cl[i, :] - cl_mod[-1]))
            else:
                i += 1
    cl_mod = np.asarray(cl_mod, dtype=np.float32)
    return cl_mod


def path_gen(init_pos, init_vel, end_pos, end_vel):
    init_pos_x = init_pos[0]
    init_vel_x = init_vel[0]
    end_pos_x = [end_pos[i][0] for i in range(len(end_pos))]
    end_vel_x = end_vel[0]

    init_pos_y = init_pos[1]
    init_vel_y = init_vel[1]
    end_pos_y = [end_pos[i][1] for i in range(len(end_pos))]
    end_vel_y = end_vel[1]

    d_x = init_pos_x
    c_x = init_vel_x
    b_x = ((end_pos_x - 3 * c_x - d_x) - (end_vel_x - c_x)) / 3
    a_x = (end_pos_x - d_x - 3 * c_x - 9 * b_x) / 27

    d_y = init_pos_y
    c_y = init_vel_y
    b_y = ((end_pos_y - 3 * c_y - d_y) - (end_vel_y - c_y)) / 3
    a_y = (end_pos_y - d_y - 3 * c_y - 9 * b_y) / 27

    traj = np.zeros(shape=(len(end_vel_x), 31, 2), dtype=np.float32)
    for i in range(31):
        t = 0.1 * i
        x = a_x * t ** 3 + b_x * t ** 2 + c_x * t + d_x
        y = a_y * t ** 3 + b_y * t ** 2 + c_y * t + d_y

        traj[:, i, 0] = x
        traj[:, i, 1] = y
    return traj


def action_input_reform(ref_path, action):
    mask = np.zeros(shape=(30, 102))
    for i in range(30):
        action_pos = action[i]
        nearest_idx = np.argmin(np.linalg.norm(action_pos - ref_path, axis=1))
        if nearest_idx == 0:
            heading = np.arctan2(ref_path[1, 1] - ref_path[0, 1], ref_path[1, 0] - ref_path[0, 0])
        elif nearest_idx == 99:
            heading = np.arctan2(ref_path[99, 1] - ref_path[98, 1], ref_path[99, 0] - ref_path[98, 0])
        else:
            heading_prev = np.arctan2(ref_path[nearest_idx, 1] - ref_path[nearest_idx - 1, 1], ref_path[nearest_idx, 0] - ref_path[nearest_idx - 1, 0])
            heading_next = np.arctan2(ref_path[nearest_idx + 1, 1] - ref_path[nearest_idx, 1], ref_path[nearest_idx + 1, 0] - ref_path[nearest_idx, 0])
            heading = np.mean([heading_next, heading_prev])
        R = np.asarray([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        pos_del = action_pos - ref_path[nearest_idx]
        pos_new = np.matmul(R, pos_del)
        r = np.sqrt(np.sum(pos_new ** 2))
        alpha = np.arctan2(pos_new[1], pos_new[0])
        mask[i, nearest_idx] = 1
        mask[i, 100] = np.min([5, 1 / r])
        mask[i, 101] = alpha / np.pi

    return mask
