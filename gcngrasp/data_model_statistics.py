import argparse
import os
import copy
import pickle
import sys
import time
import os.path as osp
import shlex
import shutil
import subprocess

import numpy as np
import torch
import torch.utils.data as data
import tqdm
from ordered_set import OrderedSet

from pprint import pprint

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../'))
sys.path.append(os.path.join(BASE_DIR, '../'))
from visualize import draw_scene, get_gripper_control_points
from geometry_utils import regularize_pc_point_count
from data.SGNLoader import pc_normalize, get_task1_hits
from data.data_specification import TASKS
from utils.splits import get_split_data, parse_line, get_ot_pairs_taskgrasp
from data.Dataloader import BaselineData

import matplotlib.pyplot as plt
import open3d as o3d
from config import get_cfg_defaults
from models.baseline import BaselineNet

def num_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def model_statistics(model, cfg):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"attention layers ({cfg.num_attn_layers}): {num_parameters(model.attention_layers)}")
    print(f"pointnet layers: {num_parameters(model.pointnet)}")
    print(f"total: {total_params}")

def data_statistics(dataset: BaselineData):
    stats, label_per_obj_task = dataset.get_statistics()
    print(f"label cnt: {stats['label cnt']}")
    for name in stats:
        stat = stats[name]
        vals = [(stat[k] if isinstance(stat[k], int) else len(stat[k])) for k in stat]
        num = len(vals)
        mean = sum(vals)/num
        print(f"{name}: {vals[:20]} ... -> (cnt, mean): {num} {mean}")

    # analyze object task combination statistics
    total_per_combination = [
        sum(label_per_obj_task[k])
        for k in label_per_obj_task
    ]
    num = len(total_per_combination)
    mean = sum(total_per_combination)/num
    print(f"cnt per (object, task): {num} {mean}")

    vals = [
        label_per_obj_task[k][1] / sum(label_per_obj_task[k])
        for k in label_per_obj_task
    ]
    num = len(vals)
    mean = sum(vals)/num
    print(f"all per (object, task): {num} {mean}")
    non_zero_vals = [x for x in vals if x>0]
    num = len(non_zero_vals)
    mean = sum(non_zero_vals)/num
    print(f"non-zero per (object, task): {num} {mean}")

def dataloader_statistics(dataloader):
    print("="*20 + "dataloader stats" + "="*20)
    cnt_label = [0, 0]
    item_cnt = 0
    for batch in dloader:
        object_pc, grasp_pc, task_id, label = batch
        item_cnt += label.shape[0]
        for i in range(label.shape[0]):
            cur_label = int(label[i].item())
            cnt_label[cur_label] += 1
        if item_cnt%1000 == 0:
            print(f"{item_cnt}: {cnt_label}")
    print(f"labels {cnt_label}")

def pointcloud_stats(dataloader):
    res = []
    def pc_stats(pc):
        print(pc.shape)
        mean = torch.mean(pc, dim=0)
        pc = pc - mean
        std = torch.std(pc, dim=0)
        mins = torch.min(pc, axis=0)[0]
        maxs = torch.max(pc, axis=0)[0]
        res.append((mean.numpy(), std.numpy(), mins.numpy(), maxs.numpy()))

    for batch in dloader:
        object_pc, grasp_pc, task_id, label = batch
        object_pc = object_pc[:, :, :3]
        for i in range(object_pc.shape[0]):
            pc_stats(object_pc[i])
        if len(res) > 5:
            break

    for x in res:
        print(x)

def get_cfg(args):
    cfg = get_cfg_defaults()

    if args.cfg_file != '':
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)
        else:
            raise FileNotFoundError(args.cfg_file)

    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument('--base_dir', default='', type=str)
    parser.add_argument(
        '--cfg_file',
        help='yaml file in YACS config format to override default configs',
        default='',
        type=str)
    args = parser.parse_args()

    cfg = get_cfg(args)

    model = BaselineNet(cfg)

    model.prepare_data()
    dset = model.train_dset
    #dset = model.val_dset
    dloader = model.train_dataloader()

    #model_statistics(model, cfg)
    #data_statistics(dset)
    #dataloader_statistics(dloader)
    pointcloud_stats(dloader)
