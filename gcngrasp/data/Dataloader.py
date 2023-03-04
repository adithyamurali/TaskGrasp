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

import matplotlib.pyplot as plt
import open3d as o3d

import random
from collections import defaultdict


def visualize_pc(object_pc, grasp_pc):
    grasp_color = torch.Tensor([0, 255, 0]).double().repeat((grasp_pc.shape[0], 1))
    all_pc = torch.cat(
        [object_pc, torch.cat([grasp_pc, grasp_color], dim=1)],
        dim=0
    )
    draw_scene(pc=all_pc)

class BaselineData(data.Dataset):
    def __init__(
            self,
            num_points,
            transforms=None,
            train=0,
            download=True,
            base_dir=None,
            folder_dir='',
            normal=True,
            tasks=None,
            map_obj2class=None,
            class_list=None,
            split_mode=None,
            split_idx=0,
            split_version=0,
            pc_scaling=True,
            use_task1_grasps=True,
            graph_data_path='',
            include_reverse_relations=True):
        """

        Args:
            num_points: Number of points in point cloud (used to downsample data to a fixed number)
            transforms: Used for data augmentation during training
            train: 1 for train, 0 for test, 2 for validation
            base_dir: location of dataset
            folder_dir: name of dataset
            tasks: list of tasks
            class_list: list of object classes
            map_obj2class: dictionary mapping dataset object to corresponding WordNet class
            split_mode: Choose between held-out instance ('i'), tasks ('t') and classes ('o')
            split_version: Choose 1, 0 is deprecated
            split_idx: For each split mode, there are 4 cross validation splits (choose between 0-3)
            pc_scaling: True if you want to scale the point cloud by the standard deviation
            include_reverse_relations: True since we are modelling a undirected graph
            use_task1_grasps: True if you want to include the grasps from the object-task pairs
                rejected in Stage 1 (and add these grasps are negative samples)

            Deprecated args (not used anymore): normal, download, 
        """
        super().__init__()
        self._pc_scaling = pc_scaling
        self._split_mode = split_mode
        self._split_idx = split_idx
        self._split_version = split_version
        self._num_points = num_points
        self._transforms = transforms
        self._tasks = tasks
        self._num_tasks = len(self._tasks)

        task1_results_file = os.path.join(
            base_dir, folder_dir, 'task1_results.txt')
        assert os.path.exists(task1_results_file)

        self._train = train
        self._map_obj2class = map_obj2class
        data_dir = os.path.join(base_dir, folder_dir, "scans")

        data_txt_splits = {
            0: 'test_split.txt',
            1: 'train_split.txt',
            2: 'val_split.txt'}

        if self._train not in data_txt_splits:
            raise ValueError("Unknown split arg {}".format(self._train))

        self._parse_func = parse_line
        lines = get_split_data(
            base_dir,
            folder_dir,
            self._train,
            self._split_mode,
            self._split_idx,
            self._split_version,
            use_task1_grasps,
            data_txt_splits,
            self._map_obj2class,
            self._parse_func,
            get_ot_pairs_taskgrasp,
            get_task1_hits)

        self._data = []
        self._obj_data = {}
        self._grasps = {}
        self._object_classes = class_list

        start = time.time()
        correct_counter = 0

        all_object_instances = []

        self._object_task_pairs_dataset = []
        self._data_labels = []
        self._data_label_counter = {0: 0, 1: 0}

        self.init_statistics()

        good_ind = []
        #for i in tqdm.trange(len(lines[:2000])):
        for i in tqdm.trange(len(lines)):
            obj, obj_class, grasp_id, task, label = parse_line(lines[i])
            obj_class = self._map_obj2class[obj]
            # using updated class from wordnet
            self.update_statistics(obj, obj_class, grasp_id, task, label)

            all_object_instances.append(obj)
            self._object_task_pairs_dataset.append("{}-{}".format(obj, task))

            if obj not in self._obj_data:
                pc_file = os.path.join(data_dir, obj, "fused_pc_clean.npy")
                if not os.path.exists(pc_file):
                    raise ValueError(
                        'Unable to find processed point cloud file {}'.format(pc_file))
                pc = np.load(pc_file)
                # TODO: why subtract mean?
                pc_mean = pc[:, :3].mean(axis=0)
                pc[:, :3] -= pc_mean
                self._obj_data[obj] = pc

            grasp_file = os.path.join(
                data_dir, obj, "grasps", str(grasp_id), "grasp.npy")
            if grasp_file not in self._grasps:
                grasp = np.load(grasp_file)
                self._grasps[grasp_file] = grasp

            task_id = self._tasks.index(task)
            class_id = self._object_classes.index(obj_class) if self._object_classes is not None else -1

            self._data.append(
                (obj, grasp_file, task_id, obj, class_id, label))
            self._data_labels.append(int(label))
            if label:
                correct_counter += 1
                self._data_label_counter[1] += 1
                good_ind.append(i)
            else:
                self._data_label_counter[0] += 1

        pprint(self._data_label_counter)
        self._all_object_instances = list(set(all_object_instances))
        self._len = len(self._data)
        # TODO: changed this for debugging
        #self._len = 128
        #for i in range(0, self._len, 2):
        #    self._data[i], self._data[good_ind[i]] = self._data[good_ind[i]], self._data[i]
        print('Loading files from {} took {}s; overall dataset size {}, proportion successful grasps {:.2f}'.format(
            data_txt_splits[self._train], time.time() - start, self._len, float(correct_counter / self._len)))

        self._data_labels = np.array(self._data_labels)

    def init_statistics(self):
        self.stat_label_cnt = defaultdict(lambda: 0)
        self.stat_cnt_per_obj = defaultdict(lambda: 0)
        self.stat_cnt_per_task = defaultdict(lambda: 0)
        self.stat_cnt_per_class = defaultdict(lambda: 0)
        self.stat_cnt_per_grasp = defaultdict(lambda: 0)

        self.stat_obj_per_class = defaultdict(lambda: set())
        self.stat_task_per_class = defaultdict(lambda: set())
        self.stat_class_per_task = defaultdict(lambda: set())
        self.stat_task_per_grasp = defaultdict(lambda: set())
        self.stat_grasp_per_obj = defaultdict(lambda: set())

        self.label_per_obj_task = defaultdict(lambda: [0, 0])

    def update_statistics(self, obj, obj_class, grasp_id, task, label):
        self.stat_label_cnt[label] += 1
        self.stat_cnt_per_obj[obj] += 1
        self.stat_cnt_per_task[task] += 1
        self.stat_cnt_per_class[obj_class] += 1
        self.stat_cnt_per_grasp[(obj, grasp_id)] += 1

        self.stat_obj_per_class[obj_class].add(obj)
        self.stat_task_per_class[obj_class].add(task)
        self.stat_class_per_task[task].add(obj_class)
        self.stat_task_per_grasp[(obj, grasp_id)].add(task)
        self.stat_grasp_per_obj[obj].add(grasp_id)

        self.label_per_obj_task[(obj, task)][label] += 1
    
    def get_statistics(self):
        return ({
            "label cnt": self.stat_label_cnt,
            "obj cnt": self.stat_cnt_per_obj,
            "task cnt": self.stat_cnt_per_task,
            "class cnt": self.stat_cnt_per_class,
            "grasp cnt": self.stat_cnt_per_grasp,
            "obj per class": self.stat_obj_per_class,
            "task per class": self.stat_task_per_class,
            "class per task": self.stat_class_per_task,
            "task per grasp": self.stat_task_per_grasp,
            "grasp per obj": self.stat_grasp_per_obj,
        }, self.label_per_obj_task)

    @property
    def weights(self):
        N = self.__len__()
        weights = {
            0: float(N) /
            self._data_label_counter[0],
            1: float(N) /
            self._data_label_counter[1]}
        weights_sum = sum(weights.values())
        weights[0] = weights[0] / weights_sum
        weights[1] = weights[1] / weights_sum
        weights_data = np.zeros(N)
        weights_data[self._data_labels == 0] = weights[0]
        weights_data[self._data_labels == 1] = weights[1]
        return weights_data

    def get_task_from_id(self, task_id):
        return self._tasks[task_id]

    def get_instance_from_id(self, instance_id):
        return self._all_object_instances[instance_id]

    def __getitem__(self, idx):
        obj, grasp_file, task_id, obj, class_id, label = self._data[idx]
        obj_data = self._obj_data[obj]
        pc = regularize_pc_point_count(
            obj_data, self._num_points, use_farthest_point=False)

        grasp = self._grasps[grasp_file]

        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc.T).T
        grasp_pc = grasp_pc[:, :3]

        # TODO: latent indicator to pass all to pointnet
        #latent = np.concatenate(
        #    [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        #latent = np.expand_dims(latent, axis=1)

        # TODO: normalization
        all_pc = np.concatenate([pc[:, :3], grasp_pc], axis=0)
        all_pc = pc_normalize(all_pc, grasp=None, pc_scaling=self._pc_scaling)
        pc[:, :3] = all_pc[:pc.shape[0], :]
        grasp_pc = all_pc[-grasp_pc.shape[0]:, :]

        #pc = np.concatenate([pc, latent], axis=1)

        # TODO: data augmentations?
        #if self._transforms is not None:
        #    pc = self._transforms(pc)

        instance_id = self._all_object_instances.index(obj)

        label = float(label)

        return pc, grasp_pc, task_id, instance_id, class_id, grasp, label

    def __len__(self):
        return self._len

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument('--base_dir', default='', type=str)
    args = parser.parse_args()

    if args.base_dir != '':
        if not os.path.exists(args.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    args.base_dir))
    else:
        args.base_dir = os.path.join(os.path.dirname(__file__), '../../data')

    folder_dir = 'taskgrasp'
    _, _, _, name2wn = pickle.load(
        open(os.path.join(args.base_dir, folder_dir, 'misc.pkl'), 'rb'))

    dset = BaselineData(
        4096,
        transforms=None,
        train=1,  # train
        base_dir=args.base_dir,
        folder_dir=folder_dir,
        normal=False,
        tasks=TASKS,
        map_obj2class=name2wn,
        split_mode='t',
        split_idx=0,
        split_version=0,
        pc_scaling=True,
        use_task1_grasps=True,
        graph_data_path='kb2_task_wn_noi',
        include_reverse_relations=True
    )

    # TODO: enable weighted sampler to have equal number of positive/negative samples
    """
    weights = dset.weights
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))

    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=16,
        sampler=sampler)
    """
    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=1)

    with torch.no_grad():
        for batch in dloader:
            object_pc, grasp_pc, task_id, _, _, _, label = batch
            task_name = TASKS[task_id]
            print(f"visualizing for task {task_name} ({task_id.numpy()[0]}) -> label {label.numpy()[0]}")
            visualize_pc(object_pc[0], grasp_pc[0])
