import os
import copy
import pickle
import sys
import time
import os.path as osp
import shlex
import shutil
import subprocess

#import lmdb
#import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
import tqdm

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../'))
from data.data_specification import TASKS_SG14K
from data.SGNLoader import *
from geometry_utils import regularize_pc_point_count
from visualize import draw_scene, get_gripper_control_points

def get_task1_hits_sg14k(object_task_pairs, num_grasps=20):
    candidates = object_task_pairs['False']
    lines = []
    label = -1  # All grasps are negatives
    for ot in candidates:
        for grasp_idx in range(num_grasps):
            obj, task = ot.split('-')
            line = "{}-{}-{}-{}-{}-{}:{}\n".format(
                obj, str(grasp_idx), task, 'na', 'na', 'na', label)
            lines.append(line)
    return lines


def get_ot_pairs_sg14k(task1_results_file):
    lines = read_txt_file_lines(task1_results_file)
    object_task_pairs = defaultdict(list)
    for line in lines:
        assert isinstance(line, str)
        line = line.split('\n')[0]
        (obj_instance, task, _, label) = line.split('-')
        ot_pair = "{}-{}".format(obj_instance, task)
        object_task_pairs[label].append(ot_pair)
    object_task_pairs = dict(object_task_pairs)
    return object_task_pairs


def parse_line_sg14(line):
    assert isinstance(line, str)
    line = line.split('\n')[0]
    (data_dsc, label) = line.split(':')
    label = int(label)
    label = label == 1

    obj, grasp_id, task, _, _, _ = data_dsc.split('-')
    obj = str(obj)
    grasp_id = int(grasp_id)
    task = str(task)
    obj_class = obj[obj.find('_') + 1:]
    return obj, obj_class, grasp_id, task, label


class SG14K(data.Dataset):
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
            use_task1_grasps=True):
        """
        train: 1 for train, 0 for test, 2 for validation
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

        self._train = train
        self._map_obj2class = map_obj2class
        data_dir = os.path.join(base_dir, folder_dir, "scans")

        data_txt_splits = {
            0: 'test_split.txt',
            1: 'train_split.txt',
            2: 'val_split.txt'}
        if train not in data_txt_splits:
            raise ValueError("Unknown split arg {}".format(train))

        self._parse_func = parse_line_sg14

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
            get_ot_pairs_sg14k,
            get_task1_hits_sg14k)

        self._data = []
        self._pc = {}
        self._pc_mean = {}
        self._grasps = {}
        self._object_classes = class_list
        self._num_object_classes = len(self._object_classes)

        start = time.time()
        correct_counter = 0

        all_object_instances = []

        self._object_task_pairs_dataset = []
        self._data_labels = []
        self._data_label_counter = {0: 0, 1: 0}

        for i in tqdm.trange(len(lines)):
            obj, obj_class, grasp_id, task, label = self._parse_func(lines[i])
            obj_class = self._map_obj2class[obj]
            all_object_instances.append(obj)
            self._object_task_pairs_dataset.append("{}-{}".format(obj, task))

            pc_file = os.path.join(data_dir, obj, "pc.npy")
            if pc_file not in self._pc:
                if not os.path.exists(pc_file):
                    raise ValueError(
                        'Unable to find processed point cloud file {}'.format(pc_file))
                pc = np.load(pc_file)
                pc_mean = pc[:, :3].mean(axis=0)
                pc[:, :3] -= pc_mean
                self._pc[pc_file] = pc
                self._pc_mean[pc_file] = pc_mean

            grasp_idx = os.path.join(data_dir, obj, "grasps", str(grasp_id))
            if grasp_idx not in self._grasps:
                grasps = np.load(os.path.join(data_dir, obj, "grasps.npy"))
                grasp = grasps[int(grasp_id), :, :]
                grasp[:3, 3] -= self._pc_mean[pc_file]
                self._grasps[grasp_idx] = grasp

            self._data.append(
                (grasp_idx, pc_file, obj, obj_class, grasp_id, task, label))
            self._data_labels.append(int(label))
            if label:
                correct_counter += 1
                self._data_label_counter[1] += 1
            else:
                self._data_label_counter[0] += 1

        self._all_object_instances = list(set(all_object_instances))
        self._len = len(self._data)
        print('Loading files from {} took {}s; overall dataset size {}, proportion successful grasps {:.2f}'.format(
            data_txt_splits[self._train], time.time() - start, self._len, float(correct_counter / self._len)))

        self._data_labels = np.array(self._data_labels)

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

    def __getitem__(self, idx):

        grasp_idx, pc_file, obj, obj_class, grasp_id, task, label = self._data[idx]
        pc = self._pc[pc_file]
        pc = regularize_pc_point_count(
            pc, self._num_points, use_farthest_point=False)
        pc_color = pc[:, 3:]
        pc = pc[:, :3]

        grasp = self._grasps[grasp_idx]
        task_id = self._tasks.index(task)
        class_id = self._object_classes.index(obj_class)
        instance_id = self._all_object_instances.index(obj)

        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc.T).T
        grasp_pc = grasp_pc[:, :3]

        latent = np.concatenate(
            [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        latent = np.expand_dims(latent, axis=1)
        pc = np.concatenate([pc, grasp_pc], axis=0)

        pc, grasp = pc_normalize(pc, grasp, pc_scaling=self._pc_scaling)
        pc = np.concatenate([pc, latent], axis=1)

        if self._transforms is not None:
            pc = self._transforms(pc)

        label = float(label)

        return pc, pc_color, task_id, class_id, instance_id, grasp, label

    def __len__(self):
        return self._len

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


if __name__ == "__main__":

    base_dir = '/media/adithya/bed7087e-028a-403f-b9e7-7b91d723ea20/SG/data/'
    folder_dir = 'SG14000_small'
    _, _, _, name2wn = pickle.load(
        open(os.path.join(base_dir, folder_dir, 'misc.pkl'), 'rb'))

    dset = SG14KCageDataset(
        4096,
        transforms=None,
        train=0,
        base_dir=base_dir,
        folder_dir=folder_dir,
        normal=False,
        tasks=TASKS_SG14K,
        map_obj2class=name2wn,
        split_mode='otg',
        split_idx=0,
        split_version=0,
        pc_scaling=True,
        use_task1_grasps=True,
    )

    weights = dset.weights
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))

    dloader = torch.utils.data.DataLoader(dset, batch_size=16, sampler=sampler)
    counter = 0
    with torch.no_grad():
        for batch in dloader:
            pc, pc_color, task_id, class_id, instance_id, grasp, label = batch
            print(counter)
            counter += 1

            # pc_0 = pc[0, :, :3]
            # grasp_0 = grasp[0, :, :]
            # draw_scene(pc_0, np.expand_dims(grasp_0, axis=0))
