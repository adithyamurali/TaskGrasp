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

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from ordered_set import OrderedSet
from networkx import convert_node_labels_to_integers
from networkx.generators.ego import ego_graph
from networkx.algorithms.operators.binary import compose

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../'))
from visualize import draw_scene, get_gripper_control_points
from geometry_utils import regularize_pc_point_count
from data.SGNLoader import pc_normalize, get_task1_hits
from data.data_specification import TASKS
from utils.splits import get_split_data, parse_line, get_ot_pairs_taskgrasp

def extract_subgraph(graph, object, task, hop=3):
    """
    This function extract n-hop neighbors for an object and task pairs.

    ToDo: using the subgraph is slightly more involved since node index in the subgraph needs to be adjusted

    :param object:
    :param task:
    :param hop:
    :return:
    """
    object_subgraph = ego_graph(graph, object, radius=hop, undirected=True)
    task_subgraph = ego_graph(graph, task, radius=hop, undirected=True)
    subgraph = compose(object_subgraph, task_subgraph)
    return subgraph


class GCNTaskGrasp(data.Dataset):
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
            include_reverse_relations=True,
            subgraph_sampling=True,
            sampling_radius=2,
            instance_agnostic_mode=1):
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
                instance_agnostic_mode, subgraph_sampling, sampling_radius
        """
        super().__init__()
        if graph_data_path != '':
            graph_data_path = os.path.join(
                base_dir,
                'knowledge_graph',
                graph_data_path,
                'graph_data.pkl')
            assert os.path.exists(graph_data_path)
        self._graph_data_path = graph_data_path
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
        self._pc = {}
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
            obj, obj_class, grasp_id, task, label = parse_line(lines[i])
            obj_class = self._map_obj2class[obj]
            all_object_instances.append(obj)
            self._object_task_pairs_dataset.append("{}-{}".format(obj, task))

            pc_file = os.path.join(data_dir, obj, "fused_pc_clean.npy")
            if pc_file not in self._pc:
                if not os.path.exists(pc_file):
                    raise ValueError(
                        'Unable to find processed point cloud file {}'.format(pc_file))
                pc = np.load(pc_file)
                pc_mean = pc[:, :3].mean(axis=0)
                pc[:, :3] -= pc_mean
                self._pc[pc_file] = pc

            grasp_file = os.path.join(
                data_dir, obj, "grasps", str(grasp_id), "grasp.npy")
            if grasp_file not in self._grasps:
                grasp = np.load(grasp_file)
                self._grasps[grasp_file] = grasp

            self._data.append(
                (grasp_file, pc_file, obj, obj_class, grasp_id, task, label))
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

        # Graph loader
        self.include_reverse_relations = include_reverse_relations
        self.sampling_raidus = sampling_radius

        # data
        self.seeds = []
        self.graph = None

        self.ent2id = {}
        self.rel2id = {}
        self.id2ent = {}
        self.id2rel = {}

        self.edge_index = None
        self.edge_type = None

        # preprocess
        self.load_graph(self._graph_data_path)
        self.build_dicts()

        self.graph_idx = convert_node_labels_to_integers(self.graph)
        self.graph_num_nodes = len(list(self.graph_idx))

        self.sanity_nodes = copy.deepcopy(list(self.graph.nodes))
        assert len(self.graph_idx.nodes) == len(self.graph.nodes)
        self.node_name2idx = {
            ent: idx for idx,
            ent in enumerate(
                list(
                    self.graph.nodes))}
        self.node_idx2name = {
            idx: ent for idx,
            ent in enumerate(
                list(
                    self.graph.nodes))}
        assert len(
            self.node_idx2name.values()) == len(
            self.node_name2idx.values()) == len(
            self.graph.nodes)

    def load_graph(self, graph_data_path):
        # load a networkx graph and seeds as a list of (object_id,
        # wordnet_synset, conceptnet_name)
        with open(graph_data_path, "rb") as fh:
            self.graph, self.seeds = pickle.load(fh)

    def build_dicts(self):

        # build dictionaries
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for node in self.graph.nodes:
            ent_set.add(node)
        for edge in self.graph.edges:
            rel_set.add(self.graph[edge[0]][edge[1]]["relation"])
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        # reverse relation id is: idx+len(rel2id)
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id)
                            for idx, rel in enumerate(rel_set)})
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

    def build_adjancy_matrix(self, graph, include_reverser_relations=True):
        """
        Builds a adjancy matrix for a networkx graph.

        Args:
            graph:

        Returns:
            edge_index: adjacency matrix
            edge_type
        """
        # convert the graph into triples
        data = []
        for edge in self.graph.edges:
            rel = self.graph[edge[0]][edge[1]]["relation"]
            sub = edge[0]
            obj = edge[1]
            rel_id = self.rel2id[rel]
            sub_id = self.ent2id[sub]
            obj_id = self.ent2id[obj]
            data.append((sub_id, rel_id, obj_id))

        # build adjancy matrix
        edge_index, edge_type = [], []
        for sub, rel, obj in data:
            edge_index.append((sub, obj))
            edge_type.append(rel)
            # Important: include reverse relations
            if include_reverser_relations:
                inv_rel_id = rel + len(self.rel2id)
                edge_index.append((obj, sub))
                edge_type.append(inv_rel_id)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_type = torch.LongTensor(edge_type)

        return edge_index, edge_type

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

        grasp_file, pc_file, obj, obj_class, grasp_id, task, label = self._data[idx]
        pc = self._pc[pc_file]
        pc = regularize_pc_point_count(
            pc, self._num_points, use_farthest_point=False)
        pc_color = pc[:, 3:]
        pc = pc[:, :3]

        grasp = self._grasps[grasp_file]
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
        try:
            task_gid = self.node_name2idx[task]
        except KeyError:
            print('keyerror {}'.format(task))
            embed()

        obj_class_gid = self.node_name2idx[obj_class]

        # Sanity check to make sure mappings are correct
        assert self.sanity_nodes[obj_class_gid] == obj_class
        assert self.sanity_nodes[task_gid] == task

        instance_gid = obj_class_gid
        G_s = extract_subgraph(self.graph_idx, instance_gid, task_gid)
        edge_index = np.array(list(G_s.edges)).T

        if self.include_reverse_relations:
            edge_src = np.expand_dims(edge_index[0, :], 0)
            edge_dest = np.expand_dims(edge_index[1, :], 0)
            edge_reverse = np.concatenate([edge_dest, edge_src], axis=0)
            edge_index = np.concatenate([edge_index, edge_reverse], axis=1)

        node_x_idx = np.arange(self.graph_num_nodes)

        return pc, pc_color, task_id, task_gid, instance_gid, obj_class_gid, class_id, instance_id, grasp, node_x_idx, edge_index, label

    def __len__(self):
        return self._len

    @staticmethod
    def collate_fn(batch):
        """ This function overrides defaul batch collate function and aggregates 
        the graph and point clound data across the batch into a single graph tensor """

        pc = torch.stack([torch.as_tensor(_[0]) for _ in batch], dim=0)
        pc_color = torch.stack([torch.as_tensor(_[1]) for _ in batch], dim=0)
        task_id = torch.stack([torch.tensor(_[2]) for _ in batch], dim=0)
        task_gid = torch.stack([torch.tensor(_[3]) for _ in batch], dim=0)
        instance_gid = torch.stack([torch.tensor(_[4]) for _ in batch], dim=0)
        obj_class_gid = torch.stack([torch.tensor(_[5]) for _ in batch], dim=0)
        class_id = torch.stack([torch.tensor(_[6]) for _ in batch], dim=0)
        instance_id = torch.stack([torch.tensor(_[7]) for _ in batch], dim=0)
        grasp = torch.stack([torch.tensor(_[8]) for _ in batch], dim=0)
        node_x_idx = torch.cat([torch.tensor(_[9]) for _ in batch], dim=0)
        label = torch.stack([torch.tensor(_[11]) for _ in batch], dim=0)


        edge_indices = []
        batch_size = pc.shape[0]
        offsets = np.zeros(batch_size + 1)

        # TODO: Assumption, the sub-graphs are of the same size (i.e. passing
        # in the full graph)
        for idx, data_pt in enumerate(batch):
            node_x_idx_batch = data_pt[9]
            edge_index_batch = data_pt[10]
            edge_index_batch += int(offsets[idx])
            offsets[idx + 1] = offsets[idx] + node_x_idx_batch.shape[0]
            edge_index_batch = torch.tensor(edge_index_batch)
            edge_indices.append(edge_index_batch)
        edge_index = torch.cat(edge_indices, dim=1)

        # Add grasp nodes
        grasp_edges = []
        for idx in range(batch_size):
            node_src = obj_class_gid[idx].item() + int(offsets[idx])
            node_dest = node_x_idx.shape[0] + idx
            grasp_edges += [(node_src, node_dest), (node_dest, node_src)]
        grasp_edges = torch.tensor(np.array(grasp_edges).T)
        edge_index = torch.cat([edge_index, grasp_edges], dim=1)

        # Add latent vector
        latent = np.zeros(node_x_idx.shape[0])
        for idx, data_pt in enumerate(batch):
            task_gidx = task_gid[idx].item() + int(offsets[idx])
            latent[task_gidx] = 1

            obj_class_gidx = obj_class_gid[idx].item() + int(offsets[idx])
            latent[obj_class_gidx] = 1

        latent = torch.tensor(np.concatenate([latent, np.ones(batch_size)]))
        assert latent.shape[0] == node_x_idx.shape[0] + batch_size

        return pc, pc_color, task_id, task_gid, instance_gid, obj_class_gid, class_id, instance_id, grasp, node_x_idx, latent, edge_index, label


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
        assert args.base_dir == ''
        args.base_dir = os.path.join(os.path.dirname(__file__), '../../data')

    folder_dir = 'taskgrasp'
    _, _, _, name2wn = pickle.load(
        open(os.path.join(base_dir, folder_dir, 'misc.pkl'), 'rb'))

    dset = GCNTaskGrasp(
        4096,
        transforms=None,
        train=2,  # validation
        base_dir=base_dir,
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
        include_reverse_relations=True,
        subgraph_sampling=True,
        sampling_radius=2,
        instance_agnostic_mode=1
    )

    weights = dset.weights
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))

    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=16,
        sampler=sampler,
        collate_fn=GraspGCNDataset.collate_fn)

    with torch.no_grad():
        for batch in dloader:
            pc, pc_color, task_id, task_gid, instance_gid, obj_class_gid, class_id, instance_id, grasp, node_x_idx, latent, edge_index, label = batch
