import argparse
import os
import tqdm
import time
import copy
import sys
import pickle

import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from networkx import convert_node_labels_to_integers

CODE_DIR = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(CODE_DIR)

from utils.splits import get_ot_pairs_taskgrasp
from models.gcn import GCNGrasp
from data.GCNLoader import extract_subgraph
from data.SGNLoader import pc_normalize
from config import get_cfg_defaults
from geometry_utils import farthest_grasps, regularize_pc_point_count
from visualize import draw_scene, get_gripper_control_points

DEVICE = "cuda"

def load_model(cfg):
    """
    Loads GCNGrasp model from checkpoint

    Args:
        cfg: YACS config object
    Returns:
        model: GCNGrasp model 
        graph: networkx DiGraph object
    """

    _, _, _, name2wn = pickle.load(
        open(os.path.join(cfg.base_dir, cfg.folder_dir, 'misc.pkl'), 'rb'))

    graph_data_path = os.path.join(
        cfg.base_dir,
        'knowledge_graph',
        cfg.graph_data_path,
        'graph_data.pkl')

    assert os.path.exists(graph_data_path)

    model = GCNGrasp(cfg)

    with open(graph_data_path, "rb") as fh:
        graph, seeds = pickle.load(fh)

    model.build_graph_embedding(graph)
    model_weights = torch.load(
        cfg.weight_file,
        map_location=DEVICE)['state_dict']
    # This is just for backward compatibility for a deprecated model:
    if "class_embedding.weight" in model_weights:
        del model_weights["class_embedding.weight"]
    if "task_embedding.weight" in model_weights:
        del model_weights["task_embedding.weight"]
    model.load_state_dict(model_weights)
    model = model.to(DEVICE)
    model.eval()

    return model, graph


def test(model, pc, node_x_idx, latent, edge_index):

    latent = torch.unsqueeze(latent, dim=1)
    latent = latent.type(torch.cuda.FloatTensor)
    pc = pc.type(torch.cuda.FloatTensor)

    node_x_idx = node_x_idx.to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    with torch.no_grad():
        logits = model(
            pc,
            node_x_idx,
            latent,
            edge_index)
    logits = logits.squeeze()
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)
    return probs, preds


def load_pc_and_grasps(data_dir, obj_name):
    obj_dir = os.path.join(data_dir, obj_name)

    pc_file = os.path.join(obj_dir, 'fused_pc_clean.npy')
    grasps_file = os.path.join(obj_dir, 'fused_grasps_clean.npy')

    if not os.path.exists(pc_file):
        print('Unaable to find clean pc and grasps ')
        pc_file = os.path.join(obj_dir, 'fused_pc.npy')
        grasps_file = os.path.join(obj_dir, 'fused_grasps.npy')
        if not os.path.exists(pc_file):
            raise ValueError(
                'Unable to find un-processed point cloud file {}'.format(pc_file))

    pc = np.load(pc_file)
    grasps = np.load(grasps_file)

    # Ensure that grasp and pc is mean centered
    pc_mean = pc[:, :3].mean(axis=0)
    pc[:, :3] -= pc_mean
    grasps[:, :3, 3] -= pc_mean

    grasps = farthest_grasps(
        grasps, num_clusters=32, num_grasps=min(
            50, grasps.shape[0]))

    grasp_idx = 0

    pc[:, :3] += pc_mean
    grasps[:, :3, 3] += pc_mean

    return pc, grasps


def main(args, cfg):

    task = args.task
    obj_class = args.obj_class
    obj_name = args.obj_name
    data_dir = args.data_dir

    model, graph = load_model(cfg)
    graph_idx = convert_node_labels_to_integers(graph)
    graph_num_nodes = len(list(graph_idx))

    sanity_nodes = copy.deepcopy(list(graph.nodes))
    assert len(graph_idx.nodes) == len(graph.nodes)
    node_name2idx = {ent: idx for idx, ent in enumerate(list(graph.nodes))}
    node_idx2name = {idx: ent for idx, ent in enumerate(list(graph.nodes))}
    assert len(
        node_idx2name.values()) == len(
        node_name2idx.values()) == len(
            graph.nodes)

    pc, grasps = load_pc_and_grasps(data_dir, obj_name)
    pc_input = regularize_pc_point_count(
        pc, cfg.num_points, use_farthest_point=False)

    pc_mean = pc_input[:, :3].mean(axis=0)
    pc_input[:, :3] -= pc_mean
    grasps[:, :3, 3] -= pc_mean

    preds = []
    probs = []

    all_grasps_start_time = time.time()

    # TODO: Make this a batch operation instead of a loop...
    for i in tqdm.trange(len(grasps)):
        start = time.time()
        grasp = grasps[i]

        pc_color = pc_input[:, 3:]
        pc = pc_input[:, :3]

        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc.T).T
        grasp_pc = grasp_pc[:, :3]

        latent = np.concatenate(
            [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        latent = np.expand_dims(latent, axis=1)
        pc = np.concatenate([pc, grasp_pc], axis=0)

        pc, grasp = pc_normalize(pc, grasp, pc_scaling=cfg.pc_scaling)
        pc = np.concatenate([pc, latent], axis=1)

        obj_class_gid = node_name2idx[obj_class]
        task_gid = node_name2idx[task]
        assert sanity_nodes[obj_class_gid] == obj_class
        assert sanity_nodes[task_gid] == task

        instance_gid = obj_class_gid
        G_s = extract_subgraph(graph_idx, instance_gid, task_gid)
        edge_index = np.array(list(G_s.edges)).T

        if cfg.include_reverse_relations:
            edge_src = np.expand_dims(edge_index[0, :], 0)
            edge_dest = np.expand_dims(edge_index[1, :], 0)
            edge_reverse = np.concatenate([edge_dest, edge_src], axis=0)
            edge_index = np.concatenate([edge_index, edge_reverse], axis=1)

        node_x_idx = np.arange(graph_num_nodes)
        latent = np.zeros(node_x_idx.shape[0])
        latent[task_gid] = 1
        latent[obj_class_gid] = 1
        latent = np.concatenate([latent, np.ones(1)])
        assert latent.shape[0] == node_x_idx.shape[0] + 1

        pc = torch.tensor([pc])
        node_x_idx = torch.tensor(node_x_idx)
        edge_index = torch.tensor(edge_index)
        latent = torch.tensor(latent)

        prob, pred = test(model, pc, node_x_idx, latent, edge_index)

        preds.append(pred.tolist())
        probs.append(prob.tolist())

    print('Inference took {}s for {} grasps'.format(time.time() - all_grasps_start_time, len(grasps)))
    preds = np.array(preds)
    probs = np.array(probs)

    grasp_colors = np.stack([np.ones(probs.shape[0]) -
                             probs, probs, np.zeros(probs.shape[0])], axis=1)
    draw_scene(
        pc_input,
        grasps,
        grasp_colors=list(grasp_colors),
        max_grasps=len(grasps))

    best_grasp = copy.deepcopy(grasps[np.argmax(probs)])
    draw_scene(pc_input, np.expand_dims(best_grasp, axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visualize data and stuff")
    parser.add_argument('--task', help='', default='')
    parser.add_argument('--obj_class', help='', default='')
    parser.add_argument('--data_dir', help='location of sample data', default='')
    parser.add_argument('--obj_name', help='', default='')
    parser.add_argument(
        'cfg_file',
        help='yaml file in YACS config format to override default configs',
        default='',
        type=str)

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file != '':
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)
        else:
            raise ValueError('Please provide a valid config file for the --cfg_file arg')

    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    cfg.batch_size = 16

    if len(cfg.gpus) == 1:
        torch.cuda.set_device(cfg.gpus[0])

    experiment_dir = os.path.join(cfg.log_dir, cfg.weight_file)

    weight_files = os.listdir(os.path.join(experiment_dir, 'weights'))
    assert len(weight_files) == 1
    cfg.weight_file = os.path.join(experiment_dir, 'weights', weight_files[0])

    if args.data_dir == '':
        args.data_dir = os.path.join(cfg.base_dir, 'sample_data')
    assert os.path.exists(args.data_dir)

    cfg.freeze()
    print(cfg)

    main(args, cfg)
