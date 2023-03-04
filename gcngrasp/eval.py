import argparse
import os
import sys
import pickle
#import omegaconf
import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from collections import defaultdict
import matplotlib.pyplot as plt

from config import get_cfg_defaults
import torch.nn.functional as F

from models.sgn import SemanticGraspNet
from models.gcn import GCNGrasp
from models.baseline import BaselineNet
from data.SGNLoader import SGNTaskGrasp
from data.GCNLoader import GCNTaskGrasp
from data.SG14KLoader import SG14K, get_ot_pairs_sg14k
from data.data_specification import TASKS, TASKS_SG14K
from utils.splits import get_ot_pairs_taskgrasp
from data.Dataloader import BaselineData

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../'))
from visualize import draw_scene, mkdir

DEVICE = "cuda"

def visualize_batch(pc, grasps):
    """ Visualizes all the data in the batch, just for debugging """

    for i in range(pc.shape[0]):
        pcc = pc[i, :, :3]
        grasp = grasps[i, :, :]
        draw_scene(pcc, [grasp, ])


def visualize_batch_wrong(pc, grasps, labels, preds):
    """ Visualizes incorrect predictions """

    for i in range(pc.shape[0]):
        if labels[i] != preds[i]:
            print('labels {}, prediction {}'.format(labels[i], preds[i]))
            pcc = pc[i, :, :3]
            grasp = grasps[i, :, :]
            draw_scene(pcc, [grasp, ])

def main(cfg, save=False, visualize=False, experiment_dir=None):

    _, _, _, name2wn = pickle.load(
        open(os.path.join(cfg.base_dir, cfg.folder_dir, 'misc.pkl'), 'rb'))
    class_list = pickle.load(
        open(
            os.path.join(
                cfg.base_dir,
                'class_list.pkl'),
            'rb')) if cfg.use_class_list else list(
        name2wn.values())

    if cfg.dataset_class == 'SGNTaskGrasp':
        dset = SGNTaskGrasp(
            cfg.num_points,
            transforms=None,
            train=0,
            base_dir=cfg.base_dir,
            folder_dir=cfg.folder_dir,
            normal=cfg.model.use_normal,
            tasks=TASKS,
            map_obj2class=name2wn,
            class_list=class_list,
            split_mode=cfg.split_mode,
            split_idx=cfg.split_idx,
            split_version=cfg.split_version,
            pc_scaling=cfg.pc_scaling,
            use_task1_grasps=cfg.use_task1_grasps
        )
    elif cfg.dataset_class == 'GCNTaskGrasp':
        dset = GCNTaskGrasp(
            cfg.num_points,
            transforms=None,
            train=0,
            base_dir=cfg.base_dir,
            folder_dir=cfg.folder_dir,
            normal=cfg.model.use_normal,
            tasks=TASKS,
            map_obj2class=name2wn,
            class_list=class_list,
            split_mode=cfg.split_mode,
            split_idx=cfg.split_idx,
            split_version=cfg.split_version,
            pc_scaling=cfg.pc_scaling,
            use_task1_grasps=cfg.use_task1_grasps,
            graph_data_path=cfg.graph_data_path,
            include_reverse_relations=cfg.include_reverse_relations,
            subgraph_sampling=cfg.subgraph_sampling,
            sampling_radius=cfg.sampling_radius,
            instance_agnostic_mode=cfg.instance_agnostic_mode
        )
    elif cfg.dataset_class == 'SG14K':
        dset = SG14K(
            cfg.num_points,
            transforms=None,
            train=0,
            base_dir=cfg.base_dir,
            folder_dir=cfg.folder_dir,
            normal=cfg.model.use_normal,
            tasks=TASKS,
            map_obj2class=name2wn,
            class_list=class_list,
            split_mode=cfg.split_mode,
            split_idx=cfg.split_idx,
            split_version=cfg.split_version,
            pc_scaling=cfg.pc_scaling,
            use_task1_grasps=cfg.use_task1_grasps
        )
    elif cfg.dataset_class == 'BaselineData':
        dset = BaselineData(
            cfg.num_points,
            transforms=None,
            train=0,
            base_dir=cfg.base_dir,
            folder_dir=cfg.folder_dir,
            normal=cfg.model.use_normal,
            tasks=TASKS,
            map_obj2class=name2wn,
            class_list=class_list,
            split_mode=cfg.split_mode,
            split_idx=cfg.split_idx,
            split_version=cfg.split_version,
            pc_scaling=cfg.pc_scaling,
            use_task1_grasps=cfg.use_task1_grasps,
            graph_data_path=cfg.graph_data_path,
            include_reverse_relations=cfg.include_reverse_relations
        )

    if cfg.algorithm_class == 'SemanticGraspNet':
        model = SemanticGraspNet(cfg)
    elif cfg.algorithm_class == 'GCNGrasp':
        model = GCNGrasp(cfg)
        model.build_graph_embedding(dset.graph)
    elif cfg.algorithm_class == 'Baseline':
        model = BaselineNet(cfg)
    else:
        raise ValueError('Unknown class name {}'.format(cfg.algorithm_class))

    assert model._class_list == class_list
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

    if cfg.algorithm_class == 'GCNGrasp':
        dloader = torch.utils.data.DataLoader(
            dset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=GCNTaskGrasp.collate_fn)
    else:
        dloader = torch.utils.data.DataLoader(
            dset, batch_size=cfg.batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []
    all_data_vis = {}
    all_data_pc = {}

    # Our data annotation on MTurk happens in 2 stages, see paper for more details

    # Only considering Stage 2 grasps
    task1_results_file = os.path.join(
        cfg.base_dir, cfg.folder_dir, 'task1_results.txt')
    assert os.path.exists(task1_results_file)

    if cfg.dataset_class in ['SGNTaskGrasp', 'GCNTaskGrasp', 'BaselineData']:
        object_task_pairs = get_ot_pairs_taskgrasp(task1_results_file)
        TASK2_ot_pairs = object_task_pairs['True'] + \
            object_task_pairs['Weak True']
        TASK1_ot_pairs = object_task_pairs['False'] + \
            object_task_pairs['Weak False']
    elif cfg.dataset_class in ['SG14K']:
        object_task_pairs = get_ot_pairs_sg14k(task1_results_file)
        TASK2_ot_pairs = object_task_pairs['True'] + object_task_pairs['False']
        TASK1_ot_pairs = []
    else:
        raise ValueError('Unknown class {}'.format(cfg.dataset_class))

    all_preds_2 = []
    all_probs_2 = []
    all_labels_2 = []

    all_preds_2_v2 = defaultdict(dict)
    all_probs_2_v2 = defaultdict(dict)
    all_labels_2_v2 = defaultdict(dict)

    # Only considering Stage 1 grasps
    all_preds_1 = []
    all_probs_1 = []
    all_labels_1 = []

    print('Running evaluation on Test set')
    with torch.no_grad():
        for batch in tqdm(dloader):

            if cfg.algorithm_class == 'SemanticGraspNet':
                pc, pc_color, tasks, classes, instances, grasps, labels = batch

                pc = pc.type(torch.cuda.FloatTensor)
                tasks = tasks.to(DEVICE)
                classes = classes.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(pc, tasks, classes)
                logits = logits.squeeze()

            elif cfg.algorithm_class == 'GCNGrasp':

                pc, pc_color, tasks, tasks_gid, instances_gid, obj_class_gid, classes, instances, grasps, node_x_idx, latent, edge_index, labels = batch

                latent = torch.unsqueeze(latent, dim=1)
                latent = latent.type(torch.cuda.FloatTensor)
                pc = pc.type(torch.cuda.FloatTensor)
                tasks = tasks.to(DEVICE)
                classes = classes.to(DEVICE)
                labels = labels.to(DEVICE)
                node_x_idx = node_x_idx.to(DEVICE)
                edge_index = edge_index.to(DEVICE)
                tasks_gid = tasks_gid.to(DEVICE)
                obj_class_gid = obj_class_gid.to(DEVICE)

                logits = model(
                    pc,
                    node_x_idx,
                    latent,
                    edge_index)
                logits = logits.squeeze()

            elif cfg.algorithm_class == 'Baseline':
                object_pcs, grasp_pcs, tasks, instances, classes, grasps, labels = batch

                object_pcs = object_pcs.type(torch.cuda.FloatTensor)
                grasp_pcs = grasp_pcs.to(DEVICE)
                tasks = tasks.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model.forward(object_pcs, grasp_pcs, tasks)
                logits = logits.squeeze()

                pc = object_pcs[:, :, :3]
                pc_color = object_pcs[:, :, 3:]

            probs = torch.sigmoid(logits)
            preds = torch.round(probs)

            try:
                preds = preds.cpu().numpy()
                probs = probs.cpu().numpy()
                labels = labels.cpu().numpy()

                all_preds += list(preds)
                all_probs += list(probs)
                all_labels += list(labels)
            except TypeError:
                all_preds.append(preds.tolist())
                all_probs.append(probs.tolist())
                all_labels.append(labels.tolist()[0])

            tasks = tasks.cpu().numpy()
            instances = instances.cpu().numpy()
            for i in range(tasks.shape[0]):
                task = tasks[i]
                task = TASKS[task]
                instance_id = instances[i]
                obj_instance_name = dset._all_object_instances[instance_id]
                ot = "{}-{}".format(obj_instance_name, task)

                try:
                    pred = preds[i]
                    prob = probs[i]
                    label = labels[i]
                except IndexError:
                    # TODO: This is very hacky, fix it
                    pred = preds.tolist()
                    prob = probs.tolist()
                    label = labels.tolist()[0]

                if ot in TASK2_ot_pairs:
                    all_preds_2.append(pred)
                    all_probs_2.append(prob)
                    all_labels_2.append(label)

                    try:
                        all_preds_2_v2[obj_instance_name][task].append(pred)
                        all_probs_2_v2[obj_instance_name][task].append(prob)
                        all_labels_2_v2[obj_instance_name][task].append(label)

                    except KeyError:
                        all_preds_2_v2[obj_instance_name][task] = [pred, ]
                        all_probs_2_v2[obj_instance_name][task] = [prob, ]
                        all_labels_2_v2[obj_instance_name][task] = [label, ]

                elif ot in TASK1_ot_pairs:
                    all_preds_1.append(pred)
                    all_probs_1.append(prob)
                    all_labels_1.append(label)
                elif ot in ROUND1_GOLD_STANDARD_PROTOTYPICAL_USE:
                    all_preds_2.append(pred)
                    all_probs_2.append(prob)
                    all_labels_2.append(label)

                    try:
                        all_preds_2_v2[obj_instance_name][task].append(pred)
                        all_probs_2_v2[obj_instance_name][task].append(prob)
                        all_labels_2_v2[obj_instance_name][task].append(label)

                    except KeyError:
                        all_preds_2_v2[obj_instance_name][task] = [pred, ]
                        all_probs_2_v2[obj_instance_name][task] = [prob, ]
                        all_labels_2_v2[obj_instance_name][task] = [label, ]

                else:
                    raise Exception('Unknown ot {}'.format(ot))

            if visualize or save:

                pc = pc.cpu().numpy()
                grasps = grasps.cpu().numpy()
                classes = classes.cpu().numpy()
                pc_color = pc_color.cpu().numpy()

                # Uncomment the following for debugging
                # visualize_batch(pc, grasps)
                # visualize_batch_wrong(pc, grasps, labels, preds)

                for i in range(pc.shape[0]):
                    pc_i = pc[i, :, :]
                    if cfg.dataset_class != 'BaselineData':
                        pc_i = pc_i[np.where(pc_i[:, 3] == 0), :3].squeeze(0)
                    pc_color_i = pc_color[i, :, :3]
                    pc_i = np.concatenate([pc_i, pc_color_i], axis=1)
                    grasp = grasps[i, :, :]
                    task = tasks[i]
                    task = TASKS[task]
                    instance_id = instances[i]
                    obj_instance_name = dset._all_object_instances[instance_id]
                    obj_class = classes[i]
                    obj_class = class_list[obj_class]

                    try:
                        pred = preds[i]
                        prob = probs[i]
                        label = labels[i]
                    except IndexError:
                        pred = preds.tolist()
                        prob = probs.tolist()
                        label = labels.tolist()[0]

                    ot = "{}-{}".format(obj_instance_name, task)
                    grasp_datapt = (grasp, prob, pred, label)
                    if ot in all_data_vis:
                        all_data_vis[ot].append(grasp_datapt)
                        all_data_pc[ot] = pc_i
                    else:
                        all_data_vis[ot] = [grasp_datapt, ]
                        all_data_pc[ot] = pc_i

    # Stage 1+2 grasps
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    random_probs = np.random.uniform(low=0, high=1, size=(len(all_probs)))
    results = {
        'preds': all_preds,
        'probs': all_probs,
        'labels': all_labels,
        'random': random_probs}

    # Only Stage 2 grasps
    all_preds_2 = np.array(all_preds_2)
    all_probs_2 = np.array(all_probs_2)
    all_labels_2 = np.array(all_labels_2)
    random_probs_2 = np.random.uniform(low=0, high=1, size=(len(all_probs_2)))
    results_2 = {
        'preds': all_preds_2,
        'probs': all_probs_2,
        'labels': all_labels_2,
        'random': random_probs_2}

    # Only Stage 1 grasps
    all_preds_1 = np.array(all_preds_1)
    all_probs_1 = np.array(all_probs_1)
    all_labels_1 = np.array(all_labels_1)
    random_probs_1 = np.random.uniform(low=0, high=1, size=(len(all_probs_1)))
    results_1 = {
        'preds': all_preds_1,
        'probs': all_probs_1,
        'labels': all_labels_1,
        'random': random_probs_1}

    # Only Stage 2 grasps
    random_probs_2 = np.random.uniform(low=0, high=1, size=(len(all_probs_2)))
    results_2_v2 = {
        'preds': all_preds_2_v2,
        'probs': all_probs_2_v2,
        'labels': all_labels_2_v2,
        'random': random_probs_2}

    if save:
        mkdir(os.path.join(experiment_dir, 'results'))
        pickle.dump(
            results,
            open(
                os.path.join(
                    experiment_dir,
                    'results',
                    "results.pkl"),
                'wb'))

        mkdir(os.path.join(experiment_dir, 'results1'))
        pickle.dump(
            results_1,
            open(
                os.path.join(
                    experiment_dir,
                    'results1',
                    "results.pkl"),
                'wb'))

        mkdir(os.path.join(experiment_dir, 'results2'))
        pickle.dump(
            results_2,
            open(
                os.path.join(
                    experiment_dir,
                    'results2',
                    "results.pkl"),
                'wb'))

    if save or visualize:
        mkdir(os.path.join(experiment_dir, 'results2_ap'))
        pickle.dump(
            results_2_v2,
            open(
                os.path.join(
                    experiment_dir,
                    'results2_ap',
                    "results.pkl"),
                'wb'))

        # TODO - Write separate script for loading and visualizing predictions
        # mkdir(os.path.join(experiment_dir, 'visualization_data'))
        # pickle.dump(
        #     all_data_vis,
        #     open(
        #         os.path.join(
        #             experiment_dir,
        #             'visualization_data',
        #             "predictions.pkl"),
        #         'wb'))

    if visualize:

        mkdir(os.path.join(experiment_dir, 'visualization'))
        mkdir(os.path.join(experiment_dir, 'visualization', 'task1'))
        mkdir(os.path.join(experiment_dir, 'visualization', 'task2'))

        print('saving ot visualizations')
        for ot in all_data_vis.keys():

            if ot in TASK1_ot_pairs:
                save_dir = os.path.join(
                    experiment_dir, 'visualization', 'task1')
            elif ot in TASK2_ot_pairs:
                save_dir = os.path.join(
                    experiment_dir, 'visualization', 'task2')
            else:
                continue

            pc = all_data_pc[ot]
            grasps_ot = all_data_vis[ot]
            grasps = [elem[0] for elem in grasps_ot]
            probs = np.array([elem[1] for elem in grasps_ot])
            preds = np.array([elem[2] for elem in grasps_ot])
            labels = np.array([elem[3] for elem in grasps_ot])

            grasp_colors = np.stack(
                [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_colors=list(grasp_colors), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_gt.png'.format(ot)))

            grasp_colors = np.stack(
                [np.ones(preds.shape[0]) - preds, preds, np.zeros(preds.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_colors=list(grasp_colors), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_pred.png'.format(ot)))

            grasp_colors = np.stack(
                [np.ones(probs.shape[0]) - probs, probs, np.zeros(probs.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_colors=list(grasp_colors), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_probs.png'.format(ot)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument(
        'cfg_file',
        help='yaml file in YACS config format to override default configs',
        default='',
        type=str)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file != '':
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)

    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    cfg.batch_size = args.batch_size
    if args.gpus == -1:
        args.gpus = [0, ]
    cfg.gpus = args.gpus

    experiment_dir = os.path.join(cfg.log_dir, cfg.weight_file)

    weight_files = os.listdir(os.path.join(experiment_dir, 'weights'))
    weight_files = list(sorted(weight_files))
    print(weight_files)
    #assert len(weight_files) == 1
    print("using", weight_files[-1])
    cfg.weight_file = os.path.join(experiment_dir, 'weights', weight_files[-1])

    cfg.freeze()
    print(cfg)
    main(
        cfg,
        save=args.save,
        visualize=args.visualize,
        experiment_dir=experiment_dir)
