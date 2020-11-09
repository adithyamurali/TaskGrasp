import sys
import os
import pickle
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric.nn import SAGEConv, BatchNorm, GCNConv
from torchvision import transforms

from data.GCNLoader import GCNTaskGrasp

from data.data_specification import TASKS
import data.data_utils as d_utils
from models.sgn import BNMomentumScheduler

class GraphNet(torch.nn.Module):
    """Class for Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, conv_type):
        super(GraphNet, self).__init__()

        if conv_type == 'GCNConv':
            print('Using GCN Conv')
            ConvLayer = GCNConv
        elif conv_type == 'SAGEConv':
            print('Using SAGE Conv')
            ConvLayer = SAGEConv
        else:
            raise NotImplementedError('Undefine graph conv type {}'.format(conv_type))

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(ConvLayer(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(ConvLayer(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(ConvLayer(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        return self.convs[-1](x, edge_index)

class GCNGrasp(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self._build_model()

    def _build_model(self):

        pc_dim = 1

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[pc_dim, 32, 32, 64], [pc_dim, 64, 64, 128], [pc_dim, 64, 96, 128]],
                use_xyz=self.cfg.model.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.cfg.model.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.cfg.model.use_xyz,
            )
        )

        _, _, _, self.name2wn = pickle.load(open(os.path.join(self.cfg.base_dir, self.cfg.folder_dir, 'misc.pkl'),'rb'))
        self._class_list = pickle.load(open(os.path.join(self.cfg.base_dir, 'class_list.pkl'),'rb')) if self.cfg.use_class_list else list(self.name2wn.values())

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.cfg.embedding_size)
        )

        self.fc_layer3 = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.gcn = GraphNet(
            in_channels=self.cfg.embedding_size+1,
            hidden_channels=128,
            out_channels=128,
            num_layers=self.cfg.gcn_num_layers,
            conv_type=self.cfg.gcn_conv_type)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, node_x_idx, latent, edge_index):
        """ Forward pass of GCNGrasp

        Args:
            pointcloud: Variable(torch.cuda.FloatTensor) [B, N, 4] tensor, 
                B is batch size, N is the number of points. The last channel is (x,y,z,feature)
            node_x_idx: [V*B,1] graph index used to lookup embedding dictionary
            latent: tensor of size [V*B + B, 1] where V is size of the graph, used to indicate goal task and classes
            edge_index: graph adjaceny matrix of size [2, E*B], where E is the number of edges in the graph

        returns:
            logits: binary classification logits
        """

        xyz, features = self._break_up_pc(pointcloud)

        for i, module in enumerate(self.SA_modules):
            xyz, features = module(xyz, features)
        shape_embedding = self.fc_layer(features.squeeze(-1))

        node_embedding = self.graph_embedding(node_x_idx)
        node_embedding = torch.cat([node_embedding, shape_embedding], dim=0)
        node_embedding = torch.cat([node_embedding, latent], dim=1)
        
        output = self.gcn(node_embedding, edge_index)
        batch_size = pointcloud.shape[0]
        output = output[-batch_size:, :]

        logits = self.fc_layer3(output)

        return logits

    def training_step(self, batch, batch_idx):
        pc, _, _, tasks, _, classes, _, _, _, node_x_idx, latent, edge_index, labels = batch

        latent = torch.unsqueeze(latent, dim=1)
        latent = latent.type(torch.cuda.FloatTensor)

        logits = self.forward(pc, node_x_idx, latent, edge_index)
        logits = logits.squeeze()

        loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))
        
        with torch.no_grad():
            pred = torch.round(torch.sigmoid(logits))
            acc = (pred == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
        pc, _, _, tasks, _, classes, _, _, _, node_x_idx, latent, edge_index, labels = batch

        latent = torch.unsqueeze(latent, dim=1)
        latent = latent.type(torch.cuda.FloatTensor)

        logits = self.forward(pc, node_x_idx, latent, edge_index)
        logits = logits.squeeze()

        try:
            loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))
        except ValueError:
            assert labels.type(torch.cuda.FloatTensor).shape[0] == 1
            logits = logits.unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))

        pred = torch.round(torch.sigmoid(logits))
        acc = (pred == labels).float().mean()

        return dict(val_loss=loss, val_acc=acc)

    def validation_end(self, outputs):
        reduced_outputs = {}
        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.cfg.optimizer.lr_decay
            ** (
                int(
                    self.global_step
                    * self.cfg.batch_size
                    / self.cfg.optimizer.decay_step
                )
            ),
            self.cfg.optimizer.lr_clip / self.cfg.optimizer.lr,
        )
        bn_lbmd = lambda _: max(
            self.cfg.optimizer.bn_momentum
            * self.cfg.optimizer.bnm_decay
            ** (
                int(
                    self.global_step
                    * self.cfg.batch_size
                    / self.cfg.optimizer.decay_step
                )
            ),
            self.cfg.optimizer.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def build_graph_embedding(self, graph, pretrained_embedding_file=''):
        """
        Creates and initializes embedding weights for tasks and class nodes in the graph.

        Args:
            graph: networkx DiGraph object
        """
        graph_size = len(list(graph.nodes))
        self.graph_embedding = nn.Embedding(graph_size, self.cfg.embedding_size)

        if pretrained_embedding_file != '':
            if not os.path.exists(pretrained_embedding_file):
                raise FileNotFoundError('Unable to locate pretrained embedding file {}'.format(pretrained_embedding_file))
            else:
                print('Loading pretrained embedding from {}'.format(pretrained_embedding_file))
            embeddings_dict = pickle.load(open(pretrained_embedding_file, 'rb'))
            embeddings = np.zeros([graph_size, self.cfg.embedding_size])
            # assert set(list(embeddings_dict.keys())) == set(list(graph.nodes))

            for i, node in enumerate(list(graph.nodes)):
                embeddings[i, :] = embeddings_dict[node]

            self.graph_embedding.weight = nn.Parameter(torch.tensor(embeddings).type(torch.cuda.FloatTensor))
        
            if self.cfg.embedding_mode == 2:
                print('Freezing embedding layer weights')
                # No fine-tuning of network
                self.graph_embedding.weight.requires_grad = False
            else:
                print('Fine-tuning network weights')
                # Fine-tuning encoder weights
                assert self.cfg.embedding_mode == 1
        else:
            # Embedding random initialization
            pass

    def prepare_data(self):
        """ Initializes datasets used for training, validation and testing """

        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudGraspToTensor(),
                d_utils.PointcloudGraspScale(),
                d_utils.PointcloudGraspRotate(axis=np.array([1.0, 0.0, 0.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspRotate(axis=np.array([0.0, 1.0, 0.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspRotate(axis=np.array([0.0, 0.0, 1.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspTranslate(),
                d_utils.PointcloudGraspJitter(),
                d_utils.PointcloudGraspRandomInputDropout(),
            ]
        )

        self.train_dset = GCNTaskGrasp(
            self.cfg.num_points,
            transforms=train_transforms,
            train=1,
            base_dir=self.cfg.base_dir,
            folder_dir=self.cfg.folder_dir,
            normal=self.cfg.model.use_normal,
            tasks=TASKS,
            map_obj2class=self.name2wn,
            class_list=self._class_list,
            split_mode=self.cfg.split_mode,
            split_idx=self.cfg.split_idx,
            split_version=self.cfg.split_version,
            pc_scaling=self.cfg.pc_scaling,
            use_task1_grasps=self.cfg.use_task1_grasps,
            graph_data_path=self.cfg.graph_data_path,
            include_reverse_relations=self.cfg.include_reverse_relations,
            subgraph_sampling=self.cfg.subgraph_sampling,
            sampling_radius=self.cfg.sampling_radius,
            instance_agnostic_mode=self.cfg.instance_agnostic_mode
        )

        pretrained_embedding_file = ''
        if self.cfg.embedding_mode != 0:
            pretrained_embedding_file = os.path.join(self.cfg.base_dir, 
            'knowledge_graph', 'embeddings', '{}_node2vec.pkl'.format(self.cfg.embedding_model))

        self.build_graph_embedding(self.train_dset.graph, pretrained_embedding_file=pretrained_embedding_file)

        if self.cfg.weighted_sampling:
            weights = self.train_dset.weights
            self._train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        self.val_dset = GCNTaskGrasp(
            self.cfg.num_points,
            transforms=train_transforms,
            train=2,
            base_dir=self.cfg.base_dir,
            folder_dir=self.cfg.folder_dir,
            normal=self.cfg.model.use_normal,
            tasks=TASKS,
            map_obj2class=self.name2wn,
            class_list=self._class_list,
            split_mode=self.cfg.split_mode,
            split_idx=self.cfg.split_idx,
            split_version=self.cfg.split_version,
            pc_scaling=self.cfg.pc_scaling,
            use_task1_grasps=self.cfg.use_task1_grasps,
            graph_data_path=self.cfg.graph_data_path,
            include_reverse_relations=self.cfg.include_reverse_relations,
            subgraph_sampling=self.cfg.subgraph_sampling,
            sampling_radius=self.cfg.sampling_radius,
            instance_agnostic_mode=self.cfg.instance_agnostic_mode
        )

    def _build_dataloader(self, dset, mode):
        if self.cfg.weighted_sampling and mode == "train":
            return DataLoader(
                dset,
                batch_size=self.cfg.batch_size,
                num_workers=4,
                pin_memory=True,
                drop_last=mode == "train",
                sampler=self._train_sampler,
                collate_fn=GCNTaskGrasp.collate_fn
            )
        else:
            return DataLoader(
                dset,
                batch_size=self.cfg.batch_size,
                shuffle=mode == "train",
                num_workers=4,
                pin_memory=True,
                drop_last=mode == "train",
                collate_fn=GCNTaskGrasp.collate_fn
            )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
