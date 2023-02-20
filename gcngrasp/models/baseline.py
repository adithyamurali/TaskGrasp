"""
Adapted from gcn.py
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

import pickle
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from data.SGNLoader import SGNTaskGrasp
from data.SG14KLoader import SG14K
from data.Dataloader import BaselineData
from data.data_specification import TASKS, TASKS_SG14K
import data.data_utils as d_utils
from models.position_encodings import RotaryPositionEncoding3D

from .layers import RelativeCrossAttentionLayer, CrossAttentionLayer, FeedforwardLayer

import einops

class PointNetLayers(nn.Module):
    def __init__(self, use_xyz):
        super().__init__()

        input_embedding_size = 3

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024*4,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[input_embedding_size, 4, 8, 8], [input_embedding_size, 4, 8, 8], [input_embedding_size, 4, 8, 16]],
                use_xyz=use_xyz,
            )
        )

        input_channels = 8 + 8 + 16

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 32, 16, 8],
                    [input_channels, 32, 16, 8],
                    [input_channels, 32, 16, 16],
                ],
                use_xyz=use_xyz,
            )
        )

        input_channels = 8 + 8 + 16

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 32, 32, 20],
                    [input_channels, 32, 32, 20],
                    [input_channels, 32, 32, 20],
                ],
                use_xyz=use_xyz,
            )
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    # TODO: do we want to process rgb?
    def forward(self, pointcloud):
        """
        input: b x n x 6
        """
        # TODO: colors normalize
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            #print(xyz, features)
            xyz, features = module(xyz, features)
        features = einops.rearrange(features, "b c n -> b n c")
        return xyz, features

class AttentionLayers(nn.Module):
    def __init__(self, embedding_dim, num_attn_layers, num_attn_heads):
        super().__init__()
        self.points_self_attn_layers = nn.ModuleList()
        self.points_ffw_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.query_ffw_layers = nn.ModuleList()
        for _ in range(num_attn_layers):
            self.points_self_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            #self.points_self_attn_layers.append(CrossAttentionLayer(embedding_dim, num_attn_heads))
            self.points_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))
            self.cross_attn_layers.append(CrossAttentionLayer(embedding_dim, num_attn_heads))
            self.query_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, point_tokens, point_pos, query_tokens):
        #print("got:", point_tokens.shape, point_pos.shape, query_tokens.shape)
        for i in range(len(self.cross_attn_layers)):
            new_point_tokens = self.points_self_attn_layers[i](
                query=point_tokens, value=point_tokens,
                query_pos=point_pos, value_pos=point_pos
            )
            new_query_tokens = self.cross_attn_layers[i](
                query=query_tokens, value=point_tokens,
                query_pos=None, value_pos=point_pos
            )
            point_tokens = self.points_ffw_layers[i](new_point_tokens)
            query_tokens = self.points_ffw_layers[i](new_query_tokens)
        return point_tokens, query_tokens

class BaselineNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        print("constructing baseline")

        self.cfg = cfg

        #point_features_size = 32
        self.pointnet = PointNetLayers(cfg.model.use_xyz)
        self.position_encoding = RotaryPositionEncoding3D(cfg.embedding_size)

        # TODO: maybe 6 not 7 points?
        self.grasp_embedding = nn.Embedding(7, self.cfg.embedding_size)
        self.query_embedding = nn.Embedding(1, self.cfg.embedding_size)

        self.attention_layers = AttentionLayers(embedding_dim=self.cfg.embedding_size, num_attn_layers=self.cfg.num_attn_layers, num_attn_heads=self.cfg.num_attn_heads)

        self.prediction_layer = nn.Sequential(
            nn.Linear(self.cfg.embedding_size, 1)
        )

        #_, _, _, self.name2wn = pickle.load(open(os.path.join(self.cfg.base_dir, self.cfg.folder_dir, 'misc.pkl'),'rb'))
        #self._class_list = pickle.load(open(os.path.join(self.cfg.base_dir, 'class_list.pkl'),'rb')) if self.cfg.use_class_list else list(self.name2wn.values())

        #task_vocab_size = len(TASKS)
        #self.task_embedding = nn.Embedding(task_vocab_size, self.cfg.embedding_size)

        #class_vocab_size = len(self._class_list)
        #self.class_embedding = nn.Embedding(class_vocab_size, self.cfg.embedding_size)

    def print_debug(self):
        print(self.query_embedding.weight.grad)
        print(self.grasp_embedding.weight.grad)
        print(self.attention_layers.points_ffw_layers[0].linear1.weight.grad)

    def pe_tokens(self, positions, features):
        #print("got", positions.shape, features.shape)
        pos_embed = self.position_encoding(positions)
        #print(pos_embed.shape)
        # TODO: fix?!
        #pos_embed = einops.rearrange(pos_embed, "b n x y -> b n (x y)")
        #print(pos_embed.shape)
        #res = torch.cat([pos_embed, features], dim=-1)
        return pos_embed

    def forward(self, pointcloud, grasp_pc):
        """ Forward pass of SGN

        Args:
            pointcloud: Variable(torch.cuda.FloatTensor) [B, N, 6] tensor, 
                B is batch size, N is the number of points. The last channel is (x,y,z,feature)
            grasp_pc: [B, 7, 3]
            tasks: id of tasks used lookup emebdding dictionary
            classes: id of object classes used lookup emebdding dictionary

        returns:
            logits: binary classification logits
        """
        #print(pointcloud[:, :5, :])
        batch_size = pointcloud.shape[0]

        #print("input features", pointcloud.shape, grasp_pc.shape)
        pointcloud = pointcloud.float()
        grasp_pc = grasp_pc.float()
        xyz, object_tokens = self.pointnet(pointcloud)
        #print(xyz.shape, object_tokens.shape)

        grasp_tokens = self.grasp_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        object_pos = self.pe_tokens(xyz, object_tokens)
        grasp_pos = self.pe_tokens(grasp_pc[..., :3], grasp_tokens)
        point_tokens = torch.cat([object_tokens, grasp_tokens], dim=1)
        point_pos = torch.cat([object_pos, grasp_pos], dim=1)
        
        query_tokens = self.query_embedding.weight.repeat(1, batch_size, 1)
        #point_tokens = einops.rearrange(point_tokens, "b n c -> n b c")
        point_pos = einops.rearrange(point_pos, "b n c x -> n b c x")
        #point_pos = einops.rearrange(point_pos, "b n c -> n b c")

        #print("embeddings:", object_pos.shape, object_tokens.shape, grasp_pos.shape, grasp_tokens.shape, query_tokens.shape)
        #print("embeddings:", point_tokens.shape, point_pos.shape, query_tokens.shape)

        #object_tokens, grasp_tokens, query_token = self.attention_layers(object_tokens, grasp_tokens, query_token)
        point_tokens, query_tokens = self.attention_layers(point_tokens, point_pos, query_tokens)

        #shape_embedding = self.fc_layer(features.squeeze(-1))
        #logits = self.fc_layer2(embedding)

        #print("tokens result shape:", point_tokens.shape, query_tokens.shape)

        logits = self.prediction_layer(query_tokens[0])
        #print("logits:", logits.shape)

        return logits

    def training_step(self, batch, batch_idx):

        object_pcs, grasp_pcs, labels = batch

        logits = self.forward(object_pcs, grasp_pcs)
        logits = logits.squeeze()

        loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))
        
        with torch.no_grad():
            pred = torch.round(torch.sigmoid(logits))
            #print("predictions:", pred.sum(), pred.shape[0]-pred.sum(), labels.sum())
            acc = (pred == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
 
        object_pcs, grasp_pcs, labels = batch

        logits = self.forward(object_pcs, grasp_pcs)
        logits = logits.squeeze()

        #try:
        loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))
        #except ValueError:
        #    if labels.type(torch.cuda.FloatTensor).shape[0] == 1:
        #        logits = logits.unsqueeze(-1)
        #        loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))
        pred = torch.round(torch.sigmoid(logits))
        # TODO: check correct shape, not additional trailing 1-dim
        acc = (pred == labels).float().mean()

        return dict(val_loss=loss, val_acc=acc)

    def validation_end(self, outputs):
        reduced_outputs = {}
        for k in outputs[0]:
            reduced_outputs[k] = [o[k] for o in outputs]
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    def configure_optimizers(self):
        # TODO: pruned some fancy scheduling
        print("lr:", self.cfg.optimizer.lr)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

        return optimizer

    # TODO pruned update_embedding_weights()

    def prepare_data(self):
        """ Initializes datasets used for training, validation and testing """

        # TODO: I removed data aug

        if self.cfg.dataset_class == 'BaselineData':
            self.train_dset = BaselineData(
                self.cfg.num_points,
                #transforms=train_transforms,
                train=1,
                base_dir=self.cfg.base_dir,
                folder_dir=self.cfg.folder_dir,
                normal=self.cfg.model.use_normal,
                tasks=TASKS,
                #map_obj2class=self.name2wn,
                #class_list=self._class_list,
                split_mode=self.cfg.split_mode,
                split_idx=self.cfg.split_idx,
                split_version=self.cfg.split_version,
                pc_scaling=self.cfg.pc_scaling,
                use_task1_grasps=self.cfg.use_task1_grasps
            )
        else:
            raise ValueError('Invalid dataset class: {}'.format(self.cfg.dataset_class))

        if self.cfg.dataset_class == 'BaselineData':
            self.val_dset = BaselineData(
                self.cfg.num_points,
                #transforms=train_transforms,
                train=2,
                base_dir=self.cfg.base_dir,
                folder_dir=self.cfg.folder_dir,
                normal=self.cfg.model.use_normal,
                tasks=TASKS,
                #map_obj2class=self.name2wn,
                #class_list=self._class_list,
                split_mode=self.cfg.split_mode,
                split_idx=self.cfg.split_idx,
                split_version=self.cfg.split_version,
                pc_scaling=self.cfg.pc_scaling,
                use_task1_grasps=self.cfg.use_task1_grasps
            )
        else:
            raise ValueError('Invalid dataset class: {}'.format(self.cfg.dataset_class))

    def _build_dataloader(self, dset, mode):
        # TODO: removed weighted sampling
        return DataLoader(
            dset,
            batch_size=self.cfg.batch_size,
            shuffle=mode == "train",
            num_workers=4,
            pin_memory=True,
            drop_last=mode == "train"
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
