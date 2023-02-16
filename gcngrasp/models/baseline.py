"""
Adapted from sgn.py
"""
import os
import pickle
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from data.SGNLoader import SGNTaskGrasp
from data.SG14KLoader import SG14K
from data.Dataloader import BaselineData
from data.data_specification import TASKS, TASKS_SG14K
import data.data_utils as d_utils

class PointNetLayers(nn.Module):
    def __init__(self):
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

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    # TODO: do we want to process rgb?
    def forward(self, pointcloud):
        """
        input: b x n x 6
        """
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return xyz, features

class AttentionLayers(nn.Module):
    def __init__(self):
        pass

    def forward(self, object_tokens, grasp_tokens, query_token):
        return object_tokens, grasp_tokens, query_token

class BaselineNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.pointnet = PointNetLayers()

        self.grasp_embedding = nn.Embedding(6, self.cfg.embedding_size)
        self.query_embedding = nn.Embedding(1, self.cfg.embedding_size)

        self.attention_layers = AttentionLayers()

        self.prediction_layer = nn.Sequential(
            nn.Linear(self.cfg.embeddding_size, 1)
        )

        #_, _, _, self.name2wn = pickle.load(open(os.path.join(self.cfg.base_dir, self.cfg.folder_dir, 'misc.pkl'),'rb'))
        #self._class_list = pickle.load(open(os.path.join(self.cfg.base_dir, 'class_list.pkl'),'rb')) if self.cfg.use_class_list else list(self.name2wn.values())

        #task_vocab_size = len(TASKS)
        #self.task_embedding = nn.Embedding(task_vocab_size, self.cfg.embedding_size)

        #class_vocab_size = len(self._class_list)
        #self.class_embedding = nn.Embedding(class_vocab_size, self.cfg.embedding_size)

        """
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.cfg.embedding_size)
        )

        embeddding_size = self.cfg.embedding_size*3

        self.fc_layer2 = nn.Sequential(
            nn.Linear(embeddding_size, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        """

    def forward(self, pointcloud, grasp_pc):
        """ Forward pass of SGN

        Args:
            pointcloud: Variable(torch.cuda.FloatTensor) [B, N, 6] tensor, 
                B is batch size, N is the number of points. The last channel is (x,y,z,feature)
            grasp_pc: [B, 6, 3]
            tasks: id of tasks used lookup emebdding dictionary
            classes: id of object classes used lookup emebdding dictionary

        returns:
            logits: binary classification logits
        """

        xyz, features = self.pointnet(pointcloud)

        object_tokens = self.pc_tokens(xyz, features)
        grasp_tokens = self.pc_tokens(grasp_pc[..., :3], torch.unsqueeze(self.grasp_embedding, 0))

        object_tokens, grasp_tokens, query_token = self.attention_layers(object_tokens, grasp_tokens, self.query_embedding)

        #shape_embedding = self.fc_layer(features.squeeze(-1))
        #logits = self.fc_layer2(embedding)

        logits = self.prediction_layer(query_token)

        return logits

    def training_step(self, batch, batch_idx):

        object_pcs, grasp_pcs, labels = batch

        logits = self.forward(object_pcs, grasp_pcs)
        logits = logits.squeeze()

        loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))
        
        with torch.no_grad():
            pred = torch.round(torch.sigmoid(logits))
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

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

        return optimizer

    # TODO pruned update_embedding_weights()

    def prepare_data(self):
        """ Initializes datasets used for training, validation and testing """

        # TODO: remove data aug
        if self.cfg.dataset_class == 'SGNTaskGrasp':
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

        if self.cfg.dataset_class == 'SGNTaskGrasp':
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
