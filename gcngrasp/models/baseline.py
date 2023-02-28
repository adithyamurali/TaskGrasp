import sys
import os
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import einops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG

from data.Dataloader import BaselineData
from data.data_specification import TASKS, TASKS_SG14K
from .layers import RelativeCrossAttentionLayer, CrossAttentionLayer, FeedforwardLayer
from .position_encodings import RotaryPositionEncoding3D


class PointNetLayers(nn.Module):
    def __init__(self, use_xyz):
        super().__init__()

        input_embedding_size = 3

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024*4,
                radii=[0.1, 0.2, 0.2],
                nsamples=[16, 32, 128],
                mlps=[
                    [input_embedding_size, 4, 8, 8],
                    [input_embedding_size, 4, 8, 8],
                    [input_embedding_size, 4, 8, 16]
                ],
                use_xyz=use_xyz,
            )
        )

        input_channels = 8 + 8 + 16

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.2, 0.2, 0.4],
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
        Arguments:
            pointcloud: b x n x 6
        """
        # TODO: colors normalize
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        features = einops.rearrange(features, "b c n -> b n c")
        return xyz, features


class AttentionLayers(nn.Module):
    def __init__(self, embedding_dim, num_attn_layers, num_attn_heads):
        super().__init__()

        self.points_attn_layers = nn.ModuleList()
        self.points_ffw_layers = nn.ModuleList()
        self.query_attn_layers = nn.ModuleList()
        self.query_ffw_layers = nn.ModuleList()
        for _ in range(num_attn_layers):
            self.points_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.points_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))
            self.query_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.query_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, point_tokens, point_pos, query_tokens, task_tokens):
        point_tokens = einops.rearrange(point_tokens, "b n c -> n b c")
        query_tokens = einops.rearrange(query_tokens, "b n c -> n b c")
        task_tokens = einops.rearrange(task_tokens, "b n c -> n b c")

        # Self-attention with relative 3D embeddings between object and grasp points
        for i in range(len(self.points_attn_layers)):
            point_tokens = self.points_attn_layers[i](
                query=point_tokens, value=point_tokens,
                query_pos=point_pos, value_pos=point_pos
            )
            point_tokens = self.points_ffw_layers[i](point_tokens)

            all_tokens = torch.cat([point_tokens, task_tokens, query_tokens], dim=0)
            all_tokens = self.query_attn_layers[i](
                query=all_tokens, value=all_tokens,
                query_pos=None, value_pos=None
            )
            all_tokens = self.query_ffw_layers[i](all_tokens)
            point_tokens = all_tokens[:-2, :, :]
            task_tokens = all_tokens[-2:-1, :, :]
            query_tokens = query_tokens[-1:, :, :]

        all_tokens = einops.rearrange(all_tokens, "n b c -> b n c")
        return all_tokens[:, :-1, :], all_tokens[:, -1:, :]


class BaselineNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.pointnet = PointNetLayers(cfg.model.use_xyz)

        self.relative_position_encoding = RotaryPositionEncoding3D(cfg.embedding_size)

        # TODO: maybe 6 not 7 points?
        self.grasp_embedding = nn.Embedding(7, self.cfg.embedding_size)
        self.query_embedding = nn.Embedding(1, self.cfg.embedding_size)

        self.attention_layers = AttentionLayers(
            embedding_dim=self.cfg.embedding_size,
            num_attn_layers=self.cfg.num_attn_layers,
            num_attn_heads=self.cfg.num_attn_heads
        )

        self.prediction_layer = nn.Sequential(
            nn.Linear(self.cfg.embedding_size, 1)
        )

        _, _, _, self.name2wn = pickle.load(open(os.path.join(self.cfg.base_dir, self.cfg.folder_dir, 'misc.pkl'),'rb'))
        self._class_list = pickle.load(open(os.path.join(self.cfg.base_dir, 'class_list.pkl'),'rb')) if self.cfg.use_class_list else list(self.name2wn.values())

        task_vocab_size = len(TASKS)
        self.task_embedding = nn.Embedding(task_vocab_size, self.cfg.embedding_size)

        # class_vocab_size = len(self._class_list)
        # self.class_embedding = nn.Embedding(class_vocab_size, self.cfg.embedding_size)

    def forward(self, pointcloud, grasp_xyz, task_ids):
        """
        Arguments:
            pointcloud: [B, N, 6] where last channel is (x,y,z,r,g,b)
            grasp_xyz: [B, 7, 3] where last channel is (x,y,z)
            tasks: id of tasks used lookup embedding dictionary
            classes: id of object classes used lookup embedding dictionary

        Return:
            logits: binary classification logits
        """
        batch_size = pointcloud.shape[0]
        pointcloud = pointcloud.float()
        grasp_xyz = grasp_xyz.float()

        grasp_tokens = self.grasp_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        grasp_pos = self.relative_position_encoding(grasp_xyz)

        object_xyz, object_tokens = self.pointnet(pointcloud)
        object_pos = self.relative_position_encoding(object_xyz)

        point_tokens = torch.cat([object_tokens, grasp_tokens], dim=1)
        point_pos = torch.cat([object_pos, grasp_pos], dim=1)

        task_tokens = self.task_embedding(task_ids).unsqueeze(1)
        query_tokens = self.query_embedding.weight.repeat(batch_size, 1, 1)

        _, query_tokens = self.attention_layers(point_tokens, point_pos, query_tokens, task_tokens)

        logits = self.prediction_layer(query_tokens[:, 0, :])
        return logits

    def training_step(self, batch, batch_idx):
        object_pcs, grasp_pcs, task_ids, _, _, labels = batch

        logits = self.forward(object_pcs, grasp_pcs, task_ids)
        logits = logits.squeeze()

        loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))

        with torch.no_grad():
            pred = torch.round(torch.sigmoid(logits))
            acc = (pred == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
        object_pcs, grasp_pcs, task_ids, _, _, labels = batch

        logits = self.forward(object_pcs, grasp_pcs, task_ids)
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

    def print_debug(self):
        #print(self.query_embedding.weight.grad)
        #print(self.grasp_embedding.weight.grad)
        #print(self.attention_layers.points_ffw_layers[0].linear1.weight.grad)
        print(self.debug_ffw.weight)
        print(self.debug_ffw.weight.grad)

    def configure_optimizers(self):
        # TODO: pruned some fancy scheduling
        print("lr:", self.cfg.optimizer.lr)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

        return optimizer

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
                map_obj2class=self.name2wn,
                class_list=self._class_list,
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
                map_obj2class=self.name2wn,
                class_list=self._class_list,
                split_mode=self.cfg.split_mode,
                split_idx=self.cfg.split_idx,
                split_version=self.cfg.split_version,
                pc_scaling=self.cfg.pc_scaling,
                use_task1_grasps=self.cfg.use_task1_grasps
            )
        else:
            raise ValueError('Invalid dataset class: {}'.format(self.cfg.dataset_class))

        if self.cfg.weighted_sampling:
            weights = self.train_dset.weights
            self._train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    def _build_dataloader(self, dset, mode):
        if self.cfg.weighted_sampling and mode == "train":
            return DataLoader(
                dset,
                batch_size=self.cfg.batch_size,
                num_workers=4,
                pin_memory=True,
                drop_last=mode == "train",
                sampler=self._train_sampler
            )
        else:
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
