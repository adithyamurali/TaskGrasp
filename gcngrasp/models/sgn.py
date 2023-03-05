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
from data.data_specification import TASKS, TASKS_SG14K
import data.data_utils as d_utils
from utils.eval_metrics import APMetrics

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)

class SemanticGraspNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self._build_model()

        self.ap_metrics = APMetrics(cfg)

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

        task_vocab_size = len(TASKS)
        self.task_embedding = nn.Embedding(task_vocab_size, self.cfg.embedding_size)

        class_vocab_size = len(self._class_list)
        self.class_embedding = nn.Embedding(class_vocab_size, self.cfg.embedding_size)

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

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, tasks, classes):
        """ Forward pass of SGN

        Args:
            pointcloud: Variable(torch.cuda.FloatTensor) [B, N, 4] tensor, 
                B is batch size, N is the number of points. The last channel is (x,y,z,feature)
            tasks: id of tasks used lookup emebdding dictionary
            classes: id of object classes used lookup emebdding dictionary

        returns:
            logits: binary classification logits
        """

        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        shape_embedding = self.fc_layer(features.squeeze(-1))
        task_embedding = self.task_embedding(tasks)
        class_embedding = self.class_embedding(classes)

        embedding = torch.cat([shape_embedding, task_embedding, class_embedding], axis=1)

        logits = self.fc_layer2(embedding)

        return logits

    def training_step(self, batch, batch_idx):

        pc, _, tasks, classes, _, _, labels = batch

        logits = self.forward(pc, tasks, classes)
        logits = logits.squeeze()

        loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))
        
        with torch.no_grad():
            pred = torch.round(torch.sigmoid(logits))
            acc = (pred == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
 
        pc, _, tasks, classes, instances, _, labels = batch

        logits = self.forward(pc, tasks, classes)
        logits = logits.squeeze()

        try:
            loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))
        except ValueError:
            if labels.type(torch.cuda.FloatTensor).shape[0] == 1:
                logits = logits.unsqueeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.cuda.FloatTensor))

        probs = torch.sigmoid(logits)
        pred = torch.round(probs)
        acc = (pred == labels).float().mean()

        return dict(val_loss=loss, val_acc=acc,
                    labels=labels.cpu().numpy(), probs=probs.cpu().numpy(),
                    task_ids=tasks.cpu().numpy(), instance_ids=instances.cpu().numpy(),
                    class_ids=classes.cpu().numpy())

    def validation_end(self, outputs):
        reduced_outputs = {}
        for k in ["val_loss", "val_acc"]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        results = []
        for i in range(len(outputs)):
            for j in range(outputs[i]["labels"].shape[0]):
                results.append({key: outputs[i][key][j] for key in ["labels", "probs", "task_ids", "instance_ids", "class_ids"]})
        for x in results:
            x["instance_ids"] = self.val_dset.get_instance_from_id(x["instance_ids"])
            x["task_ids"] = self.val_dset.get_task_from_id(x["task_ids"])

        metrics = self.ap_metrics.get_map(results)

        reduced_outputs.update(metrics)

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

    def update_embedding_weights(self, pretrained_embedding_file):

        if not os.path.exists(pretrained_embedding_file):
            raise FileNotFoundError('Unable to locate pretrained embedding file {}'.format(pretrained_embedding_file))
        else:
            print('Loading pretrained embedding from {}'.format(pretrained_embedding_file))
        embeddings_dict = pickle.load(open(pretrained_embedding_file, 'rb'))

        # First update task embedding
        task_vocab_size = len(TASKS)
        task_embeddings = np.zeros([task_vocab_size, self.cfg.embedding_size])
        for i, task in enumerate(TASKS):
            try:
                task_embeddings[i, :] = embeddings_dict[task]
            except:
                raise ValueError('Missing key {}'.format(task))
        self.task_embedding.weight = nn.Parameter(torch.tensor(task_embeddings).type(torch.cuda.FloatTensor))

        # Second, update class embedding
        class_vocab_size = len(self._class_list)

        class_embeddings = np.zeros([class_vocab_size, self.cfg.embedding_size])

        for i, obj_class in enumerate(self._class_list):
            class_embeddings[i, :] = embeddings_dict[obj_class]
        self.class_embedding.weight = nn.Parameter(torch.tensor(class_embeddings).type(torch.cuda.FloatTensor))

        if self.cfg.embedding_mode == 2:
            print('Freezing embedding layer weights')
            # No fine-tuning of network
            self.task_embedding.weight.requires_grad = False
            self.class_embedding.weight.requires_grad = False
        else:
            print('Fine-tuning network weights')
            # Fine-tuning encoder weights
            assert self.cfg.embedding_mode == 1

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

        if self.cfg.dataset_class == 'SGNTaskGrasp':
            self.train_dset = SGNTaskGrasp(
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
                use_task1_grasps=self.cfg.use_task1_grasps
            )
        elif self.cfg.dataset_class == 'SG14K':
            self.train_dset = SG14K(
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
                use_task1_grasps=self.cfg.use_task1_grasps
            )
        else:
            raise ValueError('Invalid dataset class: {}'.format(self.cfg.dataset_class))

        if self.cfg.weighted_sampling:
            weights = self.train_dset.weights
            self._train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        if self.cfg.embedding_mode != 0:
            pretrained_embedding_file = os.path.join(self.cfg.base_dir, 
            'knowledge_graph', 'embeddings', '{}_node2vec.pkl'.format(self.cfg.embedding_model))
            self.update_embedding_weights(pretrained_embedding_file)
        else:
            assert self.cfg.embedding_mode == 0
            print('Initializing random weights for embedding layers')

        if self.cfg.dataset_class == 'SGNTaskGrasp':
            self.val_dset = SGNTaskGrasp(
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
                use_task1_grasps=self.cfg.use_task1_grasps
            )
        elif self.cfg.dataset_class == 'SG14K':
            self.val_dset = SG14K(
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
                use_task1_grasps=self.cfg.use_task1_grasps
            )
        else:
            raise ValueError('Invalid dataset class: {}'.format(self.cfg.dataset_class))

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
