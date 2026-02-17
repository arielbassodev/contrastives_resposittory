import torch.nn as nn
import torch.nn.functional as F
import torch
import lightning as L
from typing import  Literal
from lightly.loss import NTXentLoss
import torchvision
from torch.nn import MSELoss

from augmentation import global_global_augmentation
from utils import ContrastiveApproachType, OptimizerType
from app_logger import logger
from utils import BackBonesType
from typing import Any
import itertools
from pytorch_metric_learning.losses import SupConLoss

# Projection head for the different contrastive learning approaches
class ProjectionClass(nn.Module):

    def __init__(self, in_feature: int,hidden_feature: int, out_feature: int):
        super().__init__()
        self.in_feature     = in_feature
        self.out_feature    = out_feature       
        self.hidden_feature = hidden_feature
        self.fc1 = nn.Linear(self.in_feature, self.hidden_feature)
        self.fc2 = nn.Linear(self.hidden_feature, self.out_feature)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class CLRModel(nn.Module):
    def __init__(self, backbone, projection_head):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        x = self.projection_head(x)
        return x
# # SimCLR a contrastive learning approaches that leverages positives and negatives pairs
#
# class SimCLR(nn.Module):
#
#     def __init__(self, Backbone, Projection):
#         super().__init__()
#         self.backbone = Backbone
#         self.head_projection = Projection
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = x.flatten(start_dim=1)
#         x = self.head_projection(x)
#         return x
    
# # SLFPN another self-supervised representation learning approaches using only positive pairs, the main difference is during the training
#
# class SLFPN(nn.Module):
#
#     def __init__(self, Backbone, Projection_head):
#         super().__init__()
#         self.backbone = Backbone
#         self.projection_head = Projection_head
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = x.flatten(start_dim=1)
#         x = self.head_projection(x)
#         return x


def build_model(backbone_name: BackBonesType, projector_hidden_feature=1024, projector_out_feature=128) -> Any:
    logger.info({"Chosen backbone": backbone_name})
    model_weights = torchvision.models.get_model_weights(backbone_name).DEFAULT
    backbone = torchvision.models.get_model(name=backbone_name, weights=model_weights)
    projection_head = None
    match backbone_name:
        case "resnet18" | "resnet34" | "resnet50" | "resnet101" | "resnet152":
            projection_head = ProjectionClass(in_feature=backbone.fc.in_features,
                                              hidden_feature=projector_hidden_feature, out_feature=projector_out_feature)
            backbone.fc = nn.Identity()
        case "mobilenet_v3_small" | "mobilenet_v3_large":
            projection_head = ProjectionClass(in_feature=backbone.classifier[0].in_features,
                                              hidden_feature=projector_hidden_feature,
                                              out_feature=projector_out_feature)
            backbone.classifier = nn.Identity()
        case "vit_b_16" | "vit_b_32" | "vit_l_16" | "vit_l_32" | "vit_h_14":
            projection_head = ProjectionClass(in_feature=backbone.heads[-1].in_features,
                                              hidden_feature=projector_hidden_feature,
                                              out_feature=projector_out_feature)
            backbone.heads = nn.Identity()
        case _:
            logger.error(f"Backbone not supported: {backbone_name}")
            raise ValueError(f"Backbone not supported: {backbone_name}")
    return CLRModel(backbone, projection_head)

class CLRLightningModule(L.LightningModule):
    def __init__(self, clr_model_or_backbone_name: CLRModel | BackBonesType,
                 contrastive_approach: ContrastiveApproachType  = 'simclr',
                 optimizer_name: OptimizerType = 'adam',
                 lr: float=1e-3,
                 active_groups: list=None):
        super().__init__()
        self.backbone_name = clr_model_or_backbone_name if isinstance(clr_model_or_backbone_name, str) else None
        self.save_hyperparameters(ignore=[] if isinstance(clr_model_or_backbone_name, str) else ['clr_model'])
        projector_out_feature=128
        self.clr_model = build_model(clr_model_or_backbone_name, 1024, projector_out_feature) \
                         if isinstance(clr_model_or_backbone_name, str) else clr_model_or_backbone_name
        self.model = self.clr_model
        self.projector = ProjectionClass(in_feature=projector_out_feature,
                                         hidden_feature=1024,
                                         out_feature=projector_out_feature)
        self.contrastive_approach = contrastive_approach
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.active_groups = active_groups
        match contrastive_approach:
            case 'simclr': self.criterion = NTXentLoss()
            case 'slfpn' : self.criterion = MSELoss()
            case 'supcon': self.criterion = SupConLoss()


    def forward(self, x):
        return self.model(x)

    def _sim_clr_shared_step(self, batch):
        data, _ = batch
        first_batch, second_batch = global_global_augmentation(
            data, self.device
        )
        embd_1 = self(first_batch)
        embd_2 = self(second_batch)
        return self.criterion(embd_1, embd_2)
    def _slfpn_shared_step(self, batch):
        data, _ = batch
        batch_1, batch_2 = global_global_augmentation(data, self.device)
        embedding_batch_1 = self(batch_1)
        embedding_batch_2 = self(batch_2)
        embedding = self(data)
        loss_1 = self.criterion(embedding_batch_1, embedding_batch_2)
        loss_2 = self.criterion(embedding, embedding_batch_1)
        original_projection = self.projector(embedding)
        loss_3 = self.criterion(original_projection, embedding_batch_2.detach())
        loss = loss_1 + loss_2 + loss_3
        return loss
    def __supcon_shared_step(self, batch):
        pass
    def _shared_step(self, batch):
        loss = None
        match self.contrastive_approach:
            case 'simclr':
                loss = self._sim_clr_shared_step(batch)
            case 'slfpn':
                loss = self._slfpn_shared_step(batch)
            case 'supcon':
                loss = self.__supcon_shared_step(batch)
        return loss
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = None
        parameters = self.model.parameters() if self.projector is None \
                    else itertools.chain(self.model.parameters(), self.projector.parameters())
        match self.optimizer_name:
            case 'sgd':
                optimizer = torch.optim.SGD(parameters, lr=self.lr)
            case 'adam':
                optimizer = torch.optim.Adam(parameters, lr=self.lr)
            case 'adamw':
                optimizer = torch.optim.AdamW(parameters, lr=self.lr)
            case 'rmsprop':
                optimizer = torch.optim.RMSprop(parameters, lr=self.lr)
            case _:
                raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        return optimizer
