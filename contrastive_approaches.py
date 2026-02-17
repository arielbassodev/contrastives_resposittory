import torch.nn as nn
import torch.nn.functional as F
import torch
import lightning as L
from typing import  Literal
from lightly.loss import NTXentLoss
import torchvision
from augmentation import global_global_augmentation
from utils import ContrastiveApproachType, OptimizerType
from app_logger import logger
from utils import BackBonesType
from typing import Any

# Projection head for the different contrastive learning approaches
class Projection_class(nn.Module):
    
    def __init__(self, in_feature,hidden_feature, out_feature):
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

    def __init__(self, backbone, projector):
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        x = self.projector(x)
        return x
# SimCLR a contrastive learning approaches that leverages positives and negatives pairs
   
class SimCLR(nn.Module):
    
    def __init__(self, Backbone, Projection):
        super().__init__()
        self.backbone = Backbone
        self.head_projection = Projection
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        x = self.head_projection(x)
        return x 
    
# SLFPN another self-supervised representation learning approaches using only positive pairs, the main difference is during the training

class SLFPN(nn.Module):
    
    def __init__(self, Backbone, Projection_head):
        super().__init__()
        self.backbone = Backbone
        self.projection_head = Projection_head
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.head_projection(x)
        return x


def build_model(backbone: BackBonesType, projector_hidden_feature=1024, projector_out_feature=128 ) -> Any:
    logger.info({"Chosen backbone": backbone})
    model_weights = torchvision.models.get_model_weights(backbone).DEFAULT
    model = torchvision.models.get_model(name=backbone, weights=model_weights)
    projector = None
    match backbone:
        case "resnet18" | "resnet34" | "resnet50" | "resnet101" | "resnet152":
            projector = Projection_class(in_feature=model.fc.in_features,
                                        hidden_feature=projector_hidden_feature, out_feature=projector_out_feature)
            model.fc = nn.Identity()
        case "mobilenet_v3_small" | "mobilenet_v3_large":
            projector = Projection_class(in_feature=model.classifier[0].in_features,
                                                                    hidden_feature=projector_hidden_feature,
                                                                    out_feature=projector_out_feature)
            model.classifier = nn.Identity()
        case "vit_b_16" | "vit_b_32" | "vit_l_16" | "vit_l_32" | "vit_h_14":
            projector = Projection_class(in_feature=model.heads[-1].in_features,
                                                           hidden_feature=projector_hidden_feature,
                                                           out_feature=projector_out_feature)
            model.heads = nn.Identity()
        case _:
            logger.error(f"Backbone not supported: {backbone}")
            raise ValueError(f"Backbone not supported: {backbone}")
    return CLRModel(backbone, projector)

class CLRLightningModule(L.LightningModule):
    def __init__(self, clr_model_or_backbone_name: CLRModel | BackBonesType, contrastive_approach: ContrastiveApproachType  = 'simclr',
                 optimizer_name: OptimizerType = 'adam',
                 lr: float=1e-3,
                 active_groups: list=None):
        super().__init__()
        self.backbone_name = clr_model_or_backbone_name if isinstance(clr_model_or_backbone_name, str) else None
        self.save_hyperparameters(ignore=[] if isinstance(clr_model_or_backbone_name, str) else ['clr_model'])
        self.clr_model = build_model(clr_model_or_backbone_name, 1024, 128) \
                         if isinstance(clr_model_or_backbone_name, str) else clr_model_or_backbone_name
        self.model = self.clr_model.backbone
        self.projector = self.clr_model.projector
        self.contrastive_approach = contrastive_approach
        self.criterion = NTXentLoss()
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.active_groups = active_groups

    def forward(self, x):
        return self.model(x)

    def _sim_clr_shared_step(self, batch):
        data, _ = batch
        first_batch, second_batch = global_global_augmentation(
            data, self.active_groups, self.device
        )
        embd_1 = self(first_batch)
        embd_2 = self(second_batch)
        return self.criterion(embd_1, embd_2)
    def _slfpn_shared_step(self, batch):
        data, _ = batch
        batch_1, batch_2 = global_global_augmentation(data, self.active_groups)
        embedding_batch_1 = self(batch_1)
        embedding_batch_2 = self(batch_2)
        embedding = self(data)
        loss_1 = self.criterion(embedding_batch_1, embedding_batch_2)
        loss_2 = self.criterion(embedding, embedding_batch_2)
        original_projection = self.projector(embedding) # do we still need the projector here as the model already include the projector?
        loss_3 = self.criterion(original_projection, embedding_batch_2.detach())
        loss = loss_1 + loss_2 + loss_3
        return loss
    def _shared_step(self, batch):
        loss = self._sim_clr_shared_step(batch) if self.contrastive_approach == 'simclr' \
                else self._slfpn_shared_step(batch)
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
        match self.optimizer_name:
            case 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            case 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            case 'adamw':
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            case 'rmsprop':
                optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
            case _:
                raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        return optimizer
