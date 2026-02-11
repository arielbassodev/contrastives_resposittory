import torch.nn as nn
import torch.nn.functional as F

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
    
     
    