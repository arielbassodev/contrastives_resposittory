import matplotlib.pyplot as plt
import random
from typing import Sequence, List, Any, Callable
import torch
import torchvision.models

from app_logger import logger

from typing import Literal, Any, List, cast, get_args

BackBonesType = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                        "mobilenet_v3_small", "mobilenet_v3_large",
                        "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"
                        ]

# get_args(Literal["A", "B"]) -->	('A', 'B') , lisst(get_args(Literal["A", "B"])) ---> ["A", "B"]
BACKBONES = list(get_args(BackBonesType))

ContrastiveApproachType = Literal['simclr','slfpn', 'supcon']
CONTRASTIVES_APPROACHES = list(get_args(ContrastiveApproachType))

OptimizerType = Literal['sgd', 'adam', 'adamw', 'rmsprop']
OPTIMIZERS = list(get_args(OptimizerType))

def plot_training_loss(epoch_losses, epochs):
 plt.figure(figsize=(8, 6))
 plt.plot(range(1, epochs + 1), epoch_losses, marker='o', label='Training Loss')
 plt.xlabel('Epoch')
 plt.ylabel('Loss')
 plt.title('Training Loss per Epoch')
 plt.grid(True)
 plt.legend()
 plt.show()


class RandomMultiChoices(torch.nn.Module):
  """Apply multiple transformations randomly picked from a list."""

  def __init__(self, transforms: List[Any], n_choice=1):
   super().__init__()
   if not (0 <= n_choice < len(transforms)):
    raise ValueError(f"n_choice should be between 0 and {len(transforms) - 1}")
   self.n_choices = n_choice
   self.transforms = transforms

  def forward(self, img):
   selected_transforms = random.sample(self.transforms, k=self.n_choices)
   for t in selected_transforms:
    img = t(img)
   return img

  def __repr__(self) -> str:
   format_string = self.__class__.__name__ + "("
   for t in self.transforms:
    format_string += "\n"
    format_string += f"    {t}"
   format_string += "\n)"
   return format_string

def get_model_default_transforms(model_name: str) -> Callable:
  try:
   weight_enums = torchvision.models.get_model_weights(model_name)
   return weight_enums.DEFAULT.transforms()
  except Exception as e:
   error_msg = f'Model weights or transform for {model_name} not found'
   logger.error("%s : %s", error_msg, e)
   raise ValueError(error_msg)

    