import torch
import torchvision.transforms as transforms
from torchvision import transforms
import random
import numpy as np
from PIL import Image

transformation_groups = {
    "translations": [transforms.RandomAffine(degrees=0,translate=(0.8, 0.0))],
    "rotations":    [transforms.RandomRotation(degrees=(10, 200))],
    "ResizedCrop":  [transforms.RandomResizedCrop(size=128, scale=(0.3, 1.0))],
    "ColorJitter":  [transforms.RandomApply([transforms.ColorJitter(brightness=0.8,contrast=0.8,saturation=0.8,hue=0.4)],p=0.8)],
    "GaussianBlur": [transforms.RandomApply([transforms.GaussianBlur(kernel_size=9,sigma=(0.1, 25.0))],p=0.5)],
    "RandomHorizontalFlip": [transforms.RandomHorizontalFlip(p=0.5)],
    "RandomVerticalFlip": [transforms.RandomVerticalFlip(p=0.5)],
    # "Composite": [transforms.Compose()] compose any augmenation
}
all_transforms = []
for transform_list in transformation_groups.values():
    all_transforms.extend(transform_list)

# def get_transformations(active_groups):
#     selected = []
#     group = random.choice(active_groups)
#     augmentation = random.choice(transformation_groups[group])
#     return augmentation

def get_transformations():
    return random.choice(all_transforms)

def global_global_augmentation(batch_images, device):
    aug1 = get_transformations()
    aug2 = get_transformations()
    first_batch = []
    second_batch = []
    for img in batch_images:
        first_batch.append(aug1(img))
        second_batch.append(aug2(img))
    return torch.stack(first_batch).to(device), torch.stack(second_batch).to(device)


# def local_global_augmentation(batch_image, active_groups, segmentation_model, device):
#     segmentation_model.eval()
#     batch_image = batch_image.to(device)
#     selected_augmentation = get_transformations(active_groups)
#     first_batch = []
#     second_batch = []
#     with torch.no_grad():
#         prediction_mask = segmentation_model(batch_image)
#     batch_images = batch_image.cpu().numpy()
#     for index, image in enumerate(batch_images):
#         image_permuted = image.transpose(1, 2, 0)
#         image_permuted_1 = image_permuted.copy()
#         for i, box in enumerate(boxes):
#             if scores[i] > 0.5:
#                 x1, y1, x2, y2 = box.astype(int)
#                 mask = masks[i, 0] > 0.5
#                 mask_resized = mask[y1:y2, x1:x2]
#                 colored_object_transformed = full_transformed_image[:, y1:y2, x1:x2].permute(1, 2, 0).cpu().numpy()
#                 image_permuted_1[y1:y2, x1:x2][mask_resized] = colored_object_transformed[mask_resized]
#         image_to_save_1 = image_permuted_1.transpose(2, 0, 1)
#         first_batch.append(image_to_save_1)
#
#         boxes = prediction_mask[index]['boxes'].cpu().numpy()
#         masks = prediction_mask[index]['masks'].cpu().numpy()
#         scores = prediction_mask[index]['scores'].cpu().numpy()
#         full_image_to_transform = image.transpose(1, 2, 0)
#         full_image_to_transform = Image.fromarray((full_image_to_transform * 255).astype(np.uint8))
#         full_transformed_image = selected_augmentation(full_image_to_transform)
#         full_transformed_image = np.array(full_transformed_image) / 255.0
#         full_transformed_image = torch.from_numpy(full_transformed_image).permute(2, 0, 1).float()
#         second_batch.append(full_transformed_image)
#
#     first_batch = np.array(first_batch)
#     second_batch = np.array(second_batch)
#     first_batch = torch.from_numpy(first_batch).float().to(device)
#     second_batch = torch.from_numpy(second_batch).float().to(device)
#     return first_batch.to(device), second_batch.to(device)