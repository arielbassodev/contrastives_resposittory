import os
import pandas as pd
from PIL import Image
import lightning as L
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple
from app_logger import logger
from utils import RandomMultiChoices, get_model_default_transforms
from utils import BackBonesType
import json

class CassavaDataset(Dataset):
    def __init__(self, img_dir, csv_file, json_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

        with open(json_file, 'r') as f:
            label_map = json.load(f)

        sorted_keys = sorted(label_map.keys(), key=lambda x: int(x))
        self.classes = [label_map[k] for k in sorted_keys]

        # class_to_idx is mapping from name to integer
        self.class_to_idx = {name: int(key) for key, name in label_map.items()}

        # itertuples() returns (Index, image_id, label) for each row
        self.imgs = [
            (
                os.path.join(self.img_dir, row[1]),
                int(row[2])
            )
            for row in self.annotations.itertuples()
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, target



class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class CassavaDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str | tuple[str, str, str] = "./", batch_size: int = 32, input_img_size: Tuple[int, int] = (224, 224),
                 val_split: float = 0.2, test_split: float = 0.2, n_transforms_to_choose=2,
                 base_transform_to_use: BackBonesType = None,
                 num_workers=0, random_seed=42):
        """
        :param data_dir: is either a folder where images are already put into subfolders corresponding to their classes or a tuple consisting
        of an image folder, a csv for label, and a json for correpondence between label names and label id
        (images_dir, csv_file, json_file)
        """
        super().__init__()
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.images_dir = data_dir if isinstance(data_dir, str) else data_dir[0]
        self.csv_file = None if isinstance(data_dir, str) else data_dir[1]
        self.json_file = None if isinstance(data_dir, str) else data_dir[2]
        self.num_classes = 0
        self.batch_size = batch_size
        self.input_img_size = input_img_size
        self.classes = None
        self.class_frequencies = None
        self.val_split = val_split
        self.test_split = test_split

        self.save_hyperparameters()

        self.full_dataset = None
        self.train_ds, self.val_ds, self.test_ds, self.predict_ds = None, None, None, None
        self.train_indices, self.val_indices, self.test_indices = None, None, None
        self.base_transform_to_use = base_transform_to_use
        self.n_transforms_to_choose = n_transforms_to_choose
        # Define a transformation to convert images to tensors
        if base_transform_to_use is None:
            self.inference_transform = transforms.Compose([transforms.Resize(input_img_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
        else:
            self.inference_transform = get_model_default_transforms(base_transform_to_use)

        self.train_transform = transforms.Compose([
            # DONE: Le modele doit aussi pourvoir assez souvent voir les images dans leur forme originale
            RandomMultiChoices([
                transforms.RandomAdjustSharpness(0.5),
                transforms.ColorJitter(brightness=.5),
                transforms.RandomAutocontrast(0.5),
                transforms.RandomRotation(degrees=[90, 180]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5)],
                n_choice=n_transforms_to_choose),
            self.inference_transform])

    def _is_valid_image(self, path):
        """ Function to check if an image file is valid """
        try:
            with Image.open(path) as img:
                img.verify()  # Verify that the file is an image
            return True
        except (IOError, SyntaxError, OSError) as e:
            logger.error(f"Bad image file: {path}, error: {e}")
            return False

    def _class_frequencies(self, targets):
        unique_labels, counts = torch.unique(torch.tensor(targets), return_counts=True)
        class_frequencies = {}
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            class_frequencies[label] = count
        return class_frequencies

    def prepare_data(self):
        # We check the dataset existence, but do not download as the data path is local
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"The data directory was not found at {self.images_dir}")

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if self.full_dataset is None:
            if self.csv_file is not None:
                self.full_dataset = CassavaDataset(self.images_dir, self.csv_file, self.json_file, transform=None)
            else:
                self.full_dataset = ImageFolder(self.images_dir, is_valid_file=self._is_valid_image)
            self.num_classes = len(self.full_dataset.classes)
            self.classes = self.full_dataset.classes

            logger.info(f"Class Names: {self.classes}")
            logger.info(f"Class to Index Mapping: {self.full_dataset.class_to_idx}")
            # Split the dataset into train validation and test subset (60%/20%/20%) using stratified sampling
            # Get the labels for stratified splitting
            targets = [img[1] for img in self.full_dataset.imgs]
            logger.info("Class Frequencies : %s", self._class_frequencies(targets))

            # Split into training and temporary sets (for validation and testing)
            self.train_indices, temp_indices, train_targets, temp_targets = train_test_split(
                np.arange(len(self.full_dataset)),
                targets,
                test_size=(self.val_split + self.test_split),  # 40% for validation and test combined
                random_state=self.random_seed,
                stratify=targets
            )

            cf = self._class_frequencies(train_targets)
            logger.info("Train Class Frequencies --- %s", cf)
            self.class_frequencies = [cf[i] for i in range(self.num_classes)]

            # Split the temporary set into validation and testing sets
            self.val_indices, self.test_indices, _v, _t = train_test_split(
                temp_indices,
                temp_targets,
                # Note that the numerator is test_split since we are usint test_size here. This represent the proportion of the second part of the split which in our case is the test set, the first beeing the val set
                test_size=self.test_split / (self.val_split + self.test_split),
                # 50% of the temporary set (which is 20% of the original)
                random_state=self.random_seed,
                stratify=temp_targets
            )

            logger.info("Val Class Frequencies --- %s", self._class_frequencies(_v))
            logger.info("Test Class Frequencies --- %s", self._class_frequencies(_t))
        if stage == "fit":
            # Wrap the base subsets with the custom TransformedSubset to apply different transforms + Data augmentation
            train_subset = Subset(self.full_dataset, self.train_indices)
            val_subset = Subset(self.full_dataset, self.val_indices)
            self.train_ds = TransformedDataset(train_subset, transform=self.train_transform)

            # TODO: Do we need to augment data for validation and testing??? --No
            self.val_ds = TransformedDataset(val_subset, transform=self.inference_transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            # TODO: Do we need to augment data for validation and testing??? No
            test_subset = Subset(self.full_dataset, self.test_indices)
            self.test_ds = TransformedDataset(test_subset, transform=self.inference_transform)

        if stage == "predict":
            test_subset = Subset(self.full_dataset, self.test_indices)
            self.predict_ds = TransformedDataset(test_subset, transform=self.inference_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(self.random_seed), num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size, num_workers=self.num_workers)
