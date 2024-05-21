import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

        self.images_dir = os.path.join(data_dir, 'img_emb')
        self.references_dir = os.path.join(data_dir, 'ref_crop_emb')
        self.labels_dir = os.path.join(data_dir, 'labels')

        self.image_files = sorted(os.listdir(self.images_dir))
        self.reference_files = sorted(os.listdir(self.references_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))
        
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        reference_path = os.path.join(self.references_dir, self.reference_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        image = torch.tensor(np.load(image_path), dtype=torch.float32)
        reference = torch.tensor(np.load(reference_path), dtype=torch.float32)
        label = torch.tensor(np.load(label_path), dtype=torch.float32)
        
        return image, reference, label

class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=32):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset(self.train_dir)
            self.val_dataset = CustomDataset(self.val_dir)
        
        if stage == 'validate' or stage is None:
            self.val_dataset = CustomDataset(self.val_dir)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)