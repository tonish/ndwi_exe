import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torchgeo.transforms import  indices
import skimage
import os

def load_imgs(path):

    """load images and labels paths from the split csv"""
    with open(path, 'r') as file:
        content = file.readlines()

    data_prefix = 'v1.1/data/flood_events/HandLabeled/S2Hand'
    label_prefix = 'v1.1/data/flood_events/HandLabeled/LabelHand'

    imgs = [os.path.join(data_prefix,i.split(',')[0].replace('S1Hand','S2Hand')) for i in content]
    labels = [os.path.join(label_prefix,i.split(',')[1].strip('\n')) for i in content]
    return imgs,labels


class S2Dataset(Dataset):
    def __init__(self,  imgs_dir, labels_dir,means, stds, transform=None):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.means = means
        self.stds = stds

    def __len__(self):
        return len(self.imgs_dir)

    def __getitem__(self, idx):
        img_path = self.imgs_dir[idx]
        image = torch.tensor(skimage.io.imread(img_path), dtype = torch.float32)

        norm = transforms.Normalize(self.means, self.stds)
        image = norm(image.permute(1,2,0))
        image = image.permute(2,0,1)

        #add RS indices

        VIs = nn.Sequential(
        indices.AppendBNDVI(index_nir=7, index_blue=1),
        indices.AppendGBNDVI(index_nir=7, index_green=2, index_blue=1),
        indices.AppendGNDVI(index_nir=7, index_green=2),
        indices.AppendGRNDVI(index_nir=7, index_green=2, index_red=3),
        indices.AppendNBR(index_nir=7, index_swir=11) ,
        indices.AppendNDBI(index_swir=11, index_nir=7),
        indices.AppendNDRE(index_nir=7, index_vre1=4),
        indices.AppendNDSI(index_green=2, index_swir=11),
        indices.AppendNDVI(index_nir=7, index_red=3),
        indices.AppendNDWI(index_green=2, index_nir=7)
                        )
        image = VIs(image).squeeze(-4)

        label_path = self.labels_dir[idx]
        label_img = skimage.io.imread(label_path)>0
        label = torch.unsqueeze(torch.tensor(label_img, dtype = torch.float32),0)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

class S2DataModule(pl.LightningDataModule):
    def __init__(self, train_path,valid_path,test_path,means,stds, batch_size, num_workers):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.means = means
        self.stds = stds

    def setup(self, stage):
        self.train_ds = S2Dataset(*load_imgs(self.train_path),self.means, self.stds)
        self.val_ds = S2Dataset(*load_imgs(self.valid_path),self.means, self.stds)
        self.test_ds = S2Dataset(*load_imgs(self.test_path),self.means, self.stds)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )