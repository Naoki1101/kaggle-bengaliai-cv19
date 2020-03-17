import numpy as np
import pandas as pd

import cv2

from torch.utils.data import Dataset
from augmentation import Augmentation
from augmix import augment_and_mix

HEIGHT = 137
WIDTH = 236


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

def hwc2chw(image):
    return image.transpose((2, 0, 1))

def chw2hwc(image):
    return image.transpose((1, 2, 0))


class BengaliDataset(Dataset):
    
    def __init__(self, df, n_channels=1, is_train=True, img_size=[128, 128], transforms=None):
        self.data = df.iloc[:, list(df.columns).index('0'):].values
        self.fnames = df['image_id'].values
        self.n_channels = n_channels
        if is_train:
            self.labels_gr = df['grapheme_root'].values
            self.labels_vo = df['vowel_diacritic'].values
            self.labels_co = df['consonant_diacritic'].values
        self.img_size = img_size
        self.transforms = transforms
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = 255 - self.data[idx, :].reshape(HEIGHT, WIDTH).astype(np.uint8)
        image = (image*(255.0/image.max())).astype(np.uint8)
        # image = crop_resize(image)
        image = cv2.resize(image, dsize=(self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA)
        
        if self.transforms:
            if self.transforms.albumentations:
                aug = Augmentation().get_augmentation(self.transforms.albumentations)
                augmented = aug(image=image)
                image = augmented['image'].astype(np.float32)

        image = image.reshape(1, self.img_size[0], self.img_size[1]).astype(np.float32)
        if self.n_channels > 1:
            image = np.concatenate([image for i in range(self.n_channels)], axis=0)

        if self.is_train:
            label_gr = self.labels_gr[idx]
            label_vo = self.labels_vo[idx]
            label_co = self.labels_co[idx]
            return image, label_gr, label_vo, label_co
        else:
            return image


# class BengaliTestDataset(Dataset):
    
#     def __init__(self, df, n_channels=1, img_size=128, transforms=None, tta=5):
#         super(BengaliTestDataset, self).__init__()
#         self.data = df.iloc[:, list(df.columns).index('0'):].values
#         self.fnames = df['image_id'].values
#         self.n_channels = n_channels
#         self.img_size = img_size
#         self.transforms = transforms
#         self.tta = tta
        
#     def __len__(self):
#         return len(self.data) * self.tta
    
#     def __getitem__(self, idx):
#         new_idx = idx % len(self.data)
#         image = 255 - self.data[new_idx, :].reshape(HEIGHT, WIDTH).astype(np.uint8)
#         image = (image*(255.0/image.max())).astype(np.uint8)
#         image = crop_resize(image)
#         image = image.reshape(1, self.img_size, self.img_size).astype(np.float32)
#         if self.n_channels > 1:
#             image = np.concatenate([image for i in range(self.n_channels)], axis=0)
        
#         if self.transforms:
#             if self.transforms.albumentations:
#                 aug = Augmentation().get_augmentation(self.transforms.albumentations)
#                 augmented = aug(image=image)
#                 image = augmented['image'].astype(np.float32)

#             if self.transforms.augmix:
#                 if np.random.rand() > self.transforms.augmix.p:
#                     image = chw2hwc(image)
#                     image = augment_and_mix(image)
#                     image = hwc2chw(image)
#                     image += np.abs(image.min())
#                     image = (image / image.max()).astype(np.float32)

#             label_gr = self.labels_gr[new_idx]
#             label_vo = self.labels_vo[new_idx]
#             label_co = self.labels_co[new_idx]
#             return image, label_gr, label_vo, label_co
#         else:
#             return image
