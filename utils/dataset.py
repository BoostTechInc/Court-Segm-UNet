import os
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
# import albumentations as A


def split_on_train_val(img_dir, val_names):
    '''
    Split a dataset on training and validation ids
    '''
    names = [n for n in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, n))]
    train_ids = []
    val_ids = []

    for name in names:
        subdir = os.path.join(img_dir, name)
        ids = [os.path.join(name,file.split('.')[0])
               for file in listdir(subdir) if not file.endswith('.')]
        if any(name == n for n in val_names):
            val_ids += ids
        else:
            train_ids += ids

    logging.info(f'Data has been splitted. Train ids: {len(train_ids)}, val ids: {len(val_ids)}')
    return train_ids, val_ids


class BasicDataset(Dataset):
    def __init__(self, ids, img_dir, mask_dir, num_classes=1, target_size=(1280,720)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.target_size = target_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, pil_img, target_size):
        pil_img = pil_img.resize(target_size)
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        # To tensor:
        img_tensor = torch.from_numpy(img_trans).type(torch.FloatTensor)

        return img_tensor

    def preprocess_mask(self, pil_mask, target_size):
        pil_mask = pil_mask.resize(target_size, resample=Image.NEAREST)
        mask_nd = np.array(pil_mask)

        # Prepare one hot label:
        mask_tensor = torch.from_numpy(mask_nd).type(torch.LongTensor)
        # mask_tensor = torch.nn.functional.one_hot(mask_tensor, self.num_classes)  # one hot label
        # mask_tensor = mask_tensor.permute(2, 0, 1)     # HWC to CHW

        return mask_tensor

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.mask_dir + idx + '.*')
        img_file = glob(self.img_dir + idx + '.*')

        assert len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {}: {}'.format(idx, mask_file)
        assert len(img_file) == 1, \
            'Either no image or multiple images found for the ID {}: {}'.format(idx, img_file)
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img is not None and mask is not None

        img = self.preprocess_img(img, self.target_size)
        mask = self.preprocess_mask(mask, self.target_size)

        return {
            'image': img,
            'mask': mask
        }