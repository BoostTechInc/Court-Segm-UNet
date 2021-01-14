import os
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import json
from utils.augmentation import make_apperance_transform, make_geometric_transform, apply_transforms


def worker_init_fn(worker_id):
    '''
    Use this function to set the numpy seed of each worker
    For example, loader = DataLoader(..., worker_init_fn=worker_init_fn)
    '''
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def split_on_train_val(img_dir, val_names, only_ncaam=False):
    '''
    Split a dataset on training and validation ids
    '''
    names = [n for n in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, n))]
    train_ids = []
    val_ids = []

    for name in names:
        subdir = os.path.join(img_dir, name)
        ids = [os.path.join(name,file.split('.jpeg')[0])
               for file in listdir(subdir) if not file.endswith('.')]
        if any(name == n for n in val_names):
            val_ids += ids
        else:
            if only_ncaam and name[0] == '2':
                print('Skip', name)
                continue
            train_ids += ids

    logging.info(f'Data has been splitted. Train ids: {len(train_ids)}, val ids: {len(val_ids)}')
    return train_ids, val_ids

def open_court_template(path, num_classes, size=None, batch_size=1):
    '''
    Load the court template image that will be projected by affine matrix from STN
    '''
    template = Image.open(path)
    if size is not None:
        template = template.resize(size, resample=Image.NEAREST)
    template = np.array(template) / float(num_classes)
    template_tensor = torch.from_numpy(template).type(torch.FloatTensor)

    while template_tensor.ndim < 4:
        template_tensor = template_tensor.unsqueeze(0)
    template_tensor = template_tensor.repeat(batch_size, 1, 1, 1)

    return template_tensor

def open_court_poi(path, size=None, batch_size=1, homogeneous=True, transpose=True):
    '''
    Load the points of interest of court template
    '''
    points = None

    with open(path) as f:
        try:
            points_data = json.load(f)
            points_raw = points_data['points']
            ranges = points_data['ranges']
            assert ranges[0] == 1.0 and ranges[1] == 1.0
            points = []
            for p in points_raw:
                x,y= int(round(p['coords'][0] * size[0])), int(round(p['coords'][1] * size[1]))
                if homogeneous:
                    points.append((x,y,1))
                else:
                    points.append((x, y))
            points = np.array(points)

        except Exception as e:
            raise ValueError(f'Cannot read {path}: {str(e)}')

    points_tensor = torch.from_numpy(points).type(torch.FloatTensor)
    if transpose:
        points_tensor = points_tensor.transpose(0,1)
    points_tensor = points_tensor.unsqueeze(0)
    points_tensor = points_tensor.repeat(batch_size, 1, 1)

    return points_tensor


class BasicDataset(Dataset):
    def __init__(self, ids, img_dir, mask_dir, anno_dir=None, anno_keys=None, num_classes=1, target_size=(1280,720), aug=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.anno_dir = anno_dir
        self.ids = ids
        self.target_size = target_size
        self.num_classes = num_classes
        self.aug = aug
        self.anno_keys = anno_keys
        self.TF_apperance = None
        self.TF_img_geometric = None
        self.TF_msk_geometric = None

        assert anno_dir != None and anno_keys != None or anno_dir == None

        # Get transforms:
        if self.aug is not None:
            if 'apperance' in self.aug and self.aug['apperance'] is not None:
                self.TF_apperance = make_apperance_transform(self.aug['apperance'])
            if 'geometric' in self.aug and self.aug['geometric'] is not None:
                self.TF_img_geometric = make_geometric_transform(self.aug['geometric'],
                                                                 target_size[0],
                                                                 target_size[1],
                                                                 Image.BILINEAR)
                self.TF_msk_geometric = make_geometric_transform(self.aug['geometric'],
                                                                 target_size[0],
                                                                 target_size[1],
                                                                 Image.NEAREST)

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
        mask_tensor = torch.from_numpy(mask_nd).type(torch.LongTensor)

        return mask_tensor

    def preprocess_anno(self, np_anno):
        anno_tensor = torch.from_numpy(np_anno).type(torch.FloatTensor)
        # anno_tensor = anno_tensor.unsqueeze(0)

        return anno_tensor

    def __getitem__(self, i):
        idx = self.ids[i]
        sample = {}

        # Get image and mask paths:
        img_file = glob(self.img_dir + idx + '.jpeg')
        mask_file = glob(self.mask_dir + idx + '.png')
        anno_file = glob(self.anno_dir + idx + '.json') if self.anno_dir is not None else None
        assert len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {}: {}'.format(idx, mask_file)
        assert len(img_file) == 1, \
            'Either no image or multiple images found for the ID {}: {}'.format(idx, img_file)
        assert anno_file is None or len(anno_file) == 1, \
            'Either no json or multiple json found for the ID {}: {}'.format(idx, anno_file)

        # Open image and mask:
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        assert img is not None and mask is not None

        # Preprocess image and mask:
        img = self.preprocess_img(img, self.target_size)
        mask = self.preprocess_mask(mask, self.target_size)

        # Open and preprocess annotations:
        if anno_file is not None:
            with open(anno_file[0], 'r') as json_file:
                json_data = json.load(json_file)
                for k in self.anno_keys:
                    anno = np.asarray(json_data[k], dtype='float')
                    sample[k] = self.preprocess_anno(anno)

        # Augmentation:
        if self.aug is not None:
            img, mask = apply_transforms(img, mask,
                                         self.TF_apperance,
                                         self.TF_img_geometric,
                                         self.TF_msk_geometric)

        if mask.ndim == 3:
            mask = mask.squeeze(0)        # [1,h,w] -> [h,w]

        sample['image'] = img
        sample['mask'] = mask

        return sample