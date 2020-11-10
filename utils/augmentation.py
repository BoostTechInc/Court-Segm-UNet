import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def make_apperance_transform(aug):
    # Params by default:
    brightness = 0.35
    contrast = 0.35
    saturation = 0.25
    hue = 0.25

    # Set params from aug:
    if 'brightness' in aug: brightness = aug['brightness']
    if 'contrast' in aug: contrast = aug['contrast']
    if 'saturation' in aug: saturation = aug['saturation']
    if 'hue' in aug: hue = aug['hue']

    TF = transforms.Compose(
        [
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.GaussianBlur(5),
        ]
    )
    return TF

def make_geometric_transform(aug, target_widht=1280, target_height = 720):
    # Params by default:
    scale=(0.5, 1.0)
    hflip=0.5

    # Set params from aug:
    if 'scale' in aug: scale = aug['scale']
    if 'hflip' in aug: hflip = aug['hflip']

    ratio = target_height / float(target_widht)
    TF = transforms.Compose(
        [
            transforms.RandomResizedCrop((target_height,target_widht), scale=scale,
                                         ratio=(ratio,ratio), interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(hflip)
        ]
    )
    return TF

def apply_transforms(img, mask, TF_apperance=None, TF_geometric=None, geometric_same=True):
    '''
    :geometric_same: If True, then applies the same geometric transform to input mask
    '''
    assert TF_apperance is not None or TF_geometric is not None

    # Convert temporally to specific dtype:
    img_dtype, mask_dtype = img.dtype, mask.dtype
    img = img.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.uint8)

    # Make a seed with numpy generator:
    seed = np.random.randint(2147483647)

    # Transform img:
    if TF_apperance is not None:
        img = TF_apperance(img)
    if geometric_same:
        torch.manual_seed(seed)
    if TF_geometric is not None:
        img = TF_geometric(img)

    # Transform mask with the same seed:
    if geometric_same:
        torch.manual_seed(seed)
    if TF_geometric is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = TF_geometric(mask)

    # Convert back to the original dtype:
    img = img.to(dtype=img_dtype)
    mask = mask.to(dtype=mask_dtype)

    return img, mask


if __name__ == '__main__':
    '''
    Augmentation test
    '''
    from torch.utils.data import DataLoader
    from dataset import BasicDataset, split_on_train_val, worker_init_fn

    # Paths:
    img_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM_2/f_t/'
    mask_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM_2/m_t/'
    idxs = ['1']
    dst_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/_test/aug/'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Prepare data:
    _, ids = split_on_train_val(img_dir, idxs)
    data = BasicDataset(ids, img_dir, mask_dir, 4, (1280,720), aug=True)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8,
                        pin_memory=True, worker_init_fn=worker_init_fn)

    # Aplly the transforms to all images and masks:
    for bi, batch in enumerate(loader):
        img, mask = batch['image'][0], batch['mask'][0]

        img = transforms.ToPILImage(mode='RGB')(img)
        mask = transforms.ToPILImage(mode='L')(mask.to(dtype=torch.uint8))

        # Save:
        dst_path = os.path.join(dst_dir, '{}.jpeg'.format(bi))
        img.save(dst_path, 'JPEG')
        dst_path = os.path.join(dst_dir, '{}_mask.png'.format(bi))
        mask.save(dst_path, 'PNG')

    print ('Done!')








