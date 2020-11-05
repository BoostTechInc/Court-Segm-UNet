import numpy as np
import torch
import torch.nn.functional as F



def preds_to_masks(preds, n_classes=1):
    # Predictions to labels:
    if n_classes > 1:
        probs = F.softmax(preds, dim=1)
        masks = torch.argmax(probs, dim=1)
    else:
        masks = torch.sigmoid(preds)

    masks = masks.type(torch.IntTensor).cpu().numpy().astype(np.uint8)

    return masks


def mask_to_image(masks):
    '''
    Convert grayscale mask to RGB image
    '''
    if masks.ndim == 2:
        masks = np.expand_dims(masks, 0)    # add batch dim
    masks = np.expand_dims(masks, -1)       # add last dim, need for np.all()

    # Gray mask to RGB image:
    rgb_masks = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    rgb_masks[np.all(masks == 1, axis=3)] = (0, 255, 0)
    rgb_masks[np.all(masks == 2, axis=3)] = (255, 0, 0)
    rgb_masks[np.all(masks == 3, axis=3)] = (0, 0, 255)
    rgb_masks[np.all(masks == 4, axis=3)] = (255, 255, 255)

    return rgb_masks


# def transform():
#     from torchvision import transforms
#     # Transform:
#     tf = transforms.Compose(
#         [
#             transforms.ToPILImage(),
#             transforms.Resize((target_size), interpolation=Image.NEAREST),
#             transforms.ToTensor()
#         ]
#     )
#     for p in probs:
#         p = tf(p)