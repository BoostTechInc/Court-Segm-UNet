import torch
import torch.nn.functional as F
from dice_loss import dice_coeff


def eval_net(net, loader, device, verbose=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    imgs, mask_pred = None, None

    print ('\nStarting validation...\n')

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_classes > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()

    net.train()

    result = {'val_score': tot/n_val}
    if verbose:
        result['imgs'] = imgs.cpu()
        result['preds'] = mask_pred.cpu()

    return result
