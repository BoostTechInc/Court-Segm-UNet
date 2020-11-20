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


def eval_stn(net, loader, device, verbose=False):
    """Evaluation UNET+STN"""
    print('\nStarting validation...\n')
    ce_score, mse_score = 0, 0
    imgs, mask_pred, projected_masks = None, None, None
    mask_type = torch.long
    n_val = len(loader)

    net.eval()

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred, projected_masks = net(imgs)

        # Scores:
        ce_score += F.cross_entropy(mask_pred, true_masks).item()
        gt_masks = true_masks.to(dtype=torch.float32) / float(net.n_classes)
        mse_score += F.mse_loss(projected_masks, gt_masks).item()

    net.train()

    result = {'val_tot_score': (ce_score+mse_score)/n_val,
              'val_ce_score': ce_score/n_val,
              'val_mse_score': mse_score/n_val}
    if verbose:
        result['imgs'] = imgs.cpu()
        result['preds'] = mask_pred.cpu()
        result['projs'] = projected_masks.cpu()

    return result


def eval_reconstructor(net, loader, device, verbose=False):
    """Evaluation UNet+ResNetReg"""
    print('\nStarting validation...\n')
    ce_score, rec_score = 0, 0
    imgs, logits, rec_masks = None, None, None
    mask_type = torch.long
    n_val = len(loader)

    net.eval()

    for batch in loader:
        imgs, gt = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        gt = gt.to(device=device, dtype=mask_type)
        gt_masks = gt.to(dtype=torch.float32) / float(net.n_classes)

        with torch.no_grad():
            logits, rec_masks = net(imgs)

        # Scores:
        ce_score += F.cross_entropy(logits, gt).item()
        rec_score += F.mse_loss(rec_masks, gt_masks).item()

    net.train()

    result = {'val_ce_score': ce_score/n_val,
              'val_rec_score': rec_score/n_val}
    if verbose:
        result['imgs'] = imgs.cpu()
        result['logits'] = logits.cpu()
        result['rec_masks'] = rec_masks.cpu()

    return result