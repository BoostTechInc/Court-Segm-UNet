import argparse
import logging
import os
import sys
from tqdm import tqdm
import signal

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, split_on_train_val, worker_init_fn, load_template
from torch.utils.data import DataLoader

from eval import eval_reconstructor
from models import Reconstructor
from utils.conf_parser import parse_conf
from utils.postprocess import preds_to_masks, mask_to_image


def train_net(net, device, img_dir, mask_dir, val_names,  num_classes,
              opt='RMSprop', aug=None, cp_dir=None, log_dir=None,
              epochs=5, batch_size=1, lr=0.0001, w_decay=1e-8,
              rec_lambda=10.0, target_size=(1280,720), recon_loss ='MSE',
              vizualize=False):
    '''
    Train UNet+UNetReg+ResNetReg model
    '''
    # Prepare dataset:
    train_ids, val_ids = split_on_train_val(img_dir, val_names)
    train = BasicDataset(train_ids, img_dir, mask_dir, num_classes, target_size, aug=aug)
    val = BasicDataset(val_ids, img_dir, mask_dir, num_classes, target_size)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=True)
    n_train = len(train)
    n_val = len(val)

    # Logger:
    writer = SummaryWriter(log_dir=log_dir,
                           comment=f'LR_{lr}_BS_{batch_size}_SIZE_{target_size}_DECONV_{net.bilinear}')
    logging.info(f'''Starting training:
        Optimizer:       {opt}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Weight decay:    {w_decay}
        Reconstruction:  {recon_loss}
        Lambda:          {rec_lambda}
        Training size:   {n_train}
        Validation size: {n_val}
        Images dir:      {img_dir}
        Masks dir:       {mask_dir}
        Checkpoints dir: {cp_dir}
        Log dir:         {log_dir}
        Device:          {device.type}
        Input size:      {target_size}
        Vizualize:       {vizualize}
        Augmentation:    {aug}
    ''')

    # Oprimizer:
    if opt == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=w_decay, momentum=0.9)
    elif opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=w_decay, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=w_decay)
    else:
        print ('optimizer {} does not support yet'.format(opt))
        raise NotImplementedError

    # Scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min' if net.n_classes > 1 else 'max', patience=3)

    # Losses:
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    if recon_loss == 'MSE':
        rec_criterion = nn.MSELoss()
    elif recon_loss == 'SmoothL1':
        rec_criterion = nn.SmoothL1Loss()
    else:
        raise NotImplementedError

    global_step = 0

    # Training loop:
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # Get data:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # CPU -> GPU:
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device)

                # Forward:
                logits, rec_masks = net(imgs)

                # Caluclate CrossEntropy loss:
                ce_loss = criterion(logits, true_masks)

                # Calculate reconstruction loss for regressors:
                gt_masks = true_masks.to(dtype=torch.float32) / float(num_classes)
                rec_loss = rec_criterion(rec_masks, gt_masks) * rec_lambda

                # Total loss:
                loss = ce_loss + rec_loss
                epoch_loss += loss.item()

                # Log:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train CE', ce_loss.item(), global_step)
                writer.add_scalar('Loss/train rec2', rec_loss.item(), global_step)
                pbar.set_postfix(**{'CE_loss': ce_loss.item(),
                                    'Rec_loss': rec_loss.item(),
                                    'Tot loss': loss.item(),})

                # Backward:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                # Validation step:
                if global_step % (n_train // (5 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        if value.grad is not None:
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    # Evaluate:
                    result = eval_reconstructor(net, val_loader, device, verbose=vizualize)
                    val_ce_score = result['val_ce_score']
                    val_rec_score = result['val_rec_score']
                    val_tot_score = val_ce_score + val_rec_score
                    scheduler.step(val_tot_score)

                    # Validation log:
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('Loss/test', val_tot_score, global_step)
                    writer.add_scalar('Loss/test_CE', val_ce_score, global_step)
                    writer.add_scalar('Loss/test_rec', val_rec_score, global_step)
                    logging.info('\nValidation tot: {}, CE: {}, rec: {}'.
                                 format(val_tot_score, val_ce_score, val_rec_score))

                    if vizualize:
                        # Postprocess predicted mask for tensorboard vizualization:
                        pred_masks = preds_to_masks(result['logits'], net.n_classes)
                        pred_masks = mask_to_image(pred_masks)
                        pred_masks = np.transpose(pred_masks, (0, 3, 1, 2))
                        pred_masks = pred_masks.astype(np.float32) / 255.0
                        pred_masks = pred_masks[...,::-1]

                        rec_masks = result['rec_masks'] * num_classes
                        rec_masks = rec_masks.type(torch.IntTensor).cpu().numpy().astype(np.uint8)
                        rec_masks = mask_to_image(rec_masks)
                        rec_masks = np.transpose(rec_masks, (0, 3, 1, 2))
                        rec_masks = rec_masks.astype(np.float32) / 255.0
                        rec_masks = rec_masks[..., ::-1]

                        # Concatenate all images:
                        output = np.concatenate((result['imgs'], pred_masks, rec_masks), axis=2)

                        # Save the results for tensorboard vizualization:
                        writer.add_images('output', output, global_step)

        # Save checkpoint:
        if cp_dir is not None:
            try:
                os.mkdir(cp_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       cp_dir + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet')
    parser.add_argument('-id', '--img_dir', dest='img_dir', type=str, default=None,
                        help='Path to dir containing traininmg images')
    parser.add_argument('-md', '--mask_dir', dest='mask_dir', type=str, default=None,
                        help='Path to dir containing masks for given images')
    parser.add_argument('-vn', '--val_names', dest='val_names', type=str, default=None,
                        help='List of video names that will be used in validation step')
    parser.add_argument('-cd', '--cp_dir', dest='cp_dir', type=str, default=None,
                        help='Path for saving checkpoints')
    parser.add_argument('-ld', '--log_dir', dest='log_dir', type=str, default=None,
                        help='Path for saving tensorboard logs')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-wd', '--weight-decay', metavar='WD', type=float, nargs='?', default=1e-8,
                        help='Weight decay', dest='weight_decay')
    parser.add_argument('--rec_lambda', type=float, default=10.0,
                        help='Lambda for reconstruction loss')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--size', dest='size', default=(640,360),
                        help='Size of input')
    parser.add_argument('--n_channels', type=int, default=3,
                        help='Number of input channels', dest='n_channels')
    parser.add_argument('--n_classes', type=int, default=5,
                        help='Number of output classes', dest='n_classes')
    parser.add_argument('--bilinear', '-bl', action='store_true',
                        help="Use bilinear interpolation (True) or deconvolution",
                        default=False)
    parser.add_argument('-c', '--conf_path', dest='conf_path', type=str, default=None,
                        help='Load config from a .yaml file')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('-o', '--opt', dest='opt', type=str, default='RMSprop',
                        help='Optimizer for training')
    parser.add_argument('-a', '--aug', dest='aug', type=str, default=None,
                        help='Augmentation')
    parser.add_argument('-tp', '--temp-path', dest='temp_path', type=str, default=None,
                        help='Path to court template that will be projected by affine matrix')
    parser.add_argument('--resnet', type=str, default=None,
                        help='Specify ResNetReg model parameters')
    parser.add_argument('--rec_loss', type=str, default='MSE',
                        help='Whether to use MSE or SmoothL1 as reconstruction loss')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Read params and replace them with ones from yaml config:
    args = get_args()
    if args.conf_path is not None:
        conf = parse_conf(args.conf_path)
        for k in vars(args).keys():
            if k in conf:
                setattr(args, k, conf[k])
        if 'aug' in conf:
            args.aug = conf['aug']
        if 'resnet' in conf:
            args.resnet = conf['resnet']
            if 'pretrained' not in args.resnet:
                args.resnet['pretrained'] = None

    # CUDA or CPU:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Load court template:
    template = load_template(args.temp_path,
                             num_classes=args.n_classes,
                             target_size=args.size,
                             batch_size=args.batchsize)
    template = template.to(device=device)

    # Init Reconstructor (UNET+RESNET+STN):
    net = Reconstructor(n_channels=args.n_channels,
                        n_classes=args.n_classes,
                        template=template,
                        target_size=args.size,
                        bilinear=args.bilinear,
                        resnet_name=args.resnet['name'],
                        resnet_pretrained=args.resnet['pretrained'])
    logging.info(f'Network Reconstructor (UNet+UNetReg+ResNetReg):\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling\n'
                 f'\tResNetReg: {args.resnet}\n')

    # Restore the model from a checkpoint:
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)              # cudnn.benchmark = True: faster convolutions, but more memory

    # Define save model function:
    def save_model(a1=None, a2=None):
        path = os.path.join(args.cp_dir, 'last.pth')
        torch.save(net.state_dict(), path)
        logging.info('Saved interrupt to {}'.format(path))
        sys.exit(0)
    signal.signal(signal.SIGTERM, save_model)

    # Run training:
    try:
        if not os.path.exists(args.cp_dir): os.makedirs(args.cp_dir)
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

        train_net(net=net,
                  device=device,
                  img_dir=args.img_dir,
                  mask_dir=args.mask_dir,
                  val_names=args.val_names,
                  cp_dir=args.cp_dir,
                  log_dir=args.log_dir,
                  num_classes=args.n_classes,
                  aug=args.aug,
                  opt=args.opt,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  w_decay=args.weight_decay,
                  rec_lambda=args.rec_lambda,
                  target_size=args.size,
                  recon_loss=args.rec_loss,
                  vizualize=args.viz)
    except KeyboardInterrupt:
        save_model()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
