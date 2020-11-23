import argparse
import logging
import os

import torch
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from unet import UNet
from utils.dataset import BasicDataset
from utils.postprocess import preds_to_masks, mask_to_image


def predict_img(net,
                full_img,
                device,
                input_size):
    net.eval()

    img = BasicDataset.preprocess_img(full_img, input_size)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        preds = net(img)
        masks = preds_to_masks(preds, net.n_classes)    # GPU tensor -> CPU numpy

    return masks


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-s', '--input_size', default=(640, 360),
                        help='Size of input')
    parser.add_argument('--n_classes', type=int, default=5,
                        help='Number of output classes', dest='n_classes')
    parser.add_argument('--src_dir', '-i', metavar='INPUT', nargs='+',
                        help='the dir containing input images', required=False)
    parser.add_argument('--dst_dir', '-o', metavar='INPUT', nargs='+',
                        help='the dir where the results will be saved')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--bilinear', '-bl', action='store_true',
                        help="Use bilinear interpolation (True) or deconvolution",
                        default=False)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def get_img_paths(src_dir, dst_dir=None, png=True):
    names = [file for file in os.listdir(src_dir) if not file.endswith('.')]
    input_paths = [os.path.join(src_dir, n) for n in names]

    if dst_dir is not None:
        if png:
            output_paths = [os.path.join(dst_dir, n.split('.')[0]+'.png') for n in names]
        else:
            output_paths = [os.path.join(dst_dir, n) for n in names]
        return input_paths, output_paths
    else:
        return input_paths


def test(net, input_paths, output_paths, input_size):
    # Loop over all images:
    with tqdm(total=len(input_paths), desc='predicting', unit='img') as pbar:
        for i, (in_path, out_path) in enumerate(zip(input_paths, output_paths)):
            logging.info("\nPredicting image {} ...".format(in_path))

            # Open image:
            img = Image.open(in_path)
            img_size = (img.size[0], img.size[1])

            # Predict:
            mask = predict_img(net=net, full_img=img, input_size=input_size, device=device)

            # Postprocessing:
            rgb_mask = mask_to_image(mask)[0]
            if rgb_mask.shape[0] != img_size[0] or rgb_mask.size[1] != img_size[1]:
                rgb_mask = cv2.resize(rgb_mask, img_size, interpolation=cv2.INTER_NEAREST)

            # Save:
            cv2.imwrite(out_path, rgb_mask)
            logging.info("Mask saved to {}".format(out_path))

            pbar.update()

    print ('Done!')


if __name__ == "__main__":
    args = get_args()
    args.model = '/home/darkalert/builds/Court-Segm-UNet/checkpoints/NCAAM80k_640x360_aug_deconv/last.pth'
    args.src_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/frames_test/'
    args.dst_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/_test/preds/NCAAM80k_640x360_aug_deconv/'
    args.bilinear = False
    args.n_classes = 4

    # Get videos:
    video_names = [n for n in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, n))]
    # video_names = ['train_JadenMcDaniels_0_Isolation_IncludingPasses_Offense_2019-2020_NCAAM']

    # Load model:
    net = UNet(n_channels=3, n_classes=args.n_classes, bilinear=args.bilinear)
    logging.info("Loading model {}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    # Loop over all videos:
    for name in video_names:
        print('Processing {}...'.format(name))

        # Get paths:
        src_dir = os.path.join(args.src_dir, name)
        dst_dir = os.path.join(args.dst_dir, name)
        if not os.path.exists(dst_dir): os.makedirs(dst_dir)
        input_paths, output_paths = get_img_paths(src_dir, dst_dir)

        test(net, input_paths, output_paths, args.input_size)

    print ('All done!')