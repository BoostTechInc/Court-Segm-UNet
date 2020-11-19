import argparse
import logging
import os

import torch
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from unet import UNet, CourtReconstruction
from utils.dataset import BasicDataset, load_template
from utils.postprocess import preds_to_masks, mask_to_image


def predict_img(net, full_img, device, input_size):
    # Preprocess input image:
    img = BasicDataset.preprocess_img(full_img, input_size)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    net.eval()

    # Predict:
    with torch.no_grad():
        mask_pred, mask_proj = net(img)

    # Tensors to ndarrays:
    mask = preds_to_masks(mask_pred, net.n_classes)
    proj = mask_proj * net.n_classes
    proj = proj.type(torch.IntTensor).cpu().numpy().astype(np.uint8)

    return mask, proj


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


def get_img_paths(src_dir, dst_mask_dir, dst_proj_dir):
    names = [file for file in os.listdir(src_dir) if not file.endswith('.')]
    input_paths = [os.path.join(src_dir, n) for n in names]

    out_mask_paths = [os.path.join(dst_mask_dir, n.split('.')[0] + '.png') for n in names]
    out_proj_paths = [os.path.join(dst_proj_dir, n.split('.')[0] + '.png') for n in names]

    return input_paths, out_mask_paths, out_proj_paths


def test(net, input_paths, out_mask_paths, out_proj_paths, in_size, out_size):
    # Loop over all images:
    paths = zip(input_paths, out_mask_paths, out_proj_paths)

    with tqdm(total=len(input_paths), desc='predicting', unit='img') as pbar:
        for i, (in_path, out_mask_path, out_proj_path) in enumerate(paths):
            logging.info("\nPredicting image {} ...".format(in_path))

            # Open image:
            img = Image.open(in_path)

            # Predict:
            mask, proj = predict_img(net=net, full_img=img, input_size=in_size, device=device)

            # Postprocessing:
            rgb_mask = mask_to_image(mask)[0]
            if rgb_mask.shape[0] != out_size[0] or rgb_mask.size[1] != out_size[1]:
                rgb_mask = cv2.resize(rgb_mask, out_size, interpolation=cv2.INTER_NEAREST)

            proj = mask_to_image(proj)[0]
            if proj.shape[0] != out_size[0] or proj.size[1] != out_size[1]:
                proj = cv2.resize(proj, out_size, interpolation=cv2.INTER_NEAREST)

            # Save:
            cv2.imwrite(out_mask_path, rgb_mask)
            cv2.imwrite(out_proj_path, proj)
            logging.info("Mask and projection saved to {} and {}".format(out_mask_path, out_proj_path))

            pbar.update()

    print ('Done!')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-s', '--input_size', default=(640, 360),
                        help='Size of input')
    parser.add_argument('--output_size', default=(1280, 720),
                        help='Size of output')
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
    parser.add_argument('-tp', '--temp-path', dest='temp_path', type=str, default=None,
                        help='Path to court template that will be projected by affine matrix')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.model = '/home/darkalert/builds/Court-Segm-UNet/checkpoints/NCAAM80k_kornia_640x360_aug_deconv/CP_epoch5.pth'
    args.temp_path = '/home/darkalert/BoostJob/camera_calibration/college_court_masks/mask_template_v3_3_end_4k.png'

    args.src_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/frames_test/'
    args.dst_mask_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/_test/preds/NCAAM80k_kornia_640x360_aug_deconv/'
    args.dst_proj_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/_test/projs/NCAAM80k_kornia_640x360_aug_deconv/'

    args.bilinear = False
    args.n_classes = 4

    # Get videos:
    video_names = [n for n in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, n))]
    # video_names = ['train_JadenMcDaniels_0_Isolation_IncludingPasses_Offense_2019-2020_NCAAM']

    logging.info("Loading model {}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Load the court template:
    template = load_template(args.temp_path,
                             num_classes=args.n_classes,
                             target_size=args.output_size,
                             batch_size=1)
    template = template.to(device=device)

    # Load model:
    net = CourtReconstruction(n_channels=3,
                              n_classes=args.n_classes,
                              bilinear=args.bilinear,
                              template=template,
                              target_size=args.output_size)

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    # Loop over all videos:
    for name in video_names:
        print('Processing {}...'.format(name))

        # Get paths:
        src_dir = os.path.join(args.src_dir, name)
        dst_mask_dir = os.path.join(args.dst_mask_dir, name)
        dst_proj_dir = os.path.join(args.dst_proj_dir, name)
        if not os.path.exists(dst_mask_dir): os.makedirs(dst_mask_dir)
        if not os.path.exists(dst_proj_dir): os.makedirs(dst_proj_dir)

        input_paths, out_mask_paths, out_proj_paths = get_img_paths(src_dir, dst_mask_dir, dst_proj_dir)

        test(net, input_paths, out_mask_paths, out_proj_paths,
             in_size=args.input_size, out_size=args.output_size)

    print ('All done!')