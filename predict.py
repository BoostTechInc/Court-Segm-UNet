import argparse
import logging
import os

import torch
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from models import Reconstructor
from utils.dataset import BasicDataset, load_template
from utils.postprocess import onehot_to_image


def predict(net, full_img, device, input_size, warp=True):
    # Preprocess input image:
    img = BasicDataset.preprocess_img(full_img, input_size)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    net.eval()

    # Predict:
    with torch.no_grad():
        theta, rec_mask = net.predict(img, warp=warp)

    # Tensors to ndarrays:
    if warp:
        rec_mask = rec_mask * net.n_classes
        mask = rec_mask.type(torch.IntTensor).cpu().numpy().astype(np.uint8)
    else:
        mask = None

    return theta, mask


def get_img_paths(src_dir, dst_dir):
    names = [file for file in os.listdir(src_dir) if not file.endswith('.')]
    input_paths = [os.path.join(src_dir, n) for n in names]
    out_paths = [os.path.join(dst_dir, n.split('.')[0] + '.png') for n in names]

    return input_paths, out_paths


def test(net, input_paths, out_paths, in_size, out_size, warp):
    # Loop over all images:
    paths = zip(input_paths, out_paths)

    with tqdm(total=len(input_paths), desc='predicting', unit='img') as pbar:
        for i, (in_path, out_path) in enumerate(paths):
            # Open image:
            img = Image.open(in_path)

            # Predict:
            _, mask = predict(net=net, full_img=img, input_size=in_size, device=device, warp=warp)

            # Postprocessing:
            mask = onehot_to_image(mask, net.n_classes)[0]
            if mask.shape[0] != out_size[0] or mask.size[1] != out_size[1]:
                mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)

            # Save:
            cv2.imwrite(out_path, mask)

            pbar.update()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-s', '--input_size', default=(640, 360),
                        help='Size of input')
    parser.add_argument('--output_size', default=(1280, 720),
                        help='Size of output')
    parser.add_argument('--n_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='Number of output classes', dest='n_classes')
    parser.add_argument('--src_dir', '-i', metavar='INPUT', nargs='+',
                        help='the dir containing input images', required=False)
    parser.add_argument('--dst_dir', '-o', metavar='INPUT', nargs='+',
                        help='the dir where the results will be saved')
    parser.add_argument('--viz', '-v', action='store_true', default=False,
                        help="Visualize the images as they are processed")
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help="Use bilinear interpolation (True) or deconvolution")
    parser.add_argument('--temp-path', dest='temp_path', type=str, default='mask_template_v3_3_end_4k.png',
                        help='Path to court template that will be projected by affine matrix')
    parser.add_argument('--resnet', type=str, default='resnetreg18',
                        help='Specify ResNetReg model parameters')
    parser.add_argument('--warp', action='store_true', default=True,
                        help="Whether need to warp the template using the predicted transformation matrix or not")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Get params:
    args = get_args()
    args.model = '/home/darkalert/builds/Court-Segm-UNet/checkpoints/NCAAM80k_kornia_640x360_aug_deconv_reconstructor2/CP_epoch5.pth'
    args.temp_path = '/home/darkalert/BoostJob/camera_calibration/college_court_masks/mask_template_v3_3_end_4k.png'

    args.src_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/frames_test/'
    args.dst_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/_test/reconstructed/NCAAM80k_kornia_640x360_aug_deconv_reconstructor/'

    args.bilinear = False
    args.n_classes = 4
    args.resnet = 'resnetreg18'

    # Get videos:
    # video_names = [n for n in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, n))]
    video_names = ['train_AnthonyEdwards_5_Isolation_Offense_2019-2020_NCAAM']

    # CUDA or CPU:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Load the court template:
    template = load_template(args.temp_path,
                             num_classes=args.n_classes,
                             target_size=None,#args.output_size,
                             batch_size=1)
    template = template.to(device=device)

    # Load a Reconstructor model (UNET+RESNET+STN):
    net = Reconstructor(n_channels=args.n_channels,
                        n_classes=args.n_classes,
                        template=template,
                        target_size=args.output_size,
                        bilinear=args.bilinear,
                        resnet_name=args.resnet,
                        warp_by_nearest=True)
    logging.info("Loading model from {}".format(args.model))
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")


    # Loop over all videos:
    for name in video_names:
        logging.info('Processing {}...'.format(name))

        # Get paths:
        src_dir = os.path.join(args.src_dir, name)
        dst_dir = os.path.join(args.dst_dir, name)
        if not os.path.exists(dst_dir): os.makedirs(dst_dir)

        input_paths, out_paths = get_img_paths(src_dir, dst_dir)

        test(net, input_paths, out_paths, in_size=args.input_size, out_size=args.output_size, warp=args.warp)

    logging.info('All done!')