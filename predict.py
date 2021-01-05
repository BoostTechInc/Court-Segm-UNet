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
from utils.postprocess import preds_to_masks, onehot_to_image, overlay


def predict(net, full_img, device, input_size, mask_way='warp'):
    '''
    :mask_type: Sets the way to obtain the mask. Сan take 'warp' or 'segm'
    '''

    # Preprocess input image:
    img = BasicDataset.preprocess_img(full_img, input_size)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    net.eval()

    # Predict:
    with torch.no_grad():
        logits, rec_mask, theta = net.predict(img, warp=True if mask_way=='warp' else False)

    if mask_way == 'warp':
        mask = rec_mask * net.n_classes
        mask = mask.type(torch.IntTensor).cpu().numpy().astype(np.uint8)
    elif mask_way == 'segm':
        mask = preds_to_masks(logits, net.n_classes)
    else:
        raise NotImplementedError

    return mask, theta

def get_img_paths(src_dir, dst_dir):
    names = [file for file in os.listdir(src_dir) if not file.endswith('.')]
    input_paths = [os.path.join(src_dir, n) for n in names]
    out_paths = [os.path.join(dst_dir, n.split('.')[0]) for n in names]

    return input_paths, out_paths


def test(net, input_paths, out_paths, in_size, out_size, mask_way='warp', blend=False, out_format='jpeg'):
    # Loop over all images:
    paths = zip(input_paths, out_paths)

    with tqdm(total=len(input_paths), desc='predicting', unit='img') as pbar:
        for i, (in_path, out_path) in enumerate(paths):
            # Open image:
            img = Image.open(in_path)

            # Predict:
            mask, _ = predict(net=net, full_img=img, input_size=in_size, device=device, mask_way=mask_way)

            # Postprocessing:
            mask = onehot_to_image(mask, net.n_classes)[0]
            if mask.shape[0] != out_size[0] or mask.size[1] != out_size[1]:
                mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)

            if blend:
                img_np = np.array(img)
                if img_np.shape[0] != out_size[0] or img_np.size[1] != out_size[1]:
                    img_np = cv2.resize(img_np, out_size)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                mask = overlay(img_np, mask)

            # Save:
            if out_format == 'jpeg':
                out_path += '.jpeg'
                cv2.imwrite(out_path, mask, [cv2.IMWRITE_JPEG_QUALITY, 90])
            elif out_format == 'png':
                out_path += '.png'
                cv2.imwrite(out_path, mask)
            else:
                raise NotImplementedError

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
    parser.add_argument('--mask_way', type=str, default='warp',
                        help="Sets the way to obtain the mask. Сan take \'warp\' or \'segm\'")
    parser.add_argument('--blend', action='store_true', default=False,
                        help="Whether need to blend the mask and frame or not")
    parser.add_argument('--img2input', action='store_true', default=False,
                        help="Whether add an image to regressor input or not")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Get params:
    args = get_args()
    args.model = '/home/darkalert/builds/Court-Segm-UNet/checkpoints/NCAA2020+v2-640x360_aug-app-geo_nc4-deconv-focal-mse_pre2/CP_epoch7.pth'
    args.temp_path = '/home/darkalert/builds/Court-Segm-UNet/assets/mask_ncaa_v4_nc4_m_onehot.png'

    args.src_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/player_tracking/frames/'
    args.dst_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/player_tracking/court_mapping_focal/'

    args.bilinear = False
    args.n_classes = 4
    args.resnet = 'resnetreg18'
    args.img2input = True

    args.blend = True
    args.mask_way = 'warp'

    # Get videos:
    video_names = [n for n in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, n))]
    # video_names = ['JamalBey_0_Transition_PossessionsAndAssists_Offense_2019-2020_NCAAM']
    # video_names = ['2020_11_25_RhodeIsland_at_ArizonaState']

    # CUDA or CPU:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Load the court template:
    template = load_template(args.temp_path,
                             num_classes=args.n_classes,
                             target_size=args.output_size,
                             batch_size=1)
    template = template.to(device=device)

    # Load a Reconstructor model (UNET+RESNET+STN):
    net = Reconstructor(n_channels=args.n_channels,
                        n_classes=args.n_classes,
                        template=template,
                        target_size=args.output_size,
                        bilinear=args.bilinear,
                        resnet_name=args.resnet,
                        warp_with_nearest=True,
                        img2input=args.img2input)
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

        test(net, input_paths, out_paths,
             in_size=args.input_size, out_size=args.output_size,
             mask_way=args.mask_way, blend=args.blend)

    logging.info('All done!')