import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                input_size,
                out_threshold=0.5):
    net.eval()

    img = BasicDataset.preprocess_img(full_img, input_size)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
            probs = torch.argmax(probs, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        probs = probs.type(torch.IntTensor)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((full_img.size[1],full_img.size[0]), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask# > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

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


def mask_to_image(mask, to_rgb=True):
    if to_rgb:
        mask = np.expand_dims(mask, 2)
        rgb_mask = np.zeros((mask.shape[0],mask.shape[1],3), dtype=np.uint8)
        rgb_mask[np.all(mask == 1, axis=2),:] = (0, 255, 0)
        rgb_mask[np.all(mask == 2, axis=2),:] = (255, 0, 0)
        rgb_mask[np.all(mask == 3, axis=2),:] = (0, 0, 255)
        rgb_mask[np.all(mask == 4, axis=2),:] = (255, 255, 255)
        mask = rgb_mask[...,::-1]

    return Image.fromarray((mask).astype(np.uint8))


def get_img_paths(src_dir, dst_dir=None):
    names = [file for file in os.listdir(src_dir) if not file.endswith('.')]
    input_paths = [os.path.join(src_dir, n) for n in names]

    if dst_dir is not None:
        output_paths = [os.path.join(dst_dir, n) for n in names]
        return input_paths, output_paths
    else:
        return input_paths


if __name__ == "__main__":
    args = get_args()
    args.model = '/home/darkalert/builds/Pytorch-UNet/checkpoints/Nov04_20-20-59_DeepLearningLR_0.0001_BS_8_SIZE_(640, 360)/checkpointsCP_epoch1.pth'
    args.src_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/frames/ElijahHardy_0_Transition_PossessionsAndAssists_Offense_2019-2020_NCAAM/'
    args.dst_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/NCAAM/preds/ElijahHardy_0_Transition_PossessionsAndAssists_Offense_2019-2020_NCAAM/'


    input_paths, output_paths = get_img_paths(args.src_dir, args.dst_dir)
    if not os.path.exists(args.dst_dir): os.makedirs(args.dst_dir)

    net = UNet(n_channels=3, n_classes=args.n_classes)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, (in_path, out_path) in enumerate(zip(input_paths, output_paths)):
        logging.info("\nPredicting image {} ...".format(in_path))

        img = Image.open(in_path)

        mask = predict_img(net=net,
                           full_img=img,
                           input_size=args.input_size,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            result = mask_to_image(mask)
            result.save(out_path)

            logging.info("Mask saved to {}".format(out_path))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(in_path))
            plot_img_and_mask(img, mask)
