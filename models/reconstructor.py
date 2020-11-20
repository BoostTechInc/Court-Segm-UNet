""" Full assembly of the parts to form the complete network """

from unet.unet_parts import *
from models.resnet import resnet
import kornia


class Reconstructor(nn.Module):
    '''The Reconstructor model consists of the UNET, which learns
    to segment the court mask, and the Spatial Transformer Network
    (STN), which learns to predict the homography matrix based on
    the predicted segmentation mask.
    '''

    def __init__(self,
                 n_channels,
                 n_classes,
                 template,
                 target_size,
                 bilinear=True,
                 resnet_name='resnetreg50',
                 resnet_pretrained=None):
        super(Reconstructor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # UNet:
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # UNet gegressor that outputs the first 3x3 transformation matrix:
        #reg_channels = 8
        #self.conv_top = nn.Conv2d(1024 // factor, reg_channels, kernel_size=1)
        #self.unet_reg = Reconstructor.make_regressor(reg_channels)

        # ResNet regressor that outputs the second 3x3 transformation matrix:
        self.resnet_reg = resnet(resnet_name, resnet_pretrained)

        # Court template that will be warped by the learnt transformation matrix:
        self.template = template

        # STN warper:
        h, w = target_size[1], target_size[0]
        self.warper = kornia.HomographyWarper(h, w)                # mode='nearest' seems has a bug

    @staticmethod
    def make_regressor(channels):
        reg = nn.Sequential(
            nn.Linear(channels * 22 * 40, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 3)
        )
        # Initialize the weights/bias with identity transformation:
        reg[-1].weight.data.zero_()
        reg[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))

        return reg

    def warp(self, theta, template):
        '''Warp teamplate by predicted homographies'''
        bs = theta.shape[0]
        template = template[0:bs]

        warped = self.warper(template, theta)

        return warped.squeeze(1)

    def forward_unet(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_top = self.down4(x4)
        x = self.up1(x_top, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits, x_top

    def regress(self, x):
        x1 = self.conv_top(x)
        x1 = torch.flatten(x1, 1)
        theta = self.unet_reg(x1)

        # 3x3 transformation matrix:
        theta = theta.view(-1, 1, 3, 3)

        return theta

    def forward(self, x):
        # UNet:
        logits, x_top = self.forward_unet(x)

        # UNet regressor:
        #theta = self.regress(x_top)
        #rec_mask = self.warp(theta, self.template)

        # ResNet regressor:
        theta = self.resnet_reg(logits)
        rec_mask = self.warp(theta, self.template)

        return logits, rec_mask