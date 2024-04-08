import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Dict
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, batch_size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        grid = grid.type(torch.FloatTensor)
        grid_demo = grid
        if batch_size > 1:
            for i in range(batch_size-1):
                grid = torch.cat((grid, grid_demo), dim=0)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ConNeXt_R_Blcok(nn.Module):
    def __init__(self, dim):
        super(ConNeXt_R_Blcok, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim) # depthwise conv
        self.norm1 = nn.BatchNorm3d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.norm2 = nn.BatchNorm3d(dim)
        self.act2 = nn.GELU()
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, H, D, W) -> (B, H, D, W, C)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, H, D, W, C) -> (B, C, H, D, W)
        x = self.norm2(x)
        x = self.act2(residual + x)
        return x

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, layer_num=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(ConNeXt_R_Blcok(out_channels))
        super(Down, self).__init__(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_num=1):
        super(Up, self).__init__()
        C = in_channels // 2
        self.norm = nn.BatchNorm3d(in_channels)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.gate = nn.Linear(C, 3 * C)
        self.linear1 = nn.Linear(C, C)
        self.linear2 = nn.Linear(C, C)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(ConNeXt_R_Blcok(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.norm(x1)
        x1 = self.up(x1)
        # [B, C, H, D, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        #attention
        B, C, H, D, W = x1.shape
        x1 = x1.permute(0, 2, 3, 4, 1)
        x2 = x2.permute(0, 2, 3, 4, 1)
        gate = self.gate(x1).reshape(B, H, D, W, 3, C).permute(4, 0, 1, 2, 3, 5)
        g1, g2, g3 = gate[0], gate[1], gate[2]
        x2 = torch.sigmoid(self.linear1(g1 + x2)) * x2 + torch.sigmoid(g2) * torch.tanh(g3)
        x2 = self.linear2(x2)
        x1 = x1.permute(0, 4, 1, 2, 3)
        x2 = x2.permute(0, 4, 1, 2, 3)

        x = self.conv1x1(torch.cat([x2, x1], dim=1))
        x = self.conv(x)
        return x


class ConvNeXt_R_Encoder_Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 bilinear,
                 base_c):
        super(ConvNeXt_R_Encoder_Decoder, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_c, kernel_size=3, stride=1, padding=1),
            # nn.Conv3d(in_channels, base_c, kernel_size=7, stride=1 ,padding=3),
            nn.BatchNorm3d(base_c),
            nn.GELU(),
            ConNeXt_R_Blcok(base_c)
        )
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8, layer_num=3)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x
        # return logits

class ConvNXMorph(nn.Module):
    """
    ConvUNeXtMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        img_size,
        batch_size,
        in_channels: int = 3,
        bilinear: bool = False,
        base_c: int = 16,
        use_probs=False):
        """
        Parameters:
            img_size: Input shape. e.g. (192, 192, 192)
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(img_size)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.ConvUNeXt = ConvNeXt_R_Encoder_Decoder(
            in_channels = in_channels,
            bilinear = bilinear,
            base_c = base_c
        )

        # configure unet to flow field layer
        self.flow = nn.Conv3d(base_c, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(img_size, batch_size)

    def forward(self, x: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.ConvUNeXt(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow